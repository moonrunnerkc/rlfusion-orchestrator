# Author: Bradley R. Kinnard
# main.py - FastAPI backend for RLFusion multi-source retrieval with RL fusion
# Originally built for personal offline use, now open-sourced for public benefit.

import json
import logging
import os
import re
import sys
import types
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

warnings.filterwarnings("ignore", message="Gym has been unmaintained")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

import numpy as np
import torch
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from ollama import Client
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.config import cfg, PROJECT_ROOT
from backend.core.critique import critique, log_episode_to_replay_buffer, get_critique_instruction, strip_critique_block, check_safety
from backend.core.fusion import fuse_context
from backend.core.memory import expand_query_with_context, record_turn, get_context_for_prompt, clear_memory
from backend.core.profile import detect_and_save_memory, get_user_profile
from backend.core.retrievers import get_rag_index, retrieve
from backend.core.utils import embed_text

# Max upload size per file (10 MB)
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Max query length (characters) to prevent prompt stuffing
_MAX_QUERY_LEN = 4000

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

# numpy._core shim for stable-baselines3
if not hasattr(np, '_core'):
    _core_module = types.ModuleType('numpy._core')
    _core_module.numeric = np
    sys.modules['numpy._core'] = _core_module
    sys.modules['numpy._core.numeric'] = np

# Device configuration - supports both GPU and CPU
USE_CUDA = torch.cuda.is_available() and os.environ.get("RLFUSION_FORCE_CPU", "").lower() != "true"
if USE_CUDA:
    torch.cuda.empty_cache()
    torch.set_default_device("cuda")
    device = torch.device("cuda")
    logger.info(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    logger.info("Running on CPU")

_rl_policy = None
_boot_id: str = ""


class CQLPolicyWrapper:
    """Wrapper for CQL policy weights loaded from d3rlpy checkpoint."""
    def __init__(self, policy_weights):
        import torch.nn as nn
        dev = "cuda" if USE_CUDA else "cpu"
        self.encoder = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()).to(dev)
        self.mu = nn.Linear(256, 4).to(dev)
        self.encoder[0].weight.data = policy_weights['_encoder._layers.0.weight']
        self.encoder[0].bias.data = policy_weights['_encoder._layers.0.bias']
        self.encoder[2].weight.data = policy_weights['_encoder._layers.2.weight']
        self.encoder[2].bias.data = policy_weights['_encoder._layers.2.bias']
        self.mu.weight.data = policy_weights['_mu.weight']
        self.mu.bias.data = policy_weights['_mu.bias']
        self.encoder.eval()
        self.mu.eval()
        self._device = dev

    def predict(self, obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self._device)
            return torch.clamp(self.mu(self.encoder(obs_t)), -1.0, 1.0).cpu().numpy()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rl_policy, _boot_id
    _boot_id = uuid.uuid4().hex[:12]

    # ── Ollama health check ──────────────────────────────────
    ollama_host = cfg["llm"]["host"]
    ollama_model = cfg["llm"]["model"]
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_host}/api/tags")
            resp.raise_for_status()
            available_models = [m["name"] for m in resp.json().get("models", [])]
            # Check for base model name match (ignore tag suffixes)
            model_base = ollama_model.split(":")[0]
            if not any(model_base in m for m in available_models):
                logger.warning(
                    f"Ollama is running but model '{ollama_model}' not found. "
                    f"Available: {available_models}. "
                    f"Pull it with: ollama pull {ollama_model}"
                )
            else:
                logger.info(f"Ollama health check passed — model '{ollama_model}' available")
    except Exception as e:
        logger.error(
            f"Ollama is not reachable at {ollama_host}: {e}\n"
            f"  → Start Ollama first:  ollama serve\n"
            f"  → Then pull the model: ollama pull {ollama_model}\n"
            f"  → The server will start but queries will fail until Ollama is available."
        )

    logger.info("Initializing RAG index...")
    get_rag_index()

    cql_path = PROJECT_ROOT / "models" / "rl_policy_cql.d3"
    ppo_path = PROJECT_ROOT / "rl_policy.zip"
    policy_status = "not trained"
    map_location = "cuda" if USE_CUDA else "cpu"

    if cql_path.exists():
        logger.info("Loading CQL policy...")
        try:
            state_dict = torch.load(str(cql_path), weights_only=False, map_location=map_location)
            _rl_policy = CQLPolicyWrapper(state_dict['policy'])
            policy_status = "CQL loaded"
            logger.info("CQL policy loaded successfully")
        except Exception as e:
            logger.error(f"CQL load failed: {e}")
    elif ppo_path.exists():
        logger.info("Loading PPO policy (fallback)...")
        try:
            from stable_baselines3 import PPO
            _rl_policy = PPO.load(str(ppo_path), device=map_location)
            policy_status = "PPO loaded"
        except Exception as e:
            logger.error(f"PPO load failed: {e}")

    # Check web search status
    web_enabled = cfg.get("web", {}).get("enabled", False)
    web_api_key = os.environ.get("TAVILY_API_KEY", "")
    if web_enabled and web_api_key:
        web_status = "enabled (Tavily connected)"
    elif web_enabled:
        web_status = "enabled but TAVILY_API_KEY not set"
    else:
        web_status = "disabled"

    logger.info("=" * 64)
    logger.info("RLFUSION ORCHESTRATOR - BACKEND STARTED")
    if USE_CUDA:
        logger.info(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        logger.info("Device: CPU")
    logger.info(f"LLM: {cfg['llm']['model']}")
    logger.info(f"RL Policy: {policy_status}")
    logger.info(f"Web Search: {web_status}")
    logger.info("=" * 64)
    yield
    logger.info("Shutting down RLFusion Orchestrator")


app = FastAPI(title="RLFusion Orchestrator", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _apply_markdown_formatting(text: str) -> str:
    text = re.sub(r'\[RAG:[^\]]+\]\s*', '', text)
    text = re.sub(r'\[CAG:[^\]]+\]\s*', '', text)
    text = re.sub(r'\[GRAPH:[^\]]+\]\s*', '', text)
    text = re.sub(r'\[WEB:[^\]]+\]\s*', '', text)
    text = re.sub(r'([^\n])(#{1,3}\s)', r'\1\n\n\2', text)
    text = re.sub(r'([.!?:])\s*(-\s)', r'\1\n\n\2', text)
    text = re.sub(r'([.!?])\s+([A-Z#])', r'\1\n\n\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _compute_rl_fusion_weights(query: str, policy) -> np.ndarray:
    web_enabled = cfg.get("web", {}).get("enabled", False)

    if policy is not None:
        try:
            obs = embed_text(query).reshape(1, -1)
            action = policy.predict(obs)[0] if hasattr(policy, 'predict') else policy.predict(obs, deterministic=True)[0]
            weights = action.flatten() if hasattr(action, 'flatten') else np.array(action)
            if len(weights) == 3:
                weights = np.append(weights, 0.0)

            exp_w = np.exp(weights)
            rl_weights = exp_w / np.sum(exp_w)

            if max(rl_weights) - min(rl_weights) < 0.05:
                logger.info("Policy outputs uniform, applying query heuristics")
                q = query.lower()
                if any(kw in q for kw in ['http://', 'https://', 'website', '.com', 'look up']):
                    rl_weights = np.array([0.2, 0.1, 0.2, 0.5])
                elif any(kw in q for kw in ['how does', 'architecture', 'design', 'workflow', 'system']):
                    rl_weights = np.array([0.2, 0.1, 0.6, 0.1])
                elif any(kw in q for kw in ['what is', 'explain', 'describe', 'document']):
                    rl_weights = np.array([0.6, 0.2, 0.1, 0.1])
                else:
                    rl_weights = np.array([0.4, 0.2, 0.3, 0.1])

            if not web_enabled:
                rl_weights[3] = 0.0
                if np.sum(rl_weights[:3]) > 0:
                    rl_weights[:3] = rl_weights[:3] / np.sum(rl_weights[:3])

            logger.info(f"RL weights: RAG={rl_weights[0]:.2f} CAG={rl_weights[1]:.2f} Graph={rl_weights[2]:.2f} Web={rl_weights[3]:.2f}")
            return rl_weights
        except Exception as e:
            logger.warning(f"Policy prediction failed: {e}")

    return np.array([0.33, 0.33, 0.34, 0.0]) if not web_enabled else np.array([0.25, 0.25, 0.25, 0.25])


def _build_fusion_context(retrieval_results: Dict[str, List[Dict[str, Any]]], weights: np.ndarray) -> str:
    total_items = 15
    rag_take = max(2, int(weights[0] * total_items))
    cag_take = max(1, int(weights[1] * total_items))
    graph_take = max(1, int(weights[2] * total_items))
    web_take = max(1, int(weights[3] * total_items)) if len(weights) > 3 else 0

    rag_items = [r for r in retrieval_results["rag"] if r["score"] >= 0.40][:rag_take]
    cag_items = [c for c in retrieval_results["cag"] if c["score"] >= 0.85][:cag_take]
    graph_items = [g for g in retrieval_results["graph"] if g["score"] >= 0.50][:graph_take]
    web_items = [w for w in retrieval_results.get("web", []) if w["score"] >= 0.60][:web_take] if web_take > 0 else []

    parts = []
    for r in rag_items:
        parts.append(f"[RAG:{r['score']:.2f}|w={weights[0]:.2f}] {r['text']}")
    for c in cag_items:
        parts.append(f"[CAG:{c['score']:.2f}|w={weights[1]:.2f}] {c['text']}")
    for g in graph_items:
        parts.append(f"[GRAPH:{g['score']:.2f}|w={weights[2]:.2f}] {g['text']}")
    for w in web_items:
        parts.append(f"[WEB:{w['score']:.2f}|w={weights[3]:.2f}] Source: {w.get('url', 'unknown')}\n{w['text'][:600]}")

    logger.info(f"Fusion stats: {len(rag_items)} RAG, {len(cag_items)} CAG, {len(graph_items)} Graph, {len(web_items)} Web")
    return "\n\n".join(parts) if parts else "No high-confidence sources available."


IDENTITY_BLOCK = """IDENTITY RULES:
- "I/me/my" (when YOU speak) = RLFusion AI assistant
- "you/your" (when USER addresses you) = RLFusion AI
- "I/me/my" (when USER speaks) = The human user
- Personal details in context (name, pets, job) = THE USER's info, not yours
- You are software, you don't have pets/vehicles/personal life
"""

WEB_INSTRUCTION = """
WEB RESULTS: [WEB:...] entries are live internet search. Prioritize them for current events, businesses, prices.
Cite source URLs. Ignore unrelated [RAG:...] results.
"""


def _generate_system_prompt(mode: str, context_parts: List[str]) -> str:
    has_cag_hit = any(p.startswith("[CAG:") for p in context_parts)
    cag_only = has_cag_hit and len([p for p in context_parts if p.startswith("[CAG:")]) == len(context_parts)
    has_web = any("[WEB:" in p for p in context_parts)
    critique_suffix = get_critique_instruction()

    if cag_only:
        return "Return the exact text after [CAG:] with no changes."

    if mode == "build":
        return f"""{IDENTITY_BLOCK}
You are an elite AI architect. Be INNOVATIVE, SPECIFIC, CUTTING-EDGE.
Never output [RAG:...], [CAG:...], [GRAPH:...], [WEB:...] tags.
{critique_suffix}"""

    web_block = WEB_INSTRUCTION if has_web else ""
    return f"""{IDENTITY_BLOCK}{web_block}
You are a document-grounded assistant. Answer using ALL relevant context.
Never output source tags. Reference sources naturally.
RULES:
1. Read entire context - synthesize complete answer
2. Quote specific details from context
3. Only say "I don't have that" if context truly lacks info
4. YOU are RLFusion (AI), USER is the human
5. Ignore irrelevant context sections
{critique_suffix}"""


def _generate_user_prompt(mode: str, query: str, fused_context: str, context_parts: List[str]) -> str:
    has_cag_hit = any(p.startswith("[CAG:") for p in context_parts)
    cag_only = has_cag_hit and len([p for p in context_parts if p.startswith("[CAG:")]) == len(context_parts)

    if cag_only:
        return f"Sources:\n{fused_context}\n\nReturn ONLY the text after [CAG:] - exact copy."

    if mode == "build":
        return f"""Knowledge sources:\n{fused_context}\n\nRequest: {query}\n\nSynthesize innovative concept:"""

    return f"CONTEXT:\n{fused_context}\n\nQUESTION: {query}\n\nANSWER:"


@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, body: Dict[str, Any]) -> Dict[str, Any]:
    query = body.get("query", "").strip()
    mode = body.get("mode", "chat")

    if not query:
        return {"response": "Please provide a query.", "fusion_weights": {"rag": 0, "cag": 0, "graph": 0}, "reward": 0.0}
    if len(query) > _MAX_QUERY_LEN:
        return {"response": f"Query too long ({len(query)} chars, max {_MAX_QUERY_LEN}).", "fusion_weights": {"rag": 0, "cag": 0, "graph": 0}, "reward": 0.0}

    # Safety gate - same as /ws path
    safe, safety_reason = check_safety(query)
    if not safe:
        logger.warning("Query blocked by safety filter: %s", safety_reason)
        return {
            "response": "I can't help with that request. " + safety_reason,
            "fusion_weights": {"rag": 0, "cag": 0, "graph": 0},
            "reward": 0.0,
            "blocked": True,
            "safety_reason": safety_reason,
        }

    get_rag_index()
    rl_weights = _compute_rl_fusion_weights(query, _rl_policy)
    retrieval_results = retrieve(query)
    fused_context = _build_fusion_context(retrieval_results, rl_weights)

    client = Client(host=cfg["llm"]["host"])
    context_parts = fused_context.split("\n\n")

    response_obj = client.chat(
        model=cfg["llm"]["model"],
        messages=[
            {"role": "system", "content": _generate_system_prompt(mode, context_parts)},
            {"role": "user", "content": _generate_user_prompt(mode, query, fused_context, context_parts)}
        ],
        options={"temperature": 0.3}
    )

    generated = _apply_markdown_formatting(response_obj["message"]["content"])
    critique_result = critique(query, fused_context, generated)
    clean_response = strip_critique_block(generated)

    return {
        "response": clean_response,
        "fusion_weights": {"rag": float(rl_weights[0]), "cag": float(rl_weights[1]), "graph": float(rl_weights[2])},
        "reward": critique_result["reward"],
        "proactive_suggestions": critique_result.get("proactive_suggestions", [])
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(id(websocket))

    try:
        while True:
            data = await websocket.receive_text()
            try:
                request = json.loads(data)
                if "query" not in request and len(request.keys()) == 1:
                    payload = list(request.values())[0]
                    query = payload.get("query", payload.get("message", ""))
                    mode = payload.get("mode", "chat")
                else:
                    query = request.get("query", request.get("message", ""))
                    mode = request.get("mode", "chat")

                if request.get("clear_memory") or request.get("new_chat"):
                    clear_memory(session_id)
                    await websocket.send_json({"type": "memory_cleared"})
                    continue
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                await websocket.close()
                return

            query = query.strip()
            if not query:
                await websocket.send_json({"type": "done", "response": "Please provide a query."})
                continue
            if len(query) > _MAX_QUERY_LEN:
                await websocket.send_json({"type": "done", "response": f"Query too long ({len(query)} chars, max {_MAX_QUERY_LEN})."})
                continue

            await websocket.send_json({"type": "start"})

            # Expand query with conversation context
            expanded_query, expansion_meta = expand_query_with_context(session_id, query)
            if expansion_meta["expanded"]:
                logger.info(f"[MEMORY] Query expanded ({len(query)} → {len(expanded_query)} chars)")

            # Check for memory storage request
            is_memory_request, memory_content = detect_and_save_memory(query)
            if is_memory_request:
                preview = memory_content[:100] + "..." if len(memory_content) > 100 else memory_content
                await websocket.send_json({"type": "token", "token": f"✅ **Remembered:**\n\n> {preview}"})
                await websocket.send_json({"type": "done"})
                continue

            # Safety gate — block unsafe queries before retrieval
            safe, safety_reason = check_safety(query)
            if not safe:
                logger.warning("Query blocked by safety filter: %s", safety_reason)
                await websocket.send_json({
                    "type": "done",
                    "response": "I can't help with that request. " + safety_reason,
                    "fusion_weights": {"rag": 0, "cag": 0, "graph": 0},
                    "reward": 0.0,
                    "blocked": True,
                    "safety_reason": safety_reason,
                })
                continue

            record_turn(session_id, "user", query)
            rl_weights = _compute_rl_fusion_weights(expanded_query, _rl_policy)
            retrieval_results = retrieve(expanded_query, top_k=10)
            web_status = retrieval_results.get("web_status", "disabled")
            fused_context = _build_fusion_context(retrieval_results, rl_weights)
            actual_weights = [float(w) for w in rl_weights[:4]] if len(rl_weights) >= 4 else [float(w) for w in rl_weights] + [0.0]

            # Inject user profile for personal queries
            if any(w in query.lower() for w in ['my ', 'me ', 'i ', 'brad', 'preference', 'like', 'favorite']):
                profile = get_user_profile()
                if profile:
                    fused_context = profile + "\n" + fused_context

            # Inject conversation context
            conv_context = get_context_for_prompt(session_id)
            if conv_context:
                fused_context = conv_context + "\n\n" + fused_context

            context_parts = fused_context.split("\n\n")
            system_prompt = _generate_system_prompt(mode, context_parts)
            prompt = _generate_user_prompt(mode, query, fused_context, context_parts)
            logger.info(f"Context length: {len(fused_context)} chars")

            client = Client(host=cfg["llm"]["host"])
            full_response = ""

            for chunk in client.chat(
                model=cfg["llm"]["model"],
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_ctx": 8192},
                stream=True
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    text_chunk = chunk['message']['content']
                    full_response += text_chunk
                    await websocket.send_json({"chunk": text_chunk, "weights": actual_weights, "reward": 0.85})

            full_response = _apply_markdown_formatting(full_response)

            critique_result = critique(query, fused_context, full_response)
            clean_response = strip_critique_block(full_response)

            # Add web search notice if API key is missing (after stripping critique)
            if web_status == "no_api_key":
                logger.warning("⚠️ Adding web search API key notice to response")
                web_notice = "\n\n---\n⚠️ **Web search is enabled but no API key is configured.** To enable web search, set `TAVILY_API_KEY` in your `.env` file. Get a free key at [tavily.com](https://tavily.com)."
                clean_response += web_notice

            record_turn(session_id, "assistant", clean_response)

            log_episode_to_replay_buffer({
                "query": query, "response": clean_response,
                "weights": {"rag": actual_weights[0], "cag": actual_weights[1], "graph": actual_weights[2]},
                "reward": critique_result["reward"],
                "proactive_suggestions": critique_result.get("proactive_suggestions", []),
                "fused_context": fused_context
            })

            await websocket.send_json({
                "type": "done", "response": clean_response,
                "fusion_weights": {"rag": actual_weights[0], "cag": actual_weights[1], "graph": actual_weights[2]},
                "reward": critique_result["reward"],
                "proactive": critique_result.get("proactive_suggestions", [""])[0] if critique_result.get("proactive_suggestions") else "",
                "proactive_suggestions": critique_result.get("proactive_suggestions", []),
                "query_expanded": expansion_meta["expanded"],
                "expanded_query": expanded_query if expansion_meta["expanded"] else None,
                "web_status": web_status
            })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


@app.get("/api/config")
@limiter.limit("10/minute")
async def get_config(request: Request) -> Dict[str, Any]:
    return {"web": {"enabled": cfg.get("web", {}).get("enabled", False),
                    "max_results": cfg.get("web", {}).get("max_results", 3),
                    "search_timeout": cfg.get("web", {}).get("search_timeout", 10)}}


@app.patch("/api/config")
@limiter.limit("10/minute")
async def update_config(request: Request, body: Dict[str, Any]) -> Dict[str, Any]:
    if "web" in body and "enabled" in body["web"]:
        cfg["web"]["enabled"] = body["web"]["enabled"]
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r') as f:
            current_config = yaml.safe_load(f)
        current_config["web"]["enabled"] = cfg["web"]["enabled"]
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False, sort_keys=False)
        return {"status": "updated", "web": {"enabled": cfg["web"]["enabled"]}}
    return {"error": "Invalid config update"}


@app.get("/ping")
@limiter.limit("10/minute")
async def ping(request: Request) -> Dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "status": "alive",
        "gpu": gpu_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model": cfg["llm"]["model"],
        "policy": "CQL" if Path(cfg["rl"]["policy_path"]).exists() else "heuristic",
        "policy_exists": Path(cfg["rl"]["policy_path"]).exists(),
        "boot_id": _boot_id,
    }


@app.post("/api/upload")
@limiter.limit("10/minute")
async def upload_documents(request: Request) -> Dict[str, Any]:
    """Upload files to data/docs/ for RAG indexing."""
    from fastapi import UploadFile
    import shutil

    form = await request.form()
    files = form.getlist("files")
    if not files:
        return {"status": "error", "message": "No files provided"}

    docs_path = PROJECT_ROOT / "data" / "docs"
    docs_path.mkdir(parents=True, exist_ok=True)

    saved, skipped = [], []
    allowed = {".txt", ".md", ".pdf"}

    for f in files:
        if not hasattr(f, 'filename'):
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in allowed:
            skipped.append(f.filename)
            continue
        content = await f.read()
        if len(content) > _MAX_UPLOAD_BYTES:
            skipped.append(f"{f.filename} (>{_MAX_UPLOAD_BYTES // (1024*1024)}MB)")
            continue
        dest = docs_path / Path(f.filename).name  # strip path components
        # double-check: reject filenames with traversal attempts
        if '..' in dest.name or dest.name.startswith('.'):
            skipped.append(f"{f.filename} (invalid filename)")
            continue
        dest.write_bytes(content)
        saved.append(f.filename)
        logger.info("Uploaded: %s (%d bytes)", f.filename, len(content))

    return {
        "status": "uploaded",
        "saved": saved,
        "skipped": skipped,
        "total_saved": len(saved),
        "total_skipped": len(skipped),
    }


@app.post("/api/reindex")
@limiter.limit("3/minute")
async def reindex_documents(request: Request) -> Dict[str, Any]:
    """Rebuild the RAG FAISS index from documents in data/docs/."""
    from backend.core.retrievers import build_rag_index, _get_docs_path, _get_metadata_path
    import time

    docs_path = _get_docs_path()
    supported = list(docs_path.rglob("*.txt")) + list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.pdf"))

    if not supported:
        return {
            "status": "empty",
            "message": "No documents found. Add .txt, .md, or .pdf files to data/docs/ and try again.",
            "docs_path": str(docs_path),
        }

    t0 = time.time()
    index = build_rag_index()
    elapsed = round(time.time() - t0, 2)

    meta_path = _get_metadata_path()
    chunk_count = len(json.loads(meta_path.read_text())) if meta_path.exists() else 0

    logger.info(f"Reindex complete: {len(supported)} files, {chunk_count} chunks, {elapsed}s")
    return {
        "status": "reindexed",
        "files_processed": len(supported),
        "chunks_indexed": chunk_count,
        "elapsed_seconds": elapsed,
    }


@app.delete("/api/reset")
@limiter.limit("5/minute")
async def reset_state(request: Request) -> Dict[str, Any]:
    """Wipe all transient state: cache, episodes, replay, conversations."""
    import sqlite3
    db_path = PROJECT_ROOT / cfg["paths"]["db"]
    if not db_path.exists():
        return {"status": "no database"}
    conn = sqlite3.connect(str(db_path))
    _RESET_TABLES = ("cache", "episodes", "replay", "conversations")
    for table in _RESET_TABLES:
        # table names are hardcoded constants above, not user input
        conn.execute(f"DELETE FROM {table}")  # nosec: B608
    conn.commit()
    conn.close()
    # clear in-memory conversation state
    from backend.core.memory import conversation_memory
    conversation_memory.clear_all_sessions()
    logger.info("Full state reset via /api/reset")
    return {"status": "reset", "tables_cleared": ["cache", "episodes", "replay", "conversations"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

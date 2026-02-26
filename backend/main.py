# Author: Bradley R. Kinnard
# main.py - FastAPI backend for RLFusion multi-source retrieval with RL fusion
# Originally built for personal offline use, now open-sourced for public benefit.

import asyncio
import json
import logging
import os
import re
import sys
import time as _time_mod
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
from fastapi.responses import Response
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST,
)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.config import cfg, PROJECT_ROOT
from backend.core.model_router import get_engine
from backend.core.critique import critique, log_episode_to_replay_buffer, get_critique_instruction, strip_critique_block, check_safety
from backend.core.critique import should_route_to_stis
from backend.core.stis_client import request_stis_consensus, log_stis_resolution
from backend.core.fusion import fuse_context
from backend.core.memory import expand_query_with_context, record_turn, get_context_for_prompt, clear_memory
from backend.core.profile import detect_and_save_memory, get_user_profile
from backend.core.retrievers import get_rag_index, retrieve
from backend.core.utils import embed_text
from backend.agents.orchestrator import Orchestrator, apply_markdown_formatting

# Max upload size per file (10 MB)
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Max query length (characters) to prevent prompt stuffing
_MAX_QUERY_LEN = 4000

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

# ── Prometheus metrics ────────────────────────────────────────────────────
QUERY_LATENCY = Histogram(
    "rlfusion_query_latency_seconds",
    "End-to-end query processing latency",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60),
)
RETRIEVAL_PATH_USAGE = Counter(
    "rlfusion_retrieval_path_total",
    "Number of times each retrieval path returned results",
    ["path"],
)
FUSION_WEIGHT_DIST = Histogram(
    "rlfusion_fusion_weight",
    "Distribution of fusion weights per retrieval path",
    ["path"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
SAFETY_GATE_TRIGGERS = Counter(
    "rlfusion_safety_gate_triggers_total",
    "Number of queries blocked by safety gate",
)
REPLAY_BUFFER_SIZE = Gauge(
    "rlfusion_replay_buffer_size",
    "Number of episodes in the replay buffer",
)
CRITIQUE_REWARD = Histogram(
    "rlfusion_critique_reward",
    "Distribution of critique reward scores",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
REQUESTS_TOTAL = Counter(
    "rlfusion_http_requests_total",
    "Total HTTP requests by endpoint and method",
    ["endpoint", "method"],
)
WS_CONNECTIONS = Gauge(
    "rlfusion_ws_connections_active",
    "Number of active WebSocket connections",
)
STIS_ROUTING_EVENTS = Counter(
    "rlfusion_stis_routing_total",
    "Number of queries routed to STIS engine due to contradiction",
    ["outcome"],  # resolved, failed, skipped
)

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
_orchestrator: Orchestrator | None = None


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

    # ── Inference engine health check ────────────────────────
    engine = get_engine()
    try:
        if engine.check_health():
            logger.info(
                "Inference health check passed: engine=%s, model=%s",
                engine.engine, engine.model,
            )
        else:
            logger.warning(
                "Inference engine '%s' is running but model '%s' not found. "
                "Queries will fail until the model is available.",
                engine.engine, engine.model,
            )
    except Exception as e:
        logger.error(
            "Inference engine not reachable at %s: %s\n"
            "  Configured engine: %s\n"
            "  Configured model:  %s\n"
            "  The server will start but queries will fail until the engine is available.",
            engine.base_url, e, engine.engine, engine.model,
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

    # Initialize multi-agent orchestrator (Phase 1)
    global _orchestrator
    _orchestrator = Orchestrator(rl_policy=_rl_policy)
    logger.info("Orchestrator: multi-agent pipeline active (LangGraph)")
    logger.info("=" * 64)
    yield
    logger.info("Shutting down RLFusion Orchestrator")


app = FastAPI(title="RLFusion Orchestrator", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: object) -> Response:
    """Inject correlation ID into every request for structured log tracing."""
    corr_header = cfg.get("monitoring", {}).get("correlation_id_header", "X-Correlation-ID")
    corr_id = request.headers.get(corr_header, uuid.uuid4().hex[:12])
    request.state.correlation_id = corr_id

    endpoint = request.url.path
    method = request.method
    REQUESTS_TOTAL.labels(endpoint=endpoint, method=method).inc()

    t0 = _time_mod.perf_counter()
    response = await call_next(request)
    elapsed = _time_mod.perf_counter() - t0

    response.headers[corr_header] = corr_id
    logger.info(
        "request.complete",
        extra={"corr_id": corr_id, "path": endpoint, "method": method, "status": response.status_code, "took_ms": round(elapsed * 1000, 1)},
    )
    return response


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    # update replay buffer gauge on each scrape
    try:
        import sqlite3
        db_path = PROJECT_ROOT / cfg["paths"]["db"]
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            row = conn.execute("SELECT COUNT(*) FROM replay").fetchone()
            REPLAY_BUFFER_SIZE.set(row[0] if row else 0)
            conn.close()
    except Exception:
        pass  # db might not exist yet
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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

    rag_items = [r for r in retrieval_results["rag"] if r["score"] >= 0.50][:rag_take]
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


# Import prompts from orchestrator (single source of truth)
from backend.agents.orchestrator import (
    generate_system_prompt as _generate_system_prompt,
    generate_user_prompt as _generate_user_prompt,
    NUM_PREDICT,
)


@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, body: Dict[str, Any]) -> Dict[str, Any]:
    query = body.get("query", "").strip()
    mode = body.get("mode", "chat")

    if not query:
        return {"response": "Please provide a query.", "fusion_weights": {"rag": 0, "cag": 0, "graph": 0}, "reward": 0.0}
    if len(query) > _MAX_QUERY_LEN:
        return {"response": f"Query too long ({len(query)} chars, max {_MAX_QUERY_LEN}).", "fusion_weights": {"rag": 0, "cag": 0, "graph": 0}, "reward": 0.0}

    # Multi-agent orchestration (Phase 1)
    t_start = _time_mod.perf_counter()
    result = _orchestrator.run(query=query, mode=mode, session_id="chat")
    QUERY_LATENCY.observe(_time_mod.perf_counter() - t_start)

    # record fusion weight distribution
    fw = result["fusion_weights"]
    for path_name in ("rag", "cag", "graph"):
        FUSION_WEIGHT_DIST.labels(path=path_name).observe(fw.get(path_name, 0))
    CRITIQUE_REWARD.observe(result["reward"])

    if result["blocked"]:
        SAFETY_GATE_TRIGGERS.inc()

    response: Dict[str, Any] = {
        "response": result["response"],
        "fusion_weights": result["fusion_weights"],
        "reward": result["reward"],
        "proactive_suggestions": result["proactive_suggestions"],
    }
    if result["blocked"]:
        response["blocked"] = True
        response["safety_reason"] = result["safety_reason"]
    return response


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    WS_CONNECTIONS.inc()
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

            _t0 = _time_mod.perf_counter()

            # ---- Granular agent pipeline with per-step status updates ----
            # Each blocking agent call runs in a thread so WS frames flush
            # between steps, giving the frontend real-time status transitions.

            # Step 0: Preprocess (memory expansion, complexity classification)
            pre = await asyncio.to_thread(_orchestrator.step_preprocess, query, session_id)
            expanded_query = pre["expanded_query"]
            query_expanded = pre["query_expanded"]
            complexity = pre["complexity"]

            # Memory request: short-circuit before any agents run
            if pre["is_memory_request"]:
                await websocket.send_json({"type": "pipeline", "agents": [
                    {"name": "safety", "status": "skipped", "detail": "Memory request"},
                    {"name": "retrieval", "status": "skipped", "detail": ""},
                    {"name": "fusion", "status": "skipped", "detail": ""},
                    {"name": "generation", "status": "skipped", "detail": ""},
                ]})
                preview = str(pre["memory_content"])[:100]
                if len(str(pre["memory_content"])) > 100:
                    preview += "..."
                await websocket.send_json({"type": "token", "token": f"\u2705 **Remembered:**\n\n> {preview}"})
                await websocket.send_json({"type": "done"})
                continue

            # Step 1: Safety gate
            await websocket.send_json({"type": "pipeline", "agents": [
                {"name": "safety", "status": "running", "detail": "Scanning attack patterns + OOD check"},
                {"name": "retrieval", "status": "pending", "detail": ""},
                {"name": "fusion", "status": "pending", "detail": ""},
                {"name": "generation", "status": "pending", "detail": ""},
            ]})
            safety_result = await asyncio.to_thread(_orchestrator.step_safety, query, complexity)

            if safety_result.get("blocked", False):
                SAFETY_GATE_TRIGGERS.inc()
                block_reason = safety_result.get("safety_reason", "Unknown")
                logger.warning("Query blocked by safety filter: %s", block_reason)
                await websocket.send_json({"type": "pipeline", "agents": [
                    {"name": "safety", "status": "blocked", "detail": f"Blocked: {block_reason[:60]}"},
                    {"name": "retrieval", "status": "skipped", "detail": ""},
                    {"name": "fusion", "status": "skipped", "detail": ""},
                    {"name": "generation", "status": "skipped", "detail": ""},
                ]})
                await websocket.send_json({
                    "type": "done",
                    "response": "I can't help with that request. " + block_reason,
                    "fusion_weights": {"rag": 0, "cag": 0, "graph": 0},
                    "reward": 0.0,
                    "blocked": True,
                    "safety_reason": block_reason,
                })
                continue

            safety_detail = f"Passed ({complexity} query)"
            _t_safety = _time_mod.perf_counter()
            logger.info("[TIMING] safety: %.0f ms", (_t_safety - _t0) * 1000)

            # Step 2: Retrieval
            await websocket.send_json({"type": "pipeline", "agents": [
                {"name": "safety", "status": "done", "detail": safety_detail},
                {"name": "retrieval", "status": "running", "detail": f"Querying all paths (depth: {complexity})"},
                {"name": "fusion", "status": "pending", "detail": ""},
                {"name": "generation", "status": "pending", "detail": ""},
            ]})
            retrieval_result = await asyncio.to_thread(
                _orchestrator.step_retrieval,
                query=query,
                expanded_query=expanded_query,
                query_expanded=query_expanded,
                complexity=complexity,
            )

            # summarize retrieval results for the detail text
            rr = retrieval_result.get("retrieval_results", {})
            rag_n = len(rr.get("rag", []))
            cag_n = len(rr.get("cag", []))
            graph_n = len(rr.get("graph", []))
            web_n = len(rr.get("web", []))
            retrieval_detail = f"{rag_n} RAG, {cag_n} CAG, {graph_n} Graph, {web_n} Web"
            _t_retrieval = _time_mod.perf_counter()
            logger.info("[TIMING] retrieval: %.0f ms", (_t_retrieval - _t_safety) * 1000)

            # record retrieval path usage
            for path_label, count in [("rag", rag_n), ("cag", cag_n), ("graph", graph_n), ("web", web_n)]:
                if count > 0:
                    RETRIEVAL_PATH_USAGE.labels(path=path_label).inc()

            # Step 3: Fusion
            await websocket.send_json({"type": "pipeline", "agents": [
                {"name": "safety", "status": "done", "detail": safety_detail},
                {"name": "retrieval", "status": "done", "detail": retrieval_detail},
                {"name": "fusion", "status": "running", "detail": "Computing RL policy weights"},
                {"name": "generation", "status": "pending", "detail": ""},
            ]})
            fusion_result = await asyncio.to_thread(
                _orchestrator.step_fusion,
                query=query,
                expanded_query=expanded_query,
                retrieval_results=retrieval_result.get("retrieval_results", {}),
            )

            # Build prompts (fast, no status update needed)
            actual_weights = fusion_result.get("actual_weights", [0.25, 0.25, 0.25, 0.25])
            web_status = retrieval_result.get("web_status", "disabled")
            fused_context = fusion_result.get("fused_context", "")

            # summarize fusion weights for detail text
            w = actual_weights
            fusion_detail = f"RAG {w[0]*100:.0f}% CAG {w[1]*100:.0f}% Graph {w[2]*100:.0f}%"
            if len(w) > 3 and w[3] > 0.01:
                fusion_detail += f" Web {w[3]*100:.0f}%"
            _t_fusion = _time_mod.perf_counter()
            logger.info("[TIMING] fusion: %.0f ms", (_t_fusion - _t_retrieval) * 1000)

            # Step 3.5: STIS contradiction check
            stis_routed = False
            stis_response = None
            gen_detail = ""
            full_response = ""
            rr = retrieval_result.get("retrieval_results", {})
            if cfg.get("stis", {}).get("enabled", False):
                stis_decision = await asyncio.to_thread(
                    should_route_to_stis,
                    rr.get("rag", []),
                    rr.get("graph", []),
                    query=query,
                )
                if stis_decision["route_to_stis"]:
                    contradiction = stis_decision["contradiction"]
                    await websocket.send_json({"type": "pipeline", "agents": [
                        {"name": "safety", "status": "done", "detail": safety_detail},
                        {"name": "retrieval", "status": "done", "detail": retrieval_detail},
                        {"name": "fusion", "status": "done", "detail": fusion_detail},
                        {"name": "generation", "status": "running", "detail": "STIS consensus (contradiction detected)"},
                    ]})
                    stis_result = await asyncio.to_thread(
                        request_stis_consensus,
                        query,
                        str(contradiction["rag_claim"]),
                        str(contradiction["graph_claim"]),
                    )
                    # log the STIS event regardless of outcome
                    await asyncio.to_thread(
                        log_stis_resolution,
                        query,
                        str(contradiction["rag_claim"]),
                        str(contradiction["graph_claim"]),
                        float(contradiction["similarity"]),
                        float(stis_decision["best_cswr"]),
                        stis_result,
                    )
                    if stis_result["resolved"]:
                        stis_routed = True
                        stis_response = stis_result["resolution"]["text"]
                        STIS_ROUTING_EVENTS.labels(outcome="resolved").inc()
                        logger.info("STIS resolved contradiction for query: '%.50s...'", query)
                    else:
                        STIS_ROUTING_EVENTS.labels(outcome="failed").inc()
                        logger.warning("STIS failed, falling back to Ollama: %s", stis_result["error"])
                else:
                    STIS_ROUTING_EVENTS.labels(outcome="skipped").inc()

            prepared = _orchestrator.build_prompts(
                query=query,
                mode=mode,
                session_id=session_id,
                fused_context=fused_context,
                actual_weights=actual_weights,
                web_status=web_status,
                expanded_query=expanded_query,
                query_expanded=query_expanded,
            )

            record_turn(session_id, "user", query)
            # update fused_context from build_prompts (may include profile/conv context)
            fused_context = prepared["fused_context"]
            ctx_len = len(fused_context)
            logger.info("Context length: %d chars", ctx_len)
            _t_stis_prompt = _time_mod.perf_counter()
            logger.info("[TIMING] stis+prompt: %.0f ms", (_t_stis_prompt - _t_fusion) * 1000)

            # Step 4: LLM Generation (skip if STIS already resolved)
            if stis_routed and stis_response:
                full_response = stis_response
                token_count = len(stis_response.split())
                gen_detail = f"STIS consensus ({token_count} words)"
                # send the full STIS response as a single token chunk
                await websocket.send_json({"type": "pipeline", "agents": [
                    {"name": "safety", "status": "done", "detail": safety_detail},
                    {"name": "retrieval", "status": "done", "detail": retrieval_detail},
                    {"name": "fusion", "status": "done", "detail": fusion_detail},
                    {"name": "generation", "status": "done", "detail": gen_detail},
                ]})
                await websocket.send_json({"chunk": full_response, "weights": actual_weights, "reward": 0.0})
            else:
                pass  # fall through to normal Ollama generation below

            if not stis_routed:
                engine = get_engine()
                model_name = engine.model
                await websocket.send_json({"type": "pipeline", "agents": [
                    {"name": "safety", "status": "done", "detail": safety_detail},
                    {"name": "retrieval", "status": "done", "detail": retrieval_detail},
                    {"name": "fusion", "status": "done", "detail": fusion_detail},
                    {"name": "generation", "status": "running", "detail": f"Streaming {model_name} ({ctx_len} chars context)"},
                ]})

                full_response = ""
                token_count = 0
                _ttft_logged = False

                _num_predict = NUM_PREDICT.get(mode, 800)
                for text_chunk in engine.stream(
                    messages=[{"role": "system", "content": prepared["system_prompt"]}, {"role": "user", "content": prepared["user_prompt"]}],
                    temperature=0.1, num_ctx=4096, num_predict=_num_predict,
                ):
                    if not _ttft_logged:
                        _ttft = (_time_mod.perf_counter() - _t0) * 1000
                        logger.info("[TIMING] TTFT: %.0f ms", _ttft)
                        _ttft_logged = True
                    full_response += text_chunk
                    token_count += 1
                    await websocket.send_json({"chunk": text_chunk, "weights": actual_weights, "reward": 0.0})

                gen_detail = f"{token_count} tokens via {model_name}"
                _t_gen = _time_mod.perf_counter()
                logger.info("[TIMING] generation: %.0f ms (TTFT included)", (_t_gen - _t_stis_prompt) * 1000)

            # Step 5: Send done immediately, run critique async (Opt 6)
            # Record the turn before critique so user sees the response instantly
            record_turn(session_id, "assistant", full_response)

            # Show critique as running in pipeline UI
            await websocket.send_json({"type": "pipeline", "agents": [
                {"name": "safety", "status": "done", "detail": safety_detail},
                {"name": "retrieval", "status": "done", "detail": retrieval_detail},
                {"name": "fusion", "status": "done", "detail": fusion_detail},
                {"name": "generation", "status": "done", "detail": gen_detail},
            ]})

            # Send done message with placeholder reward (critique runs async)
            await websocket.send_json({
                "type": "done", "response": full_response,
                "fusion_weights": {
                    "rag": actual_weights[0] if len(actual_weights) > 0 else 0.0,
                    "cag": actual_weights[1] if len(actual_weights) > 1 else 0.0,
                    "graph": actual_weights[2] if len(actual_weights) > 2 else 0.0,
                },
                "reward": 0.0,
                "proactive": "",
                "proactive_suggestions": [],
                "query_expanded": query_expanded,
                "expanded_query": expanded_query if query_expanded else None,
                "web_status": web_status,
            })
            _t_done = _time_mod.perf_counter()
            logger.info("[TIMING] TOTAL (query to done): %.0f ms", (_t_done - _t0) * 1000)

            # Fire-and-forget async critique
            async def _run_critique_async(
                ws: WebSocket,
                q: str,
                resp: str,
                ctx: str,
                weights: list[float],
                w_status: str,
            ) -> None:
                try:
                    result = await asyncio.to_thread(
                        _orchestrator.finalize,
                        query=q,
                        llm_response=resp,
                        fused_context=ctx,
                        actual_weights=weights,
                        web_status=w_status,
                    )
                    critique_reward = result["reward"]
                    CRITIQUE_REWARD.observe(critique_reward)
                    for wpath in ("rag", "cag", "graph"):
                        FUSION_WEIGHT_DIST.labels(path=wpath).observe(result["fusion_weights"].get(wpath, 0))

                    # send critique frame to frontend (non-blocking update)
                    await ws.send_json({
                        "type": "critique",
                        "reward": critique_reward,
                        "proactive_suggestions": result["proactive_suggestions"],
                        "response": result["response"],
                    })
                except (WebSocketDisconnect, RuntimeError):
                    logger.debug("WS closed before critique delivery")
                except Exception as exc:
                    logger.warning("Async critique failed: %s", exc)

            asyncio.create_task(_run_critique_async(
                websocket, query, full_response, fused_context,
                actual_weights, web_status,
            ))

    except WebSocketDisconnect:
        WS_CONNECTIONS.dec()
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

    # probe STIS engine health (non-blocking, best-effort)
    stis_info: Dict[str, Any] = {"enabled": False, "model": None, "status": "offline"}
    if cfg.get("stis", {}).get("enabled", False):
        stis_info["enabled"] = True
        try:
            import httpx
            stis_host = cfg["stis"].get("host", "http://localhost:8100")
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{stis_host}/health")
            if resp.status_code == 200:
                health = resp.json()
                stis_info["model"] = health.get("model_id", "Qwen2.5-1.5B")
                stis_info["status"] = "ready" if health.get("model_loaded") else "standby"
                stis_info["agents"] = health.get("num_agents", 2)
        except Exception:
            stis_info["status"] = "offline"

    return {
        "status": "alive",
        "gpu": gpu_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model": cfg["llm"]["model"],
        "policy": "CQL" if Path(cfg["rl"]["policy_path"]).exists() else "heuristic",
        "policy_exists": Path(cfg["rl"]["policy_path"]).exists(),
        "boot_id": _boot_id,
        "stis": stis_info,
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
    allowed = {".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

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
    supported = (list(docs_path.rglob("*.txt")) + list(docs_path.rglob("*.md"))
                  + list(docs_path.rglob("*.pdf")))
    image_files = [f for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tiff")
                   for f in docs_path.rglob(ext)]

    if not supported and not image_files:
        return {
            "status": "empty",
            "message": "No documents found. Add .txt, .md, .pdf, or image files to data/docs/ and try again.",
            "docs_path": str(docs_path),
        }

    t0 = time.time()
    index = build_rag_index()
    elapsed = round(time.time() - t0, 2)

    meta_path = _get_metadata_path()
    chunk_count = len(json.loads(meta_path.read_text())) if meta_path.exists() else 0

    # report image index stats if multimodal is enabled
    image_count = 0
    img_meta = PROJECT_ROOT / "indexes" / "image_metadata.json"
    if img_meta.exists():
        image_count = len(json.loads(img_meta.read_text()))

    total_files = len(supported) + len(image_files)
    logger.info(
        "Reindex complete: %d text files, %d images, %d chunks, %.2fs",
        len(supported), len(image_files), chunk_count, elapsed,
    )
    return {
        "status": "reindexed",
        "files_processed": total_files,
        "chunks_indexed": chunk_count,
        "images_indexed": image_count,
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


@app.post("/api/fine-tune")
@limiter.limit("1/hour")
async def fine_tune_endpoint(request: Request, body: Dict[str, Any]) -> Dict[str, Any]:
    """Kick off LoRA fine-tuning on high-reward replay episodes. Admin-only."""
    from backend.rl.fine_tune import SFTJobConfig, default_config, run_sft

    # admin auth: require RLFUSION_ADMIN_KEY as Bearer token
    admin_key = os.environ.get("RLFUSION_ADMIN_KEY", "")
    auth_header = request.headers.get("Authorization", "")
    if not admin_key or auth_header != f"Bearer {admin_key}":
        return {
            "status": "unauthorized",
            "message": "Set RLFUSION_ADMIN_KEY env var and pass as Bearer token.",
        }

    # build config from request body, falling back to defaults
    defaults = default_config()
    job_config = SFTJobConfig(
        base_model=body.get("base_model", defaults["base_model"]),
        lora_rank=int(body.get("lora_rank", defaults["lora_rank"])),
        lora_alpha=int(body.get("lora_alpha", defaults["lora_alpha"])),
        lora_dropout=float(body.get("lora_dropout", defaults["lora_dropout"])),
        learning_rate=float(body.get("learning_rate", defaults["learning_rate"])),
        num_epochs=int(body.get("num_epochs", defaults["num_epochs"])),
        batch_size=int(body.get("batch_size", defaults["batch_size"])),
        max_seq_length=int(body.get("max_seq_length", defaults["max_seq_length"])),
        min_reward=float(body.get("min_reward", defaults["min_reward"])),
        max_episodes=int(body.get("max_episodes", defaults["max_episodes"])),
        val_split=float(body.get("val_split", defaults["val_split"])),
        output_dir=str(body.get("output_dir", defaults["output_dir"])),
    )

    # run training (blocking; heavy workload)
    result = await asyncio.to_thread(run_sft, job_config)
    corr_id = uuid.uuid4().hex[:8]
    logger.info("Fine-tune job %s: status=%s episodes=%d", corr_id, result["status"], result["episodes_used"])
    return {**result, "job_id": corr_id}


@app.get("/api/images/{image_path:path}")
async def serve_image(image_path: str) -> Any:
    """Serve images from data/images/ for multimodal retrieval results.

    Path traversal is blocked: only files under data/images/ are served.
    """
    from fastapi.responses import FileResponse

    # reject traversal attempts
    if ".." in image_path or image_path.startswith("/"):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Invalid path"}, status_code=400)

    full_path = PROJECT_ROOT / "data" / "images" / image_path
    resolved = full_path.resolve()
    allowed_root = (PROJECT_ROOT / "data" / "images").resolve()

    if not str(resolved).startswith(str(allowed_root)):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Path outside allowed directory"}, status_code=403)

    if not resolved.exists() or not resolved.is_file():
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Image not found"}, status_code=404)

    suffix = resolved.suffix.lower().lstrip(".")
    mime_map = {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "webp": "image/webp", "bmp": "image/bmp", "tiff": "image/tiff",
        "svg": "image/svg+xml",
    }
    media_type = mime_map.get(suffix, "application/octet-stream")
    return FileResponse(str(resolved), media_type=media_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

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
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST,
)
from pydantic import ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.api.auth import assert_admin_key_configured, require_admin
from backend.api.integrity import verify_model_checksums
from backend.api.models import ChatRequest, ConfigPatch, FineTuneRequest
from backend.config import cfg, PROJECT_ROOT
from backend.core.model_router import get_engine
from backend.core.critique import critique, log_episode_to_replay_buffer, get_critique_instruction, strip_critique_block, check_safety
from backend.core.memory import expand_query_with_context, record_turn, get_context_for_prompt, clear_memory
from backend.core.profile import detect_and_save_memory, get_user_profile
from backend.core.retrievers import retrieve
from backend.core.utils import embed_text
from backend.agents.orchestrator import Orchestrator, apply_markdown_formatting

# Max upload size per file (10 MB)
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Max query length (characters) to prevent prompt stuffing
_MAX_QUERY_LEN = 4000
# CAG threshold lookup (single source of truth in backend.config.yaml).
_CAG_CFG = cfg.get("cag", {}) or {}
_CAG_FAST_PATH = float(_CAG_CFG.get("fast_path_threshold", 0.90))
# WS hardening: per-connection caps. Mirrors slowapi's "10/minute" on /chat.
_WS_AUTH_TIMEOUT_S = 2.0
_WS_MAX_FRAME_BYTES = 32 * 1024
_WS_MAX_FRAMES_PER_MINUTE = 10

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
    """Wrapper for CQL policy weights loaded from d3rlpy checkpoint.

    Handles both legacy 4-output policies and new 2-output policies.
    For legacy policies, outputs 4D then the caller extracts CAG+Graph dims.
    """
    def __init__(self, policy_weights):
        import torch.nn as nn
        dev = "cuda" if USE_CUDA else "cpu"

        # detect output size from mu weight shape
        mu_shape = policy_weights['_mu.weight'].shape[0]
        self.output_dim = mu_shape

        self.encoder = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()).to(dev)
        self.mu = nn.Linear(256, mu_shape).to(dev)
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

    # ── Fail-closed boot gates ───────────────────────────────
    # 1. No admin key, no boot. Several endpoints mutate disk and state.
    assert_admin_key_configured()
    # 2. Verify pinned GGUF checksums if the manifest exists.
    verify_model_checksums(PROJECT_ROOT / "models")

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

    logger.info("Retrieval paths: CAG + GraphRAG (2-path architecture)")

    cql_path = PROJECT_ROOT / "models" / "rl_policy_cql.d3"
    ppo_path = PROJECT_ROOT / "rl_policy.zip"
    policy_status = "not trained"
    map_location = "cuda" if USE_CUDA else "cpu"

    if cql_path.exists():
        logger.info("Loading CQL policy...")
        try:
            # weights_only=True refuses to unpickle arbitrary objects; d3rlpy
            # saves a dict-of-tensors that survives this stricter loader.
            state_dict = torch.load(
                str(cql_path), weights_only=True, map_location=map_location,
            )
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

    logger.info("=" * 64)
    logger.info("RLFUSION ORCHESTRATOR - BACKEND STARTED")
    if USE_CUDA:
        logger.info(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        logger.info("Device: CPU")
    logger.info(f"LLM: {cfg['llm']['model']}")
    logger.info(f"RL Policy: {policy_status}")

    # Initialize multi-agent orchestrator (Phase 1)
    global _orchestrator
    _orchestrator = Orchestrator(rl_policy=_rl_policy)
    logger.info("Orchestrator: multi-agent pipeline active (LangGraph)")

    # Start background memory monitoring (Step 8)
    from backend.core.metrics import get_monitor
    mem_monitor = get_monitor(interval=10)
    mem_monitor.start()

    logger.info("=" * 64)
    yield
    mem_monitor.stop()
    logger.info("Shutting down RLFusion Orchestrator")


app = FastAPI(title="RLFusion Orchestrator", version="0.1.0", lifespan=lifespan)


def _build_cors_origins() -> list[str]:
    """Return the explicit CORS origin allowlist from config.

    `["*"]` paired with `allow_credentials=True` is illegal per the CORS
    spec and was the previous default; refuse to start in that combo so
    a careless edit can't reintroduce it.
    """
    raw = cfg.get("cors", {}).get("allowed_origins", []) or []
    if not isinstance(raw, list) or not all(isinstance(o, str) for o in raw):
        raise RuntimeError(
            "cors.allowed_origins must be a list of origin strings, e.g. "
            "['http://localhost:5173']."
        )
    if "*" in raw:
        raise RuntimeError(
            "cors.allowed_origins contains '*' but the server enforces "
            "allow_credentials=True. Set explicit origins or remove the "
            "wildcard entry."
        )
    if not raw:
        # Empty list is a safer default than '*'; admin can set it explicitly.
        logger.warning("cors.allowed_origins is empty; browser clients will be blocked.")
    return raw


app.add_middleware(
    CORSMiddleware,
    allow_origins=_build_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Correlation-ID"],
)
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


@app.get("/metrics", dependencies=[Depends(require_admin)])
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics for scraping. Admin-gated.

    Metrics include request counts and reward distributions; treat them
    as operational data rather than public, since a hostile scraper can
    use latency histograms to profile pipeline behavior.
    """
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


def _compute_rl_fusion_weights(query: str, policy, retrieval_results=None) -> np.ndarray:
    """Compute 2-path fusion weights. Delegates to the canonical implementation."""
    from backend.agents.fusion_agent import compute_rl_weights
    return compute_rl_weights(query, policy, retrieval_results)


def _build_fusion_context(
    retrieval_results: Dict[str, List[Dict[str, Any]]],
    weights: np.ndarray,
    query: str = "",
) -> str:
    """Build fused context. Delegates to the canonical implementation."""
    from backend.agents.fusion_agent import build_fusion_context
    return build_fusion_context(retrieval_results, weights, query=query)


# Import prompts from orchestrator (single source of truth)
from backend.agents.orchestrator import (
    generate_system_prompt as _generate_system_prompt,
    generate_user_prompt as _generate_user_prompt,
    NUM_PREDICT,
)


@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, body: ChatRequest) -> Dict[str, Any]:
    query = body.query.strip()
    mode = body.mode

    if not query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query must be a non-empty string.",
        )

    # Multi-agent orchestration (Phase 1)
    t_start = _time_mod.perf_counter()
    result = _orchestrator.run(query=query, mode=mode, session_id="chat")
    QUERY_LATENCY.observe(_time_mod.perf_counter() - t_start)

    # record fusion weight distribution
    fw = result["fusion_weights"]
    for path_name in ("cag", "graph"):
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


async def _ws_authorize(websocket: WebSocket) -> bool:
    """Allow the WS handshake if the Origin is on the allowlist, otherwise
    require a first-frame bearer-token auth message.

    Returns True on success. On failure, closes the socket and returns
    False so the caller can fall through cleanly.
    """
    allowed = set(cfg.get("cors", {}).get("allowed_origins", []) or [])
    origin = websocket.headers.get("origin", "")
    if origin and origin in allowed:
        return True

    # Origin not on allowlist: require first-frame `{"auth": "Bearer ..."}`.
    try:
        frame = await asyncio.wait_for(
            websocket.receive_text(), timeout=_WS_AUTH_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("WS auth timeout; closing")
        await websocket.close(code=4401)
        return False
    except WebSocketDisconnect:
        return False

    try:
        body = json.loads(frame)
    except json.JSONDecodeError:
        await websocket.close(code=4400)
        return False

    presented = str(body.get("auth", "")).strip()
    admin_key = os.environ.get("RLFUSION_ADMIN_KEY", "")
    if not presented.startswith("Bearer ") or not admin_key:
        await websocket.close(code=4401)
        return False
    import hmac as _hmac
    if not _hmac.compare_digest(
        presented[len("Bearer "):].encode("utf-8"),
        admin_key.encode("utf-8"),
    ):
        await websocket.close(code=4401)
        return False
    return True


class _WsTokenBucket:
    """Per-connection sliding-window rate limiter for inbound frames."""

    def __init__(self, max_frames: int, window_s: float = 60.0) -> None:
        self._max = max_frames
        self._window = window_s
        self._stamps: list[float] = []

    def allow(self) -> bool:
        now = _time_mod.monotonic()
        self._stamps = [t for t in self._stamps if now - t < self._window]
        if len(self._stamps) >= self._max:
            return False
        self._stamps.append(now)
        return True


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    WS_CONNECTIONS.inc()
    session_id = str(id(websocket))
    bucket = _WsTokenBucket(_WS_MAX_FRAMES_PER_MINUTE)

    try:
        if not await _ws_authorize(websocket):
            return

        while True:
            data = await websocket.receive_text()

            # Frame-size cap before json.loads. A 1 GB JSON bomb would
            # otherwise allocate that much before the parser even errors.
            if len(data) > _WS_MAX_FRAME_BYTES:
                logger.warning(
                    "WS frame %d bytes exceeds cap %d; closing",
                    len(data), _WS_MAX_FRAME_BYTES,
                )
                await websocket.close(code=1009)
                return

            if not bucket.allow():
                logger.info("WS rate limit hit for session %s", session_id)
                await websocket.close(code=1008)
                return

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
                    "fusion_weights": {"cag": 0, "graph": 0},
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
            cag_n = len(rr.get("cag", []))
            graph_n = len(rr.get("graph", []))
            retrieval_detail = f"{cag_n} CAG, {graph_n} Graph"
            _t_retrieval = _time_mod.perf_counter()
            logger.info("[TIMING] retrieval: %.0f ms", (_t_retrieval - _t_safety) * 1000)

            # record retrieval path usage
            for path_label, count in [("cag", cag_n), ("graph", graph_n)]:
                if count > 0:
                    RETRIEVAL_PATH_USAGE.labels(path=path_label).inc()

            # CAG fast-path: strong cache hit skips fusion + generation entirely
            cag_results = rr.get("cag", [])
            if cag_n > 0 and graph_n == 0 and cag_results[0].get("score", 0) >= _CAG_FAST_PATH:
                cached_text = cag_results[0].get("text", "")
                _t_cag = _time_mod.perf_counter()
                logger.info("[CAG FAST-PATH] cache hit in %.0f ms, skipping generation",
                            (_t_cag - _t0) * 1000)
                await websocket.send_json({"type": "pipeline", "agents": [
                    {"name": "safety", "status": "done", "detail": safety_detail},
                    {"name": "retrieval", "status": "done", "detail": f"{cag_n} CAG (cache hit)"},
                    {"name": "fusion", "status": "skipped", "detail": "CAG hit"},
                    {"name": "generation", "status": "skipped", "detail": "Served from cache"},
                ]})
                await websocket.send_json({
                    "type": "done",
                    "response": cached_text,
                    "fusion_weights": {"cag": 1.0, "graph": 0.0},
                    "reward": cag_results[0].get("score", 0.9),
                    "proactive_suggestions": [],
                })
                # record the turn for memory context
                await asyncio.to_thread(record_turn, session_id, "user", query)
                await asyncio.to_thread(record_turn, session_id, "assistant", cached_text)
                continue

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
            actual_weights = fusion_result.get("actual_weights", [0.5, 0.5])
            fused_context = fusion_result.get("fused_context", "")

            # summarize fusion weights for detail text
            w = actual_weights
            fusion_detail = f"CAG {w[0]*100:.0f}% Graph {w[1]*100:.0f}%"
            _t_fusion = _time_mod.perf_counter()
            logger.info("[TIMING] fusion: %.0f ms", (_t_fusion - _t_retrieval) * 1000)

            gen_detail = ""
            full_response = ""
            rr = retrieval_result.get("retrieval_results", {})

            prepared = _orchestrator.build_prompts(
                query=query,
                mode=mode,
                session_id=session_id,
                fused_context=fused_context,
                actual_weights=actual_weights,
                expanded_query=expanded_query,
                query_expanded=query_expanded,
            )

            record_turn(session_id, "user", query)
            # update fused_context from build_prompts (may include profile/conv context)
            fused_context = prepared["fused_context"]
            ctx_len = len(fused_context)
            logger.info("Context length: %d chars", ctx_len)
            _t_prompt = _time_mod.perf_counter()
            logger.info("[TIMING] prompt: %.0f ms", (_t_prompt - _t_fusion) * 1000)

            # Step 4: LLM Generation
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
            logger.info("[TIMING] generation: %.0f ms (TTFT included)", (_t_gen - _t_prompt) * 1000)

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
                    "cag": actual_weights[0] if len(actual_weights) > 0 else 0.0,
                    "graph": actual_weights[1] if len(actual_weights) > 1 else 0.0,
                },
                "reward": 0.0,
                "proactive": "",
                "proactive_suggestions": [],
                "query_expanded": query_expanded,
                "expanded_query": expanded_query if query_expanded else None,
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
            ) -> None:
                try:
                    result = await asyncio.to_thread(
                        _orchestrator.finalize,
                        query=q,
                        llm_response=resp,
                        fused_context=ctx,
                        actual_weights=weights,
                    )
                    critique_reward = result["reward"]
                    CRITIQUE_REWARD.observe(critique_reward)
                    for wpath in ("cag", "graph"):
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
                actual_weights,
            ))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        # decrement no matter how we leave the loop: auth fail, rate limit,
        # frame overflow, JSON error, or clean disconnect.
        WS_CONNECTIONS.dec()


@app.get("/api/config")
@limiter.limit("10/minute")
async def get_config(request: Request) -> Dict[str, Any]:
    return {"web": {"enabled": cfg.get("web", {}).get("enabled", False),
                    "max_results": cfg.get("web", {}).get("max_results", 3),
                    "search_timeout": cfg.get("web", {}).get("search_timeout", 10)}}


@app.patch("/api/config", dependencies=[Depends(require_admin)])
@limiter.limit("10/minute")
async def update_config(request: Request, body: ConfigPatch) -> Dict[str, Any]:
    cfg["web"]["enabled"] = body.web.enabled
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        current_config = yaml.safe_load(f)
    current_config["web"]["enabled"] = cfg["web"]["enabled"]
    with open(config_path, 'w') as f:
        yaml.dump(current_config, f, default_flow_style=False, sort_keys=False)
    return {"status": "updated", "web": {"enabled": cfg["web"]["enabled"]}}


@app.get("/ping")
@limiter.limit("10/minute")
async def ping(request: Request) -> Dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    # report what the engine actually resolved to, not the static config value
    try:
        engine = get_engine()
        active_engine = engine.engine
        active_model = engine.model
        engine_resolution = getattr(engine, "_resolution", "")
    except Exception:
        active_engine = cfg.get("inference", {}).get("engine", "ollama")
        active_model = cfg.get("llm", {}).get("model", "?")
        engine_resolution = "engine init failed"

    return {
        "status": "alive",
        "gpu": gpu_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model": active_model,
        "inference_engine": active_engine,
        "engine_resolution": engine_resolution,
        "policy": "CQL" if Path(cfg["rl"]["policy_path"]).exists() else "heuristic",
        "policy_exists": Path(cfg["rl"]["policy_path"]).exists(),
        "boot_id": _boot_id,
    }


# Magic-byte sniffing for the allowed upload types. PDF and image files have
# stable file-magic prefixes; .txt/.md fall through to a UTF-8 decode check.
_MAGIC_BYTES = {
    ".pdf": b"%PDF-",
}


def _looks_like_extension(content: bytes, ext: str) -> bool:
    """Crude content sniff to reject mismatched extensions."""
    if ext in _MAGIC_BYTES:
        return content.startswith(_MAGIC_BYTES[ext])
    if ext in (".txt", ".md"):
        try:
            content[:4096].decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True
    return False


@app.post("/api/upload", dependencies=[Depends(require_admin)])
@limiter.limit("10/minute")
async def upload_documents(request: Request) -> Dict[str, Any]:
    """Upload .txt/.md/.pdf files to data/docs/ for indexing. Admin-only."""
    form = await request.form()
    files = form.getlist("files")
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided in form field 'files'.",
        )

    docs_path = PROJECT_ROOT / "data" / "docs"
    docs_path.mkdir(parents=True, exist_ok=True)

    saved, skipped = [], []
    allowed = {".txt", ".md", ".pdf"}

    for f in files:
        if not hasattr(f, 'filename'):
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in allowed:
            skipped.append(f"{f.filename} (extension)")
            continue
        content = await f.read()
        if len(content) > _MAX_UPLOAD_BYTES:
            skipped.append(f"{f.filename} (>{_MAX_UPLOAD_BYTES // (1024*1024)}MB)")
            continue
        if not _looks_like_extension(content, ext):
            skipped.append(f"{f.filename} (content/extension mismatch)")
            continue
        safe_name = Path(f.filename).name  # strip path components
        # double-check: reject filenames with traversal attempts
        if '..' in safe_name or safe_name.startswith('.'):
            skipped.append(f"{f.filename} (invalid filename)")
            continue
        # Hash-prefix the destination so two uploads with the same name don't
        # silently overwrite each other and so a poisoned reupload can't take
        # the slot of a trusted document.
        import hashlib as _hashlib
        digest = _hashlib.sha256(content).hexdigest()[:12]
        dest = docs_path / f"{digest}_{safe_name}"
        dest.write_bytes(content)
        saved.append(f.filename)
        logger.info("Uploaded: %s -> %s (%d bytes)", f.filename, dest.name, len(content))

    return {
        "status": "uploaded",
        "saved": saved,
        "skipped": skipped,
        "total_saved": len(saved),
        "total_skipped": len(skipped),
    }


@app.post("/api/reindex", dependencies=[Depends(require_admin)])
@limiter.limit("3/minute")
async def reindex_documents(request: Request) -> Dict[str, Any]:
    """Rechunk data/docs/ and rebuild the entity graph."""
    from backend.core.retrievers import build_doc_chunks, _get_docs_path
    import time

    docs_path = _get_docs_path()
    supported = (list(docs_path.rglob("*.txt")) + list(docs_path.rglob("*.md"))
                  + list(docs_path.rglob("*.pdf")))

    if not supported:
        return {
            "status": "empty",
            "message": "No documents found. Add .txt, .md, or .pdf files to data/docs/ and try again.",
            "docs_path": str(docs_path),
        }

    t0 = time.time()
    chunk_count = build_doc_chunks()

    # entity count comes from the graph engine, re-loaded after rebuild
    from backend.core.retrievers import _get_graph_engine
    engine = _get_graph_engine()
    entity_count = engine.node_count if engine is not None else 0

    elapsed = round(time.time() - t0, 2)
    logger.info(
        "Reindex complete: %d files, %d chunks, %d entities, %.2fs",
        len(supported), chunk_count, entity_count, elapsed,
    )
    return {
        "status": "reindexed",
        "files_processed": len(supported),
        "chunks_indexed": chunk_count,
        "entities_extracted": entity_count,
        "elapsed_seconds": elapsed,
    }


@app.delete("/api/reset", dependencies=[Depends(require_admin)])
@limiter.limit("5/minute")
async def reset_state(request: Request) -> Dict[str, Any]:
    """Wipe all transient state: cache, episodes, replay, conversations. Admin-only."""
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


@app.post("/api/fine-tune", dependencies=[Depends(require_admin)])
@limiter.limit("1/hour")
async def fine_tune_endpoint(request: Request, body: FineTuneRequest) -> Dict[str, Any]:
    """Kick off LoRA fine-tuning on high-reward replay episodes. Admin-only."""
    from backend.rl.fine_tune import SFTJobConfig, default_config, run_sft

    defaults = default_config()
    # FineTuneRequest already bounds every numeric field. Resolve output_dir
    # against the project root to make sure the validator's "relative path"
    # invariant is anchored where the rest of the app expects it.
    requested_output = (PROJECT_ROOT / body.output_dir).resolve()
    if not str(requested_output).startswith(str(PROJECT_ROOT.resolve())):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="output_dir resolves outside the project root.",
        )

    job_config = SFTJobConfig(
        base_model=body.base_model or defaults["base_model"],
        lora_rank=body.lora_rank,
        lora_alpha=body.lora_alpha,
        lora_dropout=body.lora_dropout,
        learning_rate=body.learning_rate,
        num_epochs=body.num_epochs,
        batch_size=body.batch_size,
        max_seq_length=body.max_seq_length,
        min_reward=body.min_reward,
        max_episodes=body.max_episodes,
        val_split=body.val_split,
        output_dir=str(requested_output.relative_to(PROJECT_ROOT.resolve())),
    )

    # run training (blocking; heavy workload)
    result = await asyncio.to_thread(run_sft, job_config)
    corr_id = uuid.uuid4().hex[:8]
    logger.info("Fine-tune job %s: status=%s episodes=%d", corr_id, result["status"], result["episodes_used"])
    return {**result, "job_id": corr_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

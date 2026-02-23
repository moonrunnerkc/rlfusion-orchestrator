# Author: Bradley R. Kinnard
"""STIS FastAPI server: exposes /generate for swarm consensus generation.

Model loading is LAZY: Qwen2.5-1.5B stays off-GPU until a /generate request
actually arrives. After the request completes, an idle timer starts. If no
new requests arrive within STIS_IDLE_TIMEOUT seconds (default 120), the model
auto-unloads and GPU memory is freed for Ollama. Next request re-loads (~3s).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException

from stis_engine.config import STISConfig, SwarmConfig, load_config
from stis_engine.model_loader import load_model, unload_model
from stis_engine.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from stis_engine.swarm import SwarmEngine

logger = logging.getLogger(__name__)

# module-level references
_engine: SwarmEngine | None = None
_config: STISConfig | None = None
_model_loaded: bool = False
_load_lock = threading.Lock()
_idle_timer: threading.Timer | None = None
_IDLE_TIMEOUT = 120  # seconds before auto-unload


def _schedule_idle_unload() -> None:
    """Reset the idle timer. Model unloads if no requests for _IDLE_TIMEOUT seconds."""
    global _idle_timer
    if _idle_timer is not None:
        _idle_timer.cancel()
    _idle_timer = threading.Timer(_IDLE_TIMEOUT, _do_idle_unload)
    _idle_timer.daemon = True
    _idle_timer.start()


def _do_idle_unload() -> None:
    """Callback: unload model after idle timeout to free VRAM for Ollama."""
    global _engine, _model_loaded
    with _load_lock:
        if _model_loaded:
            logger.info("Idle timeout (%ds): unloading STIS model to free VRAM", _IDLE_TIMEOUT)
            unload_model()
            _engine = None
            _model_loaded = False


def _ensure_model_loaded() -> SwarmEngine:
    """Lazy-load the model on first request. Thread-safe."""
    global _engine, _model_loaded
    with _load_lock:
        if _engine is not None and _model_loaded:
            return _engine
        logger.info("Lazy-loading STIS model (first request or post-idle reload)...")
        t0 = time.perf_counter()
        model, tokenizer = load_model(_config.model)
        _engine = SwarmEngine(model, tokenizer, _config.model, _config.swarm)
        _model_loaded = True
        logger.info("STIS model loaded in %.1fs", time.perf_counter() - t0)
        return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load config only (model stays off-GPU). Shutdown: release if loaded."""
    global _config, _IDLE_TIMEOUT

    _config = load_config()
    _IDLE_TIMEOUT = int(__import__("os").environ.get("STIS_IDLE_TIMEOUT", "120"))
    logger.info("STIS config loaded: agents=%d, threshold=%.3f, model=%s",
                 _config.swarm.num_agents, _config.swarm.similarity_threshold,
                 _config.model.model_id)
    logger.info("STIS model loading: LAZY (loads on first /generate, unloads after %ds idle)", _IDLE_TIMEOUT)
    logger.info("STIS engine ready on port %d (model NOT loaded yet)", _config.server.port)
    yield

    # shutdown: free GPU memory if loaded
    if _idle_timer is not None:
        _idle_timer.cancel()
    if _model_loaded:
        unload_model()
    _config = None
    logger.info("STIS engine shut down")


app = FastAPI(
    title="STIS Engine",
    description="Sub-Token Intuition Swarms: multi-agent consensus in continuous latent space",
    version="0.1.0",
    lifespan=lifespan,
    responses={500: {"model": ErrorResponse}},
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check. Reports model status without loading it."""
    if _config is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _engine is not None and _model_loaded:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_id=_config.model.model_id,
            hidden_dim=_engine.hidden_dim,
            num_agents=_engine.num_agents,
            similarity_threshold=_engine.similarity_threshold,
            device=device,
        )
    # config loaded but model not yet in VRAM
    return HealthResponse(
        status="ok",
        model_loaded=False,
        model_id=_config.model.model_id,
        hidden_dim=0,
        num_agents=_config.swarm.num_agents,
        similarity_threshold=_config.swarm.similarity_threshold,
        device=device,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Run swarm consensus generation. Lazy-loads model on first call.

    Accepts optional overrides for num_agents, similarity_threshold, and alpha.
    """
    if _config is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # lazy-load model into VRAM (no-op if already loaded)
    engine = await asyncio.to_thread(_ensure_model_loaded)

    # build per-request swarm config if any overrides provided
    if req.num_agents is not None or req.similarity_threshold is not None or req.alpha is not None:
        override_cfg = SwarmConfig(
            num_agents=req.num_agents or _config.swarm.num_agents,
            similarity_threshold=req.similarity_threshold or _config.swarm.similarity_threshold,
            alpha=req.alpha or _config.swarm.alpha,
            max_iterations=_config.swarm.max_iterations,
        )
        engine = SwarmEngine(
            engine._model,
            engine._tokenizer,
            _config.model,
            override_cfg,
        )

    t_start = time.perf_counter()

    try:
        result = engine.generate(req.prompt, max_new_tokens=req.max_new_tokens)
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="GPU out of memory. Try a shorter prompt or fewer agents.")
    except RuntimeError as exc:
        logger.error("Generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}")

    wall = time.perf_counter() - t_start
    logger.info("Generated %d tokens in %.2fs (sim=%.4f, iters=%d)",
                 result.total_tokens, wall, result.final_similarity, result.total_iterations)

    # reset idle timer: model stays loaded for _IDLE_TIMEOUT more seconds
    _schedule_idle_unload()

    response_data = result.to_dict()
    return GenerateResponse(**response_data)

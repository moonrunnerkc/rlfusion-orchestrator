# Author: Bradley R. Kinnard
"""Model loader for the STIS engine. Handles Qwen2.5-1.5B provisioning.

Loads the model in float16 to fit within a 3GB VRAM budget. Validates GPU
availability and falls back to CPU with a clear warning. Caches the loaded
model/tokenizer pair in module scope to avoid redundant loads.
"""
from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stis_engine.config import ModelConfig

logger = logging.getLogger(__name__)

# module-level cache to avoid reloading on every request
_cached_model: AutoModelForCausalLM | None = None
_cached_tokenizer: AutoTokenizer | None = None
_cached_model_id: str | None = None


def load_model(cfg: ModelConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen2.5-1.5B with the configured dtype and device.

    Will reuse a previously loaded model if the model_id matches.
    """
    global _cached_model, _cached_tokenizer, _cached_model_id

    if _cached_model is not None and _cached_model_id == cfg.model_id:
        logger.info("Reusing cached model: %s", cfg.model_id)
        return _cached_model, _cached_tokenizer  # type: ignore[return-value]

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.dtype, torch.float16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No CUDA GPU detected. STIS will run on CPU (slow). "
                        "float16 forced to float32 on CPU.")
        torch_dtype = torch.float32

    logger.info("Loading model %s (dtype=%s, device=%s)", cfg.model_id, torch_dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # set deterministic seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    vram_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    logger.info("Model loaded: %.1fM params, %.0f MB VRAM allocated", param_count, vram_mb)

    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_model_id = cfg.model_id

    return model, tokenizer


def unload_model() -> None:
    """Release the cached model and free GPU memory."""
    global _cached_model, _cached_tokenizer, _cached_model_id

    if _cached_model is not None:
        del _cached_model
        _cached_model = None
        _cached_tokenizer = None
        _cached_model_id = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded and GPU memory released")

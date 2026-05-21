# Author: Bradley R. Kinnard
"""Inference engine auto-detection.

Picks an engine + a real model name based on what's actually installed
on the host, in this order of preference:

1. `INFERENCE_ENGINE` env var or `inference.engine` in config.yaml that
   resolves cleanly (GGUFs on disk for `llama_cpp_dual`, ollama daemon
   reachable for `ollama`, etc.) — honored verbatim.
2. Fallback to ollama if the configured engine can't be satisfied AND the
   daemon at `inference.base_url` (or `http://localhost:11434`) responds.
3. As a last resort, return the original config unchanged so the caller
   sees a clean error from the engine itself.

For ollama, if `inference.model` is empty or not in the installed model
list, the smallest installed local (non-cloud) model is picked. No model
names are hard-coded; the choice comes from `/api/tags` on the daemon.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from backend.config import PROJECT_ROOT, get_inference_config

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def _ollama_models(base_url: str) -> list[dict[str, Any]]:
    """Return the list of locally-installed ollama models, or [] on error."""
    try:
        import httpx
        resp = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        resp.raise_for_status()
        models = resp.json().get("models", []) or []
        # cloud-hosted models report tiny `size`; we only want locally-resident ones
        return [m for m in models if int(m.get("size", 0)) > 1_000_000]
    except Exception as exc:
        logger.debug("ollama probe at %s failed: %s", base_url, exc)
        return []


# Substrings that suggest a tag is for a specialized coding/agent variant
# rather than a general chat model. Down-weighted when auto-picking.
_NON_CHAT_HINTS = ("coder", "cline", "plan", "act", "embed", "vision", "code", "agent")
# Preferred size band (in bytes). Mid-size models tend to be the best
# default for an enthusiast laptop: not too small to be unhelpful, not
# too big to OOM. Outside this band we still consider the model but
# score it lower.
_PREF_MIN_BYTES = 15 * 1024**3   # 15 GB
_PREF_MAX_BYTES = 40 * 1024**3   # 40 GB


def _model_score(m: dict[str, Any]) -> tuple[int, int]:
    """Score for auto-picking. Higher first-tuple element wins."""
    name = str(m.get("name", "")).lower()
    size = int(m.get("size", 0))
    base = 100
    for kw in _NON_CHAT_HINTS:
        if kw in name:
            base -= 25
    if _PREF_MIN_BYTES <= size <= _PREF_MAX_BYTES:
        base += 30
    elif size > _PREF_MAX_BYTES:
        base -= 10  # very large = slow on most laptops
    # ties broken by smaller size (faster)
    return (base, -size)


def _pick_ollama_model(installed: list[dict[str, Any]], preferred: str = "") -> str:
    """Resolve a model name to use against ollama.

    If `preferred` is one of the installed names (or its base before ':') use
    it verbatim — the user wins. Otherwise rank installed models by
    `_model_score` (prefers general chat models in the 15-40 GB band) and
    return the top one.
    """
    if not installed:
        return ""
    names = [m["name"] for m in installed]
    if preferred:
        if preferred in names:
            return preferred
        base = preferred.split(":")[0]
        for n in names:
            if n.startswith(f"{base}:") or n == base:
                return n
    return max(installed, key=_model_score)["name"]


def _ggufs_present(inf: dict[str, Any]) -> bool:
    cpu = Path(str(inf.get("cpu_model_path", "")))
    gpu = Path(str(inf.get("gpu_model_path", "")))
    if not cpu.is_absolute():
        cpu = PROJECT_ROOT / cpu
    if not gpu.is_absolute():
        gpu = PROJECT_ROOT / gpu
    return cpu.exists() and gpu.exists()


def resolve_inference_config() -> dict[str, Any]:
    """Return an inference config dict with engine + model resolved to what
    actually works on this host.

    Honors INFERENCE_ENGINE / INFERENCE_MODEL env overrides. The dict shape
    matches what get_inference_config() returns plus a "resolution" string
    describing why the chosen engine was selected (for log + /ping display).
    """
    inf = dict(get_inference_config())
    engine = str(inf.get("engine", "ollama"))
    auto = bool(inf.get("auto_detect", True))
    resolution = f"config: {engine}"

    forced_engine = bool(os.environ.get("INFERENCE_ENGINE"))
    if forced_engine:
        resolution = f"env INFERENCE_ENGINE={engine}"

    # llama_cpp_dual is only viable if both GGUFs exist
    if engine == "llama_cpp_dual" and not _ggufs_present(inf):
        msg = "llama_cpp_dual configured but GGUFs missing"
        if forced_engine or not auto:
            logger.warning("%s; honoring config and letting the caller fail loudly", msg)
        else:
            logger.warning("%s; auto-detect probing for ollama", msg)
            ollama_url = _DEFAULT_OLLAMA_HOST
            installed = _ollama_models(ollama_url)
            if installed:
                picked = _pick_ollama_model(installed, str(inf.get("model", "")))
                inf["engine"] = "ollama"
                inf["base_url"] = ollama_url
                inf["model"] = picked
                inf["resolution"] = (
                    f"auto: GGUFs missing -> ollama({picked})"
                )
                logger.info("Auto-detected ollama at %s with model=%s", ollama_url, picked)
                return inf
            logger.warning("ollama unreachable too; falling back to original config")

    if engine == "ollama":
        installed = _ollama_models(str(inf.get("base_url", _DEFAULT_OLLAMA_HOST)))
        requested = str(inf.get("model", ""))
        if installed:
            picked = _pick_ollama_model(installed, requested)
            if picked != requested:
                inf["model"] = picked
                resolution = f"{resolution}; auto-picked model={picked}"
                logger.info(
                    "Requested model %r not installed; using %r instead",
                    requested, picked,
                )
        else:
            resolution = f"{resolution}; ollama unreachable (will fail on use)"

    inf["resolution"] = resolution
    return inf

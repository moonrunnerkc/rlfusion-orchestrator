# Author: Bradley R. Kinnard
# backend/config.py - Configuration loader for RLFusion Orchestrator
# Originally built for personal offline use, now open-sourced for public benefit.

import os
from pathlib import Path
import yaml

# Project root is two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "backend" / "config.yaml"

# Per-host overrides (gitignored). Anything in .env (INFERENCE_MODEL,
# RLFUSION_DEVICE, RLFUSION_ADMIN_KEY, etc.) is loaded into os.environ
# before the YAML and `get_inference_config()` reads it. This is the
# right place for per-machine model picks — they don't belong in the
# committed config.yaml.
_DOTENV_PATH = PROJECT_ROOT / ".env"
if _DOTENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_DOTENV_PATH, override=False)
    except ImportError:
        # graceful fallback: parse KEY=VAL lines ourselves
        for raw in _DOTENV_PATH.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config file missing: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)


def _resolve_path(path_str: str) -> Path:
    """Resolve a path relative to project root, handling both absolute and relative."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def get_data_path() -> Path:
    """Return the data directory, creating if needed."""
    path = PROJECT_ROOT / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_db_path() -> Path:
    """Return the database directory, creating if needed."""
    path = PROJECT_ROOT / "db"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_index_path() -> Path:
    """Return the indexes directory, creating if needed."""
    path = PROJECT_ROOT / "indexes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_web_api_key() -> str:
    """Get web search API key from environment variable."""
    return os.environ.get("TAVILY_API_KEY", "")


def get_inference_config() -> dict[str, str | int]:
    """Return the inference engine configuration with env var overrides.

    Supports INFERENCE_ENGINE, INFERENCE_BASE_URL, INFERENCE_MODEL,
    and INFERENCE_API_KEY environment variables for container deployments.
    Also exposes dual-model paths for the llama_cpp_dual engine.
    """
    inf: dict[str, object] = cfg.get("inference", {})
    return {
        "engine": os.environ.get("INFERENCE_ENGINE") or str(inf.get("engine", "ollama")),
        "base_url": os.environ.get("INFERENCE_BASE_URL") or str(inf.get("base_url", "http://localhost:11434")),
        # empty string is fine — the resolver picks an installed model in that case
        "model": os.environ.get("INFERENCE_MODEL", str(inf.get("model", ""))),
        "max_concurrent": int(os.environ.get("INFERENCE_MAX_CONCURRENT", str(inf.get("max_concurrent", 4)))),
        "timeout_secs": int(os.environ.get("INFERENCE_TIMEOUT", str(inf.get("timeout_secs", 30)))),
        "openai_api_key": os.environ.get("INFERENCE_API_KEY") or str(inf.get("openai_api_key", "")),
        "cpu_model_path": str(inf.get("cpu_model_path", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf")),
        "gpu_model_path": str(inf.get("gpu_model_path", "models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")),
        "cpu_ctx_size": int(inf.get("cpu_ctx_size", 8192)),
        "gpu_ctx_size": int(inf.get("gpu_ctx_size", 8192)),
        "seed": int(inf.get("seed", 42)),
        "auto_detect": _to_bool(inf.get("auto_detect", True)),
    }


def _to_bool(val: object) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


__all__ = ["cfg", "PROJECT_ROOT", "get_project_root", "get_data_path",
           "get_db_path", "get_index_path", "get_web_api_key", "get_inference_config"]

# Author: Bradley R. Kinnard
# backend/config.py - Configuration loader for RLFusion Orchestrator
# Originally built for personal offline use, now open-sourced for public benefit.

import os
from pathlib import Path
import yaml

# Project root is two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "backend" / "config.yaml"

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


__all__ = ["cfg", "PROJECT_ROOT", "get_project_root", "get_data_path",
           "get_db_path", "get_index_path", "get_web_api_key"]

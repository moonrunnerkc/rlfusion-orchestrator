# backend/config.py
# Global config singleton - imported by every module
# No circular imports, no drama.

from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# Make it available at module level
__all__ = ["cfg"]

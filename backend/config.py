# backend/config.py — FINAL, BULLETPROOF VERSION
from pathlib import Path
import yaml

# Hard-code the ONLY correct location on your machine
CONFIG_PATH = Path("/home/brad/rlfusion/backend/config.yaml").expanduser().resolve()

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config file missing: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

print("\nCONFIG LOADED FROM:", CONFIG_PATH)
print("docs →", cfg["paths"]["docs"])
print("index →", cfg["paths"]["index"], "\n")

__all__ = ["cfg"]

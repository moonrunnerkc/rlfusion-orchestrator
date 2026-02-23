# Author: Bradley R. Kinnard
"""Entrypoint for running the STIS engine as a standalone service.

Usage:
    python -m stis_engine
    # or
    uvicorn stis_engine.server:app --host 0.0.0.0 --port 8100
"""
import logging
import uvicorn

from stis_engine.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

if __name__ == "__main__":
    cfg = load_config()
    uvicorn.run(
        "stis_engine.server:app",
        host=cfg.server.host,
        port=cfg.server.port,
        log_level="info",
    )

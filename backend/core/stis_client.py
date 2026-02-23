# Author: Bradley R. Kinnard
"""STIS fallback client: routes conflicting contexts to the STIS engine.

When RAG and Graph retrieval produce contradictory facts and CSWR confidence
is low, this client formats the conflict as opposing axioms and sends them
to the STIS microservice for mathematically forced consensus in latent space.

The client uses httpx with a strict 45-second timeout. If the STIS engine
is unreachable or times out, it falls back gracefully and lets Ollama handle
the query normally (with a logged warning).
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import TypedDict

import httpx

from backend.config import cfg, PROJECT_ROOT

logger = logging.getLogger(__name__)

# STIS engine connection defaults (overridden by config.yaml stis section)
_DEFAULT_HOST = "http://localhost:8100"
_DEFAULT_TIMEOUT = 45.0
_DEFAULT_MAX_NEW_TOKENS = 128


class STISResolution(TypedDict):
    """Result from a successful STIS consensus generation."""
    text: str
    total_tokens: int
    final_similarity: float
    total_iterations: int
    wall_time_secs: float
    axiom_1: str
    axiom_2: str


class STISFallbackResult(TypedDict):
    """Complete result from attempting STIS fallback routing."""
    resolved: bool
    resolution: STISResolution | None
    error: str | None
    latency_secs: float


def _get_stis_config() -> dict[str, object]:
    """Pull STIS connection settings from config.yaml, with safe defaults."""
    stis_cfg = cfg.get("stis", {})
    return {
        "host": stis_cfg.get("host", _DEFAULT_HOST),
        "timeout": float(stis_cfg.get("timeout_secs", _DEFAULT_TIMEOUT)),
        "max_new_tokens": int(stis_cfg.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS)),
        "num_agents": stis_cfg.get("num_agents"),
        "similarity_threshold": stis_cfg.get("similarity_threshold"),
        "alpha": stis_cfg.get("alpha"),
    }


def format_axiom_prompt(query: str, rag_claim: str, graph_claim: str) -> str:
    """Format contradicting contexts as opposing axioms for the STIS engine.

    Structures the conflict explicitly so the swarm agents must reconcile
    both claims through convergence rather than picking one arbitrarily.
    """
    return (
        f"Answer the question using only verified facts. "
        f"One source contains an error.\n\n"
        f"Question: {query}\n\n"
        f"Source A (retrieved document, possibly outdated or wrong):\n{rag_claim}\n\n"
        f"Source B (verified knowledge base):\n{graph_claim}\n\n"
        f"The verified knowledge base (Source B) is authoritative. "
        f"Give a direct, concise answer to the question:"
    )


def request_stis_consensus(
    query: str,
    rag_claim: str,
    graph_claim: str,
) -> STISFallbackResult:
    """Send a contradiction to the STIS engine for swarm consensus resolution.

    Formats the RAG and Graph claims as opposing axioms, posts to /generate,
    and returns the consensus text. Falls back gracefully on any failure.
    """
    t_start = time.perf_counter()
    stis_cfg = _get_stis_config()
    host = str(stis_cfg["host"])
    timeout = float(stis_cfg["timeout"])

    prompt = format_axiom_prompt(query, rag_claim, graph_claim)

    payload: dict[str, object] = {
        "prompt": prompt,
        "max_new_tokens": stis_cfg["max_new_tokens"],
    }
    # pass optional overrides only if configured
    if stis_cfg.get("num_agents") is not None:
        payload["num_agents"] = stis_cfg["num_agents"]
    if stis_cfg.get("similarity_threshold") is not None:
        payload["similarity_threshold"] = stis_cfg["similarity_threshold"]
    if stis_cfg.get("alpha") is not None:
        payload["alpha"] = stis_cfg["alpha"]

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{host}/generate", json=payload)

        latency = time.perf_counter() - t_start

        if resp.status_code != 200:
            error_detail = resp.text[:200] if resp.text else f"HTTP {resp.status_code}"
            logger.warning("STIS engine returned %d: %s", resp.status_code, error_detail)
            return STISFallbackResult(
                resolved=False,
                resolution=None,
                error=f"STIS HTTP {resp.status_code}: {error_detail}",
                latency_secs=round(latency, 4),
            )

        data = resp.json()
        resolution = STISResolution(
            text=data["text"],
            total_tokens=data["total_tokens"],
            final_similarity=data["final_similarity"],
            total_iterations=data["total_iterations"],
            wall_time_secs=data["wall_time_secs"],
            axiom_1=rag_claim[:500],
            axiom_2=graph_claim[:500],
        )

        logger.info(
            "STIS consensus reached: %d tokens, sim=%.4f, iters=%d, wall=%.2fs",
            resolution["total_tokens"], resolution["final_similarity"],
            resolution["total_iterations"], resolution["wall_time_secs"],
        )

        return STISFallbackResult(
            resolved=True,
            resolution=resolution,
            error=None,
            latency_secs=round(latency, 4),
        )

    except httpx.TimeoutException:
        latency = time.perf_counter() - t_start
        logger.warning("STIS engine timed out after %.1fs (limit=%.0fs)", latency, timeout)
        return STISFallbackResult(
            resolved=False,
            resolution=None,
            error=f"STIS timeout after {latency:.1f}s (limit={timeout:.0f}s)",
            latency_secs=round(latency, 4),
        )

    except httpx.ConnectError:
        latency = time.perf_counter() - t_start
        logger.warning("STIS engine unreachable at %s", host)
        return STISFallbackResult(
            resolved=False,
            resolution=None,
            error=f"STIS unreachable at {host}",
            latency_secs=round(latency, 4),
        )

    except Exception as exc:
        latency = time.perf_counter() - t_start
        logger.error("STIS request failed unexpectedly: %s", exc)
        return STISFallbackResult(
            resolved=False,
            resolution=None,
            error=f"STIS error: {exc}",
            latency_secs=round(latency, 4),
        )


def check_stis_health() -> dict[str, object]:
    """Probe the STIS engine health endpoint. Non-blocking, short timeout."""
    stis_cfg = _get_stis_config()
    host = str(stis_cfg["host"])
    try:
        resp = httpx.get(f"{host}/health", timeout=5.0)
        if resp.status_code == 200:
            return {"available": True, **resp.json()}
        return {"available": False, "error": f"HTTP {resp.status_code}"}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def log_stis_resolution(
    query: str,
    rag_claim: str,
    graph_claim: str,
    similarity: float,
    best_cswr: float,
    stis_result: STISFallbackResult,
) -> bool:
    """Log an STIS routing event to SQLite for audit and RL training data.

    Records the contradiction details, resolution outcome, and timing
    in the stis_resolutions table alongside the main episodes table.
    """
    db_path = PROJECT_ROOT / "db" / "rlfo_cache.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stis_resolutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                rag_claim TEXT,
                graph_claim TEXT,
                contradiction_similarity REAL,
                best_cswr REAL,
                resolved INTEGER NOT NULL,
                resolution_text TEXT,
                total_tokens INTEGER,
                final_similarity REAL,
                total_iterations INTEGER,
                stis_wall_time REAL,
                error TEXT,
                latency_secs REAL
            )
        """)

        resolution = stis_result.get("resolution")
        conn.execute("""
            INSERT INTO stis_resolutions
                (query, rag_claim, graph_claim, contradiction_similarity,
                 best_cswr, resolved, resolution_text, total_tokens,
                 final_similarity, total_iterations, stis_wall_time,
                 error, latency_secs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query,
            rag_claim[:500],
            graph_claim[:500],
            similarity,
            best_cswr,
            1 if stis_result["resolved"] else 0,
            resolution["text"] if resolution else None,
            resolution["total_tokens"] if resolution else None,
            resolution["final_similarity"] if resolution else None,
            resolution["total_iterations"] if resolution else None,
            resolution["wall_time_secs"] if resolution else None,
            stis_result.get("error"),
            stis_result["latency_secs"],
        ))

        conn.commit()
        conn.close()
        logger.info("STIS resolution logged: resolved=%s, query='%.50s...'",
                     stis_result["resolved"], query)
        return True

    except Exception as exc:
        logger.error("Failed to log STIS resolution: %s", exc)
        return False

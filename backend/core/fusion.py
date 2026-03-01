# Author: Bradley R. Kinnard
# fusion.py - blends CAG/Graph using default or RL-driven weights
# RAG and Web paths removed per upgrade plan Step 3.

import numpy as np
from typing import List, Dict, Any
from backend.config import cfg

import logging
logger = logging.getLogger(__name__)


def normalize_weights(weights: List[float]) -> List[float]:
    clean = [max(0.0, float(w)) if w is not None and not np.isnan(w) else 0.0 for w in weights]
    total = sum(clean)
    return [w / total for w in clean] if total > 0 else [1.0 / len(clean)] * len(clean)


def get_default_weights() -> List[float]:
    d = cfg["fusion"]["default_weights"]
    return [d["cag"], d["graph"]]


def fuse_context(query: str, rag_results: List[Dict], cag_results: List[Dict],
                 graph_results: List[Dict]) -> Dict[str, Any]:
    """Merge retrieval results using default weights. Two-path: CAG + Graph."""
    weights = get_default_weights()
    cag_w, graph_w = weights
    parts = []

    for c in cag_results:
        if c["score"] >= 0.85:
            parts.append(f"[CAG:{c['score']:.2f}] {c['text']}")

    for g in graph_results:
        if g["score"] >= 0.70:
            parts.append(f"[GRAPH:{g['score']:.2f}] {g['text']}")

    return {
        "context": "\n\n".join(parts),
        "weights": {"cag": cag_w, "graph": graph_w},
        "sources": parts
    }


def format_web_source(result: Dict) -> str:
    """Format web result for clean display - no internal API names."""
    url = result.get("url", "")
    if url.startswith("tavily://"):
        url = "Web Search"
    text = result.get("text", "")[:800]
    title = result.get("title", "").replace("Tavily: ", "")
    if title:
        return f"**{title}**\n{text}"
    return text

# Author: Bradley R. Kinnard
# fusion.py - blends RAG/CAG/Graph/Web using RL policy weights
# Originally built for personal offline use, now open-sourced for public benefit.

import torch
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from stable_baselines3 import PPO
from backend.core.utils import softmax
from backend.config import cfg, PROJECT_ROOT

_policy = None
_policy_loaded = False


def normalize_weights(weights: List[float]) -> List[float]:
    clean = [max(0.0, float(w)) if w is not None and not np.isnan(w) else 0.0 for w in weights]
    total = sum(clean)
    return [w / total for w in clean] if total > 0 else [1.0 / len(clean)] * len(clean)


def load_policy():
    """Load the RL policy if available."""
    global _policy, _policy_loaded
    if _policy_loaded:
        return _policy

    policy_path = PROJECT_ROOT / Path(cfg["rl"]["policy_path"]).name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if policy_path.exists():
        try:
            _policy = PPO.load(str(policy_path), env=None, device=device)
        except Exception:
            _policy = None
    else:
        _policy = None

    _policy_loaded = True
    return _policy


def predict_weights(query_emb: np.ndarray) -> List[float]:
    policy = load_policy()
    if policy is None:
        d = cfg["fusion"]["default_weights"]
        return [d["rag"], d["cag"], d["graph"]]

    action, _ = policy.predict(query_emb.astype(np.float32), deterministic=True)
    logits = action.tolist() if isinstance(action, np.ndarray) else action.cpu().numpy().tolist()
    return softmax(logits, temperature=cfg["fusion"]["temperature"])


def get_default_weights() -> List[float]:
    d = cfg["fusion"]["default_weights"]
    return [d["rag"], d["cag"], d["graph"]]


def fuse_context(query: str, rag_results: List[Dict], cag_results: List[Dict],
                 graph_results: List[Dict]) -> Dict[str, Any]:
    from backend.core.utils import embed_text

    query_emb = embed_text(query)
    policy = None  # load_policy() - disabled, using CQL in main.py

    if policy:
        weights = predict_weights(query_emb)
    else:
        weights = get_default_weights()

    rag_w, cag_w, graph_w = weights
    parts = []

    for r in rag_results:
        if r["score"] >= 0.65:
            parts.append(f"[RAG:{r['score']:.2f}] {r['text']}")

    for c in cag_results:
        if c["score"] >= 0.85:
            parts.append(f"[CAG:{c['score']:.2f}] {c['text']}")

    for g in graph_results:
        if g["score"] >= 0.70:
            parts.append(f"[GRAPH:{g['score']:.2f}] {g['text']}")

    return {
        "context": "\n\n".join(parts),
        "weights": {"rag": rag_w, "cag": cag_w, "graph": graph_w},
        "sources": parts
    }


def format_web_source(result: Dict) -> str:
    """Format web result for clean display - no internal API names"""
    url = result.get("url", "")
    # strip internal tavily:// prefix
    if url.startswith("tavily://"):
        url = "Web Search"
    text = result.get("text", "")[:800]
    title = result.get("title", "").replace("Tavily: ", "")
    if title:
        return f"**{title}**\n{text}"
    return text


def blend_contexts(rag_results: List[Dict], cag_results: List[Dict],
                   graph_results: List[Dict], web_results: List[Dict],
                   weights: List[float]) -> str:
    """
    Blend all sources using RL weights. 5% threshold kills noise.
    Returns clean formatted context for LLM.
    """
    weights = normalize_weights(weights)
    rag_w, cag_w, graph_w, web_w = weights
    parts = []

    if rag_w >= 0.05 and rag_results:
        texts = [r["text"][:800] for r in rag_results[:3] if r.get("score", 0) >= 0.55]
        if texts:
            section = "**Documents**\n\n" + "\n\n".join(texts)
            parts.append(section)

    if cag_w >= 0.05 and cag_results:
        texts = [c["text"][:1000] for c in cag_results[:2] if c.get("score", 0) >= 0.85]
        if texts:
            section = "**Cached Knowledge**\n\n" + "\n\n".join(texts)
            parts.append(section)

    if graph_w >= 0.05 and graph_results:
        texts = [g["text"][:600] for g in graph_results[:3] if g.get("score", 0) >= 0.50]
        if texts:
            section = "**Related Concepts**\n\n" + "\n\n".join(texts)
            parts.append(section)

    if web_w >= 0.05 and web_results:
        texts = [format_web_source(w) for w in web_results[:2] if w.get("score", 0) >= 0.60]
        if texts:
            section = "**Web Sources**\n\n" + "\n\n".join(texts)
            parts.append(section)

    if not parts:
        return "[No relevant context]"

    return "\n\n---\n\n".join(parts)

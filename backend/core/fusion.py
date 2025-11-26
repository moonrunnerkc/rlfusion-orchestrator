# Author: Bradley R. Kinnard
# backend/core/fusion.py
# The grand mixer - takes RAG, CAG, and graph results and blends them together
# using weights learned through PPO. If we haven't trained a policy yet, falls
# back to whatever's in config.yaml (which is probably just a wild guess).

import torch
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from stable_baselines3 import PPO
from backend.core.utils import softmax
from backend.core.retrievers import retrieve
from backend.config import cfg

# Keep the policy in memory - loading from disk every query would be silly
_policy: Any = None
_policy_loaded: bool = False


def load_policy() -> Any:
    """
    Load our trained PPO policy if one exists. If not, no big deal - we'll just
    use the defaults. This gets called once and cached.
    """
    global _policy, _policy_loaded

    if _policy_loaded:
        return _policy

    # Navigate up to project root (because relative paths are the worst)
    project_root = Path(__file__).parent.parent.parent
    policy_filename = Path(cfg["rl"]["policy_path"]).name
    policy_path = project_root / policy_filename

    if policy_path.exists():
        _policy = PPO.load(str(policy_path), env=None)
        print(f"Loaded trained PPO policy from {policy_path}")
    else:
        _policy = None
        print(f"No trained policy found at {policy_path} - using default weights")

    _policy_loaded = True
    return _policy


def predict_weights_with_policy(query_embedding: np.ndarray) -> List[float]:
    """
    Ask the PPO policy what weights to use for this query. If there's no policy,
    we just shrug and use the defaults. Returns [rag_weight, cag_weight, graph_weight].
    """
    policy = load_policy()
    if policy is None:
        # No policy? No problem - use the fallback weights
        defaults = cfg["fusion"]["default_weights"]
        return [defaults["rag"], defaults["cag"], defaults["graph"]]

    # convert to GPU tensor
    obs_tensor = query_embedding.astype(np.float32)   # stay on CPU, pure numpy

    # get action from policy (deterministic for inference)
    action, _ = policy.predict(obs_tensor, deterministic=True)

    # action is raw logits for 3 weights
    if isinstance(action, np.ndarray):
        logits = action.tolist()
    else:
        logits = action.cpu().numpy().tolist()

    # apply softmax with temp
    temp = cfg["fusion"]["temperature"]
    weights = softmax(logits, temperature=temp)

    return weights


def get_default_weights() -> List[float]:
    """Grab default weights from config yaml."""
    defaults = cfg["fusion"]["default_weights"]
    weights = [defaults["rag"], defaults["cag"], defaults["graph"]]

    # these better sum to 1.0 or config is broken
    total = sum(weights)
    assert abs(total - 1.0) < 1e-6, f"Default weights must sum to 1.0, got {total}"

    return weights


def tag_sources(results: List[Dict], source_name: str) -> List[Dict]:
    """Tag each result with source name so we know where it came from."""
    for r in results:
        r["source"] = source_name
    return results


def fuse_context(
    query: str,
    rag_results: List[Dict[str, Any]],
    cag_results: List[Dict[str, Any]],
    graph_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Main fusion function. Only includes results that meet strict RL thresholds.
    """
    from backend.core.utils import embed_text

    query_emb = embed_text(query)
    # RL policy or config weights (unchanged)
    policy = None  # load_policy()
    if policy is not None:
        weights = predict_weights_with_policy(query_emb)
        print(f"Using PPO policy weights: RAG={weights[0]:.3f}, CAG={weights[1]:.3f}, Graph={weights[2]:.3f}")
    else:
        weights = get_default_weights()
        print(f"Using default weights: RAG={weights[0]:.3f}, CAG={weights[1]:.3f}, Graph={weights[2]:.3f}")
    rag_w, cag_w, graph_w = weights

    context_parts = []

    # RAG — always take the best ones above baseline
    for r in rag_results:
        if r["score"] >= 0.65:
            context_parts.append(f"[RAG:{r['score']:.2f}] {r['text']}")

    # CAG — only if really confident
    for c in cag_results:
        if c["score"] >= 0.85:
            context_parts.append(f"[CAG:{c['score']:.2f}] {c['text']}")

    # Graph — only if strong and relevant
    for g in graph_results:
        if g["score"] >= 0.70:
            context_parts.append(f"[GRAPH:{g['score']:.2f}] {g['text']}")

    structured_context = "\n\n".join(context_parts)

    return {
        "context": structured_context,
        "weights": {"rag": rag_w, "cag": cag_w, "graph": graph_w},
        "sources": context_parts
    }


def blend_contexts(retrieval_results: Dict[str, Any], weights: Dict[str, float]) -> str:
    """
    Final, bulletproof blender used by FusionEnv and synthetic data generation.
    Handles both (text, score) tuples and raw strings from retrievers.
    """
    rag = retrieval_results.get("rag", [])
    cag = retrieval_results.get("cag", [])
    graph = retrieval_results.get("graph", "")

    parts = []

    # RAG: handle both string list and (text, score) tuples
    if weights.get("rag", 0) > 0.05 and rag:
        texts = []
        for item in rag[:3]:
            if isinstance(item, tuple) and len(item) >= 1:
                texts.append(item[0])
            elif isinstance(item, str):
                texts.append(item)
        if texts:
            parts.append("[RAG]\n" + "\n".join(texts))

    # CAG: usually list of strings
    if weights.get("cag", 0) > 0.05 and cag:
        texts = [item[0] if isinstance(item, tuple) else str(item) for item in cag[:3]]
        if texts:
            parts.append("[CAG]\n" + "\n".join(texts))

    # Graph: usually a single string
    if weights.get("graph", 0) > 0.05 and graph and str(graph).strip():
        parts.append(f"[Graph]\n{str(graph).strip()}")

    return "\n\n".join(parts) if parts else "No relevant context found."

# Author: Bradley R. Kinnard
# backend/core/fusion.py
# Fusion module - blends RAG/CAG/graph using PPO-learned weights
# Falls back to config.yaml defaults if no policy trained yet

import torch
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from backend.core.utils import softmax
from backend.core.retrievers import retrieve
from backend.config import cfg

# cache the policy so we don't reload on every query
_policy: Any = None
_policy_loaded: bool = False


def load_policy() -> Any:
    """Load PPO policy from disk if it exists."""
    global _policy, _policy_loaded

    if _policy_loaded:
        return _policy

    policy_path = Path(cfg["rl"]["policy_path"])

    if policy_path.exists():
        _policy = torch.load(str(policy_path), map_location="cuda")
        print(f"Loaded trained PPO policy from {cfg['rl']['policy_path']}")
    else:
        _policy = None
        print("No trained policy found - using default weights")

    _policy_loaded = True
    return _policy


def predict_weights_with_policy(query_embedding: np.ndarray) -> List[float]:
    """Use PPO policy to predict weights. Returns [rag, cag, graph]."""
    policy = load_policy()
    if policy is None:
        # no policy trained yet, use defaults
        defaults = cfg["fusion"]["default_weights"]
        return [defaults["rag"], defaults["cag"], defaults["graph"]]

    # convert to GPU tensor
    obs_tensor = torch.from_numpy(query_embedding).float().cuda()

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
    Main fusion function. Combines all three retrieval modes with learned weights.
    """
    from backend.core.utils import embed_text

    query_emb = embed_text(query)

    # get weights from policy or config
    policy = load_policy()
    if policy is not None:
        weights = predict_weights_with_policy(query_emb)
    else:
        weights = get_default_weights()

    rag_w, cag_w, graph_w = weights

    # weight and combine all results
    all_results = []

    for r in rag_results:
        all_results.append({
            "text": r["text"],
            "score": r["score"] * rag_w,
            "source": "rag"
        })

    for r in cag_results:
        all_results.append({
            "text": r["text"],
            "score": r["score"] * cag_w,
            "source": "cag"
        })

    # graph results
    for r in graph_results:
        all_results.append({
            "text": r["text"],
            "score": r["score"] * graph_w,
            "source": "graph"
        })

    # sort by score
    all_results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "fused_context": "\n\n".join([r["text"] for r in all_results]),
        "weights": {"rag": rag_w, "cag": cag_w, "graph": graph_w},
        "sources": all_results
    }

# Author: Bradley R. Kinnard
"""394-dim observation construction and simplex projection.

Single source of truth for:
  - the 10-feature retrieval vector appended to the 384-d query embedding
  - the simplex projection that turns raw action logits into [cag, graph]
    weights with a 0.05 floor (used by both `FusionEnv.step` and the live
    `compute_rl_weights`).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

EMBED_DIM = 384
NUM_FEATURES = 10
OBS_DIM = EMBED_DIM + NUM_FEATURES  # 394
SIMPLEX_FLOOR = 0.05


def build_observation(
    query: str,
    query_embedding: np.ndarray,
    retrieval_results: dict[str, list[dict[str, Any]]] | None,
) -> np.ndarray:
    """Concatenate the 384-d query embedding with the 10 retrieval features.

    Output is float32 with shape (394,). Used by FusionEnv.reset, the live
    fusion path at inference, and the trainer when it needs to rebuild
    observations from logged episode rows.
    """
    embed = np.asarray(query_embedding, dtype=np.float32).flatten()
    if embed.shape[0] != EMBED_DIM:
        raise ValueError(f"query_embedding must be {EMBED_DIM}-d, got {embed.shape[0]}")

    rr = retrieval_results or {}
    cag_results = rr.get("cag", []) or []
    graph_results = rr.get("graph", []) or []

    cache_hit = 1.0 if cag_results else 0.0
    cag_top_score = float(cag_results[0].get("score", 0.0)) if cag_results else 0.0
    graph_density = len(graph_results) / 10.0
    graph_scores = [float(g.get("score", 0.0)) for g in graph_results[:3]]
    graph_top3 = graph_scores + [0.0] * (3 - len(graph_scores))

    query_len = len(query.split()) / 50.0
    q_lower = query.lower()
    query_type = [0.0, 0.0, 0.0]
    if any(w in q_lower for w in ("what is", "who is", "define", "explain")):
        query_type[0] = 1.0
    elif any(
        w in q_lower for w in ("how does", "architecture", "design", "relationship")
    ):
        query_type[1] = 1.0
    else:
        query_type[2] = 1.0

    features = np.array(
        [cache_hit, cag_top_score, graph_density]
        + graph_top3
        + [query_len]
        + query_type,
        dtype=np.float32,
    )
    if features.shape[0] != NUM_FEATURES:
        raise RuntimeError(
            f"feature vector length {features.shape[0]} != {NUM_FEATURES}; "
            "obs_builder is misconfigured."
        )
    return np.concatenate([embed, features]).astype(np.float32)


def project_to_simplex(action: Iterable[float]) -> np.ndarray:
    """Map raw 2D action logits onto the [cag, graph] simplex with a 0.05 floor.

    Equivalent to `softmax(action) * (1 - 2*floor) + floor`, which guarantees
    each path retains at least 5% weight. Used both in FusionEnv.step and the
    live serving path to avoid train/serve divergence.
    """
    arr = np.asarray(list(action), dtype=np.float32).flatten()
    if arr.shape[0] != 2:
        raise ValueError(f"action must be 2D, got shape {arr.shape}")
    shifted = arr - float(np.max(arr))  # numerically stable softmax
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)
    floor = SIMPLEX_FLOOR
    return (probs * (1.0 - 2.0 * floor)) + floor


__all__ = [
    "EMBED_DIM",
    "NUM_FEATURES",
    "OBS_DIM",
    "SIMPLEX_FLOOR",
    "build_observation",
    "project_to_simplex",
]

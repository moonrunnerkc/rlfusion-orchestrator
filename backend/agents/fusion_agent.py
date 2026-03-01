# Author: Bradley R. Kinnard
"""Fusion agent: RL policy inference and weighted context assembly.

Owns the CQL policy prediction step and the fusion of scored retrieval
results into a single context string for the LLM. Two-path architecture:
CAG + Graph only (RAG and Web removed in Step 3).
"""
from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from backend.agents.base import PipelineState, RLPolicy
from backend.config import cfg
from backend.core.utils import embed_text

logger = logging.getLogger(__name__)


def compute_rl_weights(query: str, policy: RLPolicy | None) -> np.ndarray:
    """Compute fusion weights via RL policy or fall back to heuristics.

    Returns 2D weight array [cag, graph]. Backward-compat: expands to 4D
    [0, cag, graph, 0] when legacy callers expect 4-element arrays.
    """
    if policy is not None:
        try:
            obs = embed_text(query).reshape(1, -1)
            if hasattr(policy, "predict"):
                raw = policy.predict(obs)
                action = raw[0] if isinstance(raw, (tuple, list)) or (hasattr(raw, 'ndim') and raw.ndim > 1) else raw
            else:
                action = np.array([0.5, 0.5])

            weights = action.flatten() if hasattr(action, "flatten") else np.array(action)

            # legacy 4D policy: extract cag (idx 1) and graph (idx 2)
            if len(weights) >= 4:
                weights = np.array([weights[1], weights[2]])
            elif len(weights) == 3:
                weights = np.array([weights[1], weights[2]])

            exp_w = np.exp(weights[:2])
            rl_weights = exp_w / np.sum(exp_w)

            # near-uniform: fall back to query heuristics
            if abs(rl_weights[0] - rl_weights[1]) < 0.05:
                logger.info("Policy outputs uniform, applying query heuristics")
                rl_weights = _heuristic_weights(query)

            logger.info("RL weights: CAG=%.2f Graph=%.2f",
                        rl_weights[0], rl_weights[1])
            return rl_weights

        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Policy prediction failed: %s", exc)

    return _heuristic_weights(query)


def _heuristic_weights(query: str) -> np.ndarray:
    """Keyword-based 2-path heuristics when RL policy returns uniform outputs."""
    q = query.lower()
    # entity/relationship queries favor graph
    if any(kw in q for kw in ["how does", "architecture", "design", "workflow", "system", "relationship"]):
        return np.array([0.3, 0.7])
    # factual lookups favor cache
    if any(kw in q for kw in ["what is", "explain", "describe", "define"]):
        return np.array([0.6, 0.4])
    return np.array([0.5, 0.5])


def build_fusion_context(
    retrieval_results: dict[str, list[dict[str, float | str]]],
    weights: np.ndarray,
) -> str:
    """Assemble scored retrieval results into a fused context string.

    Two-path: CAG + Graph only. Filters by per-path score thresholds and
    allocates proportional slots based on the RL fusion weights.
    """
    total_items = 12
    cag_take = max(1, int(weights[0] * total_items))
    graph_take = max(1, int(weights[1] * total_items))

    cag_items = [c for c in retrieval_results.get("cag", []) if c.get("score", 0) >= 0.85][:cag_take]
    graph_items = [g for g in retrieval_results.get("graph", []) if g.get("score", 0) >= 0.50][:graph_take]

    # global relevance gate
    all_scores = (
        [float(c.get("score", 0)) for c in cag_items]
        + [float(g.get("score", 0)) for g in graph_items]
    )
    best_score = max(all_scores) if all_scores else 0.0
    if best_score < 0.52:
        logger.info("Relevance gate: best score %.2f < 0.52, returning empty context", best_score)
        return ""

    parts: list[str] = []
    for c in cag_items:
        parts.append(f"[CAG:{c['score']:.2f}|w={weights[0]:.2f}] {c['text']}")
    for g in graph_items:
        parts.append(f"[GRAPH:{g['score']:.2f}|w={weights[1]:.2f}] {g['text']}")

    logger.info("Fusion stats: %d CAG, %d Graph", len(cag_items), len(graph_items))
    fused = "\n\n".join(parts) if parts else "No high-confidence sources available."
    if len(fused) > 5000:
        fused = fused[:5000]
        logger.info("Fused context truncated to 5000 chars")
    return fused


class FusionAgent:
    """Owns RL policy inference and weighted context assembly.

    Pipeline role: takes retrieval results + RL policy, produces fused context
    string with per-path weight allocation.
    """
    _NAME: ClassVar[str] = "fusion"

    def __init__(self, rl_policy: RLPolicy | None = None) -> None:
        self._rl_policy = rl_policy

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def rl_policy(self) -> RLPolicy | None:
        return self._rl_policy

    @rl_policy.setter
    def rl_policy(self, policy: RLPolicy | None) -> None:
        self._rl_policy = policy

    def plan(self, state: PipelineState) -> PipelineState:
        """Check if RL policy is available, otherwise use heuristic fallback."""
        strategy = "rl_policy" if self._rl_policy is not None else "heuristic"
        logger.debug("[%s] Fusion strategy: %s", self._NAME, strategy)
        return {}  # type: ignore[return-value]

    def act(self, state: PipelineState) -> PipelineState:
        """Compute RL weights and build the fused context string."""
        query = state.get("expanded_query", state.get("query", ""))
        retrieval_results = state.get("retrieval_results", {})

        rl_weights = compute_rl_weights(query, self._rl_policy)
        fused_context = build_fusion_context(retrieval_results, rl_weights)
        # 2-path weights: [cag, graph]
        actual_weights = [float(w) for w in rl_weights[:2]]

        return {  # type: ignore[return-value]
            "rl_weights": actual_weights,
            "fused_context": fused_context,
            "actual_weights": actual_weights,
        }

    def reflect(self, state: PipelineState) -> PipelineState:
        """Check weight distribution for degeneracy (single path dominance)."""
        weights = state.get("actual_weights", [])
        if weights:
            max_w = max(weights)
            if max_w > 0.90:
                logger.warning("[%s] Single path dominates: max_weight=%.2f",
                               self._NAME, max_w)
            fused = state.get("fused_context", "")
            if fused == "No high-confidence sources available.":
                logger.warning("[%s] Empty fusion context, all sources below threshold",
                               self._NAME)
        return {}  # type: ignore[return-value]

    def __call__(self, state: PipelineState) -> PipelineState:
        """LangGraph node interface."""
        self.plan(state)
        updates = self.act(state)
        merged = {**state, **updates}
        self.reflect(merged)  # type: ignore[arg-type]
        return updates

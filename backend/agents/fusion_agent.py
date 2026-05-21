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


def compute_rl_weights(
    query: str,
    policy: RLPolicy | None,
    retrieval_results: dict[str, list[dict[str, float | str]]] | None = None,
) -> np.ndarray:
    """Compute fusion weights via RL policy or fall back to heuristics.

    Returns 2D weight array [cag, graph]. Adjusts weights post-hoc based on
    actual retrieval counts: if a path returned 0 results, its weight shifts
    to the path that did return results.
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

        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Policy prediction failed: %s", exc)
            rl_weights = _heuristic_weights(query)
    else:
        rl_weights = _heuristic_weights(query)

    # result-aware rebalancing: shift weight away from empty paths
    if retrieval_results is not None:
        cag_count = len(retrieval_results.get("cag", []))
        graph_count = len(retrieval_results.get("graph", []))
        if cag_count == 0 and graph_count > 0:
            # CAG missed, lean heavy on graph
            rl_weights = np.array([0.05, 0.95])
            logger.info("Rebalanced: CAG empty, shifting to Graph")
        elif graph_count == 0 and cag_count > 0:
            # graph empty, lean heavy on cache
            rl_weights = np.array([0.95, 0.05])
            logger.info("Rebalanced: Graph empty, shifting to CAG")
        elif cag_count == 0 and graph_count == 0:
            # both empty, equal (won't matter, context will be empty)
            rl_weights = np.array([0.5, 0.5])
            logger.info("Rebalanced: both paths empty")

    logger.info("RL weights: CAG=%.2f Graph=%.2f",
                rl_weights[0], rl_weights[1])
    return rl_weights


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


def _csw_rerank_graph(
    graph_items: list[dict[str, float | str]],
    query: str,
) -> list[dict[str, float | str]]:
    """Run CSWR scoring on graph items in-place and return them sorted by csw_score.

    The graph items must carry a `text` and a `score` (the cosine sim from
    retrieve()). score_chunks() reads `score` as the vector axis and writes
    `csw_score`, `local_stability`, `question_fit`, `drift_penalty` onto
    each item.
    """
    from backend.core.decomposer import decompose_query
    from backend.core.retrievers import score_chunks
    from backend.config import cfg as _cfg

    if not graph_items:
        return graph_items

    profile = decompose_query(query)
    profile["query_text"] = query

    for i, item in enumerate(graph_items):
        item.setdefault("id", str(item.get("id") or f"g{i}"))
        item.setdefault("local_stability", 0.0)
        item.setdefault("question_fit", 0.0)
        item.setdefault("drift_penalty", 0.0)
        item.setdefault("csw_score", 0.0)
        item.setdefault("project_coherence", 1.0)

    return score_chunks(list(graph_items), profile, _cfg.get("cswr", {}))


def build_fusion_context(
    retrieval_results: dict[str, list[dict[str, float | str]]],
    weights: np.ndarray,
    query: str = "",
) -> str:
    """Assemble scored retrieval results into a fused context string.

    Two-path: CAG + Graph. CAG entries are short, exact-match cached
    answers that bypass CSWR. Graph results are re-ranked by CSWR
    (`score_chunks`) and filtered by `cswr.min_csw_score` before slot
    allocation. The RL fusion weights determine how many slots each path
    gets out of a total of 12.
    """
    from backend.api.chunk_safety import filter_adversarial_chunks
    from backend.config import cfg as _cfg

    total_items = 12
    cag_take = max(1, int(weights[0] * total_items))
    graph_take = max(1, int(weights[1] * total_items))

    # Drop chunks that look like prompt-injection payloads BEFORE slot
    # allocation. A poisoned CAG entry or a malicious PDF page can carry
    # the same "ignore previous instructions" text the safety agent
    # already screens out of the user's query.
    safe_cag = filter_adversarial_chunks(
        list(retrieval_results.get("cag", [])), source_label="cag",
    )
    cag_cache_threshold = float(_cfg.get("cag", {}).get("cache_threshold", 0.85))
    cag_items = [c for c in safe_cag if c.get("score", 0) >= cag_cache_threshold][:cag_take]

    # CSWR re-rank graph results, gate by min_csw_score
    raw_graph = filter_adversarial_chunks(
        list(retrieval_results.get("graph", [])), source_label="graph",
    )
    min_csw = float(_cfg.get("cswr", {}).get("min_csw_score", 0.25))
    if raw_graph and query:
        scored_graph = _csw_rerank_graph(raw_graph, query)
        graph_items = [
            g for g in scored_graph if float(g.get("csw_score", 0.0)) >= min_csw
        ][:graph_take]
    else:
        # no query supplied: fall back to raw cosine gate so we still emit
        # something on training-time / replay paths that do not pass query
        graph_items = [
            g for g in raw_graph if g.get("score", 0) >= 0.50
        ][:graph_take]

    # global relevance gate: best across paths must clear 0.52
    all_scores = (
        [float(c.get("score", 0)) for c in cag_items]
        + [float(g.get("csw_score", g.get("score", 0))) for g in graph_items]
    )
    best_score = max(all_scores) if all_scores else 0.0
    if best_score < 0.52:
        logger.info(
            "Relevance gate: best score %.2f < 0.52, returning empty context",
            best_score,
        )
        return ""

    parts: list[str] = []
    for c in cag_items:
        parts.append(f"[CAG:{c['score']:.2f}|w={weights[0]:.2f}] {c['text']}")
    for g in graph_items:
        # show csw_score in the tag when present; fall back to raw score
        tag_score = float(g.get("csw_score", g.get("score", 0)))
        parts.append(f"[GRAPH:{tag_score:.2f}|w={weights[1]:.2f}] {g['text']}")

    logger.info(
        "Fusion stats: %d CAG, %d Graph (min_csw=%.2f)",
        len(cag_items), len(graph_items), min_csw,
    )
    fused = "\n\n".join(parts) if parts else ""
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

        rl_weights = compute_rl_weights(query, self._rl_policy, retrieval_results)
        fused_context = build_fusion_context(retrieval_results, rl_weights, query=query)
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
            if not fused.strip():
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

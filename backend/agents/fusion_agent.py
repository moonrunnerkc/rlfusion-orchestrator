# Author: Bradley R. Kinnard
"""Fusion agent: RL policy inference and weighted context assembly.

Owns the CQL policy prediction step and the fusion of scored retrieval
results into a single context string for the LLM. Two-path architecture:
CAG + Graph only (RAG and Web removed in Step 3).
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

from backend.agents.base import PipelineState, RLPolicy
from backend.api.injection_filter import scrub_chunks, wrap_untrusted
from backend.config import cfg
from backend.core.utils import embed_text
from backend.rl.obs_builder import build_observation, project_to_simplex

logger = logging.getLogger(__name__)


def compute_rl_weights(
    query: str,
    policy: RLPolicy | None,
    retrieval_results: dict[str, list[dict[str, float | str]]] | None = None,
    *,
    return_telemetry: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Compute fusion weights via RL policy or fall back to heuristics.

    Returns 2D weight array [cag, graph]. If `return_telemetry=True`, also
    returns a dict with `policy_weights` (before rebalancing), `policy_action`
    (raw pre-softmax tensor), `effective_weights` (after rebalancing), and
    `had_empty_path` (bool). The CAG/Graph result-aware rebalance overrides
    the policy when a path returned zero results — the trainer reads
    `policy_weights` so it learns from the policy's actual decision, not the
    override.
    """
    raw_action: list[float] | None = None
    policy_weights = np.array([0.5, 0.5], dtype=np.float32)

    if policy is not None:
        try:
            embed = embed_text(query)
            obs = build_observation(query, embed, retrieval_results)
            if hasattr(policy, "predict"):
                raw = policy.predict(np.array([obs]))
                action = raw[0] if isinstance(raw, (tuple, list)) or (hasattr(raw, 'ndim') and raw.ndim > 1) else raw
            else:
                action = np.array([0.5, 0.5])

            arr = action.flatten() if hasattr(action, "flatten") else np.array(action)
            # legacy 4D policy: extract cag (idx 1) and graph (idx 2)
            if len(arr) >= 4:
                arr = np.array([arr[1], arr[2]])
            elif len(arr) == 3:
                arr = np.array([arr[1], arr[2]])
            raw_action = [float(x) for x in arr[:2]]
            policy_weights = project_to_simplex(arr[:2])

            # near-uniform output: fall back to query heuristics
            if abs(policy_weights[0] - policy_weights[1]) < 0.05:
                logger.info("Policy outputs uniform, applying query heuristics")
                policy_weights = _heuristic_weights(query)

        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Policy prediction failed: %s", exc)
            policy_weights = _heuristic_weights(query)
    else:
        policy_weights = _heuristic_weights(query)

    effective_weights = policy_weights.copy()
    had_empty_path = False

    if retrieval_results is not None:
        cag_count = len(retrieval_results.get("cag", []))
        graph_count = len(retrieval_results.get("graph", []))
        if cag_count == 0 and graph_count > 0:
            effective_weights = np.array([0.05, 0.95], dtype=np.float32)
            had_empty_path = True
            logger.info("Rebalanced: CAG empty, shifting to Graph")
        elif graph_count == 0 and cag_count > 0:
            effective_weights = np.array([0.95, 0.05], dtype=np.float32)
            had_empty_path = True
            logger.info("Rebalanced: Graph empty, shifting to CAG")
        elif cag_count == 0 and graph_count == 0:
            effective_weights = np.array([0.5, 0.5], dtype=np.float32)
            had_empty_path = True
            logger.info("Rebalanced: both paths empty")

    logger.info(
        "RL weights: CAG=%.2f Graph=%.2f (policy=[%.2f,%.2f], rebalance=%s)",
        effective_weights[0], effective_weights[1],
        policy_weights[0], policy_weights[1], had_empty_path,
    )
    if return_telemetry:
        telemetry: dict[str, Any] = {
            "policy_weights": [float(policy_weights[0]), float(policy_weights[1])],
            "effective_weights": [float(effective_weights[0]), float(effective_weights[1])],
            "had_empty_path": had_empty_path,
            "policy_action": raw_action,
        }
        return effective_weights, telemetry
    return effective_weights


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
    from backend.config import cfg as _cfg

    total_items = 12
    cag_take = max(1, int(weights[0] * total_items))
    graph_take = max(1, int(weights[1] * total_items))

    # F0.6: screen every retrieved chunk for prompt-injection patterns before
    # it can reach the LLM. CAG entries are also scrubbed because the cache
    # can be poisoned by upstream high-reward turns that contained payloads.
    cag_raw = scrub_chunks(list(retrieval_results.get("cag", [])))
    cache_thresh = float(_cfg.get("cag", {}).get("cache_threshold", 0.85))
    cag_items = [
        c for c in cag_raw if c.get("score", 0) >= cache_thresh
    ][:cag_take]

    # CSWR re-rank graph results, gate by min_csw_score
    raw_graph = scrub_chunks(list(retrieval_results.get("graph", [])))
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
    body = "\n\n".join(parts) if parts else ""
    fused = wrap_untrusted(body) if body else ""
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

        rl_weights, telemetry = compute_rl_weights(
            query, self._rl_policy, retrieval_results, return_telemetry=True,
        )
        fused_context = build_fusion_context(retrieval_results, rl_weights, query=query)
        actual_weights = [float(w) for w in rl_weights[:2]]

        return {  # type: ignore[return-value]
            "rl_weights": actual_weights,
            "fused_context": fused_context,
            "actual_weights": actual_weights,
            "policy_weights": telemetry["policy_weights"],
            "effective_weights": telemetry["effective_weights"],
            "had_empty_path": telemetry["had_empty_path"],
            "policy_action": telemetry["policy_action"],
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

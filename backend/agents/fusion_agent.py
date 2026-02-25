# Author: Bradley R. Kinnard
"""Fusion agent: RL policy inference and weighted context assembly.

Owns the CQL/PPO policy prediction step and the fusion of scored retrieval
results into a single context string for the LLM. Moved from _compute_rl_fusion_weights
and _build_fusion_context in main.py so agents can be tested independently.
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

    Takes a query and optional policy, returns 4D weight array [rag, cag, graph, web].
    When the policy produces near-uniform outputs, applies query-type heuristics.
    """
    web_enabled = cfg.get("web", {}).get("enabled", False)

    if policy is not None:
        try:
            obs = embed_text(query).reshape(1, -1)
            # CQLPolicyWrapper returns array directly; PPO needs deterministic flag
            if hasattr(policy, "predict"):
                raw = policy.predict(obs)
                action = raw[0] if isinstance(raw, (tuple, list)) or (hasattr(raw, 'ndim') and raw.ndim > 1) else raw
            else:
                action = np.array([0.25, 0.25, 0.25, 0.25])

            weights = action.flatten() if hasattr(action, "flatten") else np.array(action)
            if len(weights) == 3:
                weights = np.append(weights, 0.0)

            exp_w = np.exp(weights)
            rl_weights = exp_w / np.sum(exp_w)

            # near-uniform policy output: fall back to query heuristics
            if max(rl_weights) - min(rl_weights) < 0.05:
                logger.info("Policy outputs uniform, applying query heuristics")
                rl_weights = _heuristic_weights(query)

            if not web_enabled:
                rl_weights[3] = 0.0
                if np.sum(rl_weights[:3]) > 0:
                    rl_weights[:3] = rl_weights[:3] / np.sum(rl_weights[:3])

            logger.info("RL weights: RAG=%.2f CAG=%.2f Graph=%.2f Web=%.2f",
                        rl_weights[0], rl_weights[1], rl_weights[2], rl_weights[3])
            return rl_weights

        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Policy prediction failed: %s", exc)

    fallback = np.array([0.33, 0.33, 0.34, 0.0]) if not web_enabled else np.array([0.25, 0.25, 0.25, 0.25])
    return fallback


def _heuristic_weights(query: str) -> np.ndarray:
    """Keyword-based weight heuristics when RL policy returns uniform outputs."""
    q = query.lower()
    if any(kw in q for kw in ["http://", "https://", "website", ".com", "look up"]):
        return np.array([0.2, 0.1, 0.2, 0.5])
    if any(kw in q for kw in ["how does", "architecture", "design", "workflow", "system"]):
        return np.array([0.2, 0.1, 0.6, 0.1])
    if any(kw in q for kw in ["what is", "explain", "describe", "document"]):
        return np.array([0.6, 0.2, 0.1, 0.1])
    return np.array([0.4, 0.2, 0.3, 0.1])


def build_fusion_context(
    retrieval_results: dict[str, list[dict[str, float | str]]],
    weights: np.ndarray,
) -> str:
    """Assemble scored retrieval results into a fused context string.

    Filters by per-path score thresholds and allocates proportional slots
    based on the RL fusion weights. If no results clear the relevance bar,
    returns empty context so the LLM answers from its own knowledge.
    """
    total_items = 15
    rag_take = max(2, int(weights[0] * total_items))
    cag_take = max(1, int(weights[1] * total_items))
    graph_take = max(1, int(weights[2] * total_items))
    web_take = max(1, int(weights[3] * total_items)) if len(weights) > 3 else 0

    # Relevance thresholds: only include chunks the model can actually use
    rag_items = [r for r in retrieval_results.get("rag", []) if r.get("score", 0) >= 0.50][:rag_take]
    cag_items = [c for c in retrieval_results.get("cag", []) if c.get("score", 0) >= 0.85][:cag_take]
    graph_items = [g for g in retrieval_results.get("graph", []) if g.get("score", 0) >= 0.50][:graph_take]
    web_items = (
        [w for w in retrieval_results.get("web", []) if w.get("score", 0) >= 0.60][:web_take]
        if web_take > 0
        else []
    )

    # Global relevance gate: if the BEST score across all paths is noise-level,
    # don't send any context. Prevents the LLM from shoehorning unrelated chunks.
    # Threshold calibrated against CSWR composite scores for this corpus:
    # genuine matches score 0.55-0.65, noise scores 0.35-0.50.
    all_scores = (
        [float(r.get("score", 0)) for r in rag_items]
        + [float(c.get("score", 0)) for c in cag_items]
        + [float(g.get("score", 0)) for g in graph_items]
        + [float(w.get("score", 0)) for w in web_items]
    )
    best_score = max(all_scores) if all_scores else 0.0
    if best_score < 0.52:
        logger.info("Relevance gate: best score %.2f < 0.52, returning empty context", best_score)
        return ""

    parts: list[str] = []
    for r in rag_items:
        parts.append(f"[RAG:{r['score']:.2f}|w={weights[0]:.2f}] {r['text']}")
    for c in cag_items:
        parts.append(f"[CAG:{c['score']:.2f}|w={weights[1]:.2f}] {c['text']}")
    for g in graph_items:
        parts.append(f"[GRAPH:{g['score']:.2f}|w={weights[2]:.2f}] {g['text']}")
    for w in web_items:
        parts.append(
            f"[WEB:{w['score']:.2f}|w={weights[3]:.2f}] Source: {w.get('url', 'unknown')}\n{str(w['text'])[:600]}"
        )

    # Cross-project divergence gate: if RAG chunks from multiple projects
    # survived, check if they're duplicative (same concept from two sources).
    # Keep only the highest-scoring project group when overlap > 0.80.
    rag_projects: dict[str, list[dict]] = {}
    for r in rag_items:
        proj = r.get("project", "default")
        rag_projects.setdefault(proj, []).append(r)

    if len(rag_projects) > 1:
        from backend.core.utils import embed_batch as _eb
        group_means = {}
        for proj, items in rag_projects.items():
            embs = _eb([str(item["text"]) for item in items])
            group_means[proj] = embs.mean(axis=0)

        # pairwise similarity between project groups
        projs = list(group_means.keys())
        for i in range(len(projs)):
            for j in range(i + 1, len(projs)):
                sim = float(np.dot(group_means[projs[i]], group_means[projs[j]]) /
                           (np.linalg.norm(group_means[projs[i]]) * np.linalg.norm(group_means[projs[j]]) + 1e-9))
                if sim > 0.80:
                    # duplicative: keep the group with higher avg score
                    avg_i = sum(r.get("score", 0) for r in rag_projects[projs[i]]) / len(rag_projects[projs[i]])
                    avg_j = sum(r.get("score", 0) for r in rag_projects[projs[j]]) / len(rag_projects[projs[j]])
                    loser = projs[j] if avg_i >= avg_j else projs[i]
                    # rebuild parts without the losing project's RAG chunks
                    parts = [p for p in parts if not any(
                        p.startswith(f"[RAG:") and r["text"] in p
                        for r in rag_projects[loser]
                    )]
                    logger.info("Divergence gate: dropped project '%s' (sim=%.2f, duplicative)", loser, sim)

    logger.info("Fusion stats: %d RAG, %d CAG, %d Graph, %d Web",
                len(rag_items), len(cag_items), len(graph_items), len(web_items))
    fused = "\n\n".join(parts) if parts else "No high-confidence sources available."
    # cap context to ~5000 chars (~1250 tokens) to fit within num_ctx budget
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
        actual_weights = [float(w) for w in rl_weights[:4]] if len(rl_weights) >= 4 else [float(w) for w in rl_weights] + [0.0]

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

# Author: Bradley R. Kinnard
"""Retrieval agent: owns CAG and GraphRAG paths + CSWR.

Wraps retrieve() from backend.core.retrievers. CAG-first with GraphRAG fallback.
RAG (FAISS) and Web (Tavily) paths removed in Step 3 upgrade.
"""
from __future__ import annotations

import logging
from typing import ClassVar

from backend.agents.base import PipelineState

logger = logging.getLogger(__name__)

# Retrieval depth multipliers per complexity level
_DEPTH_MULTIPLIERS: dict[str, int] = {
    "simple": 1,
    "complex": 2,
    "adversarial": 1,
}


class RetrievalAgent:
    """Owns CAG/Graph retrieval and CSWR scoring.

    Pipeline role: converts a query into scored, ranked retrieval results
    via CAG-first with GraphRAG fallback. Depth scales with query complexity.
    """
    _NAME: ClassVar[str] = "retrieval"

    @property
    def name(self) -> str:
        return self._NAME

    def plan(self, state: PipelineState) -> PipelineState:
        """Determine retrieval depth based on query complexity."""
        complexity = state.get("complexity", "complex")
        multiplier = _DEPTH_MULTIPLIERS.get(complexity, 1)
        query = state.get("expanded_query", state.get("query", ""))
        logger.debug("[%s] Planning retrieval: complexity=%s, depth_mult=%d, query_len=%d",
                     self._NAME, complexity, multiplier, len(query))
        return {}  # type: ignore[return-value]

    def act(self, state: PipelineState) -> PipelineState:
        """Execute retrieval: CAG-first, then GraphRAG on miss."""
        from backend.core.retrievers import retrieve

        # prefer expanded query if available (memory enriched)
        query = state.get("expanded_query", state.get("query", ""))
        complexity = state.get("complexity", "complex")
        base_top_k = 5
        multiplier = _DEPTH_MULTIPLIERS.get(complexity, 1)
        top_k = base_top_k * multiplier

        retrieval_results = retrieve(query, top_k=top_k)
        web_status = retrieval_results.get("web_status", "disabled")

        return {  # type: ignore[return-value]
            "retrieval_results": retrieval_results,
            "web_status": web_status,
        }

    def reflect(self, state: PipelineState) -> PipelineState:
        """Evaluate retrieval quality: log coverage across paths."""
        results = state.get("retrieval_results", {})
        cag_count = len(results.get("cag", []))
        graph_count = len(results.get("graph", []))
        total = cag_count + graph_count

        if total == 0:
            logger.warning("[%s] No retrieval results for query", self._NAME)
        else:
            logger.info("[%s] Retrieval: %d CAG, %d Graph",
                        self._NAME, cag_count, graph_count)
        return {}  # type: ignore[return-value]

    def __call__(self, state: PipelineState) -> PipelineState:
        """LangGraph node interface."""
        self.plan(state)
        updates = self.act(state)
        merged = {**state, **updates}
        self.reflect(merged)  # type: ignore[arg-type]
        return updates

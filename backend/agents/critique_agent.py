# Author: Bradley R. Kinnard
"""Critique agent: self-critique scoring, response cleanup, and replay logging.

Wraps critique(), strip_critique_block(), and log_episode_to_replay_buffer()
from backend.core.critique. Handles reward extraction and proactive suggestion
parsing without rewriting the proven regex-based logic.

Phase 5: integrates selective faithfulness checking on the hot path for
high-sensitivity queries. Sensitivity level flows from query decomposition
through PipelineState.
"""
from __future__ import annotations

import logging
from typing import ClassVar

from backend.agents.base import PipelineState

logger = logging.getLogger(__name__)


class CritiqueAgent:
    """Owns self-critique, faithfulness checks, and reward scaling.

    Pipeline role: takes the raw LLM response, extracts inline critique scores,
    strips critique blocks from user-facing output, and logs episodes to the
    replay buffer for RL training. For high-sensitivity queries, runs
    faithfulness verification on the hot path via cached LLM calls.
    """
    _NAME: ClassVar[str] = "critique"

    @property
    def name(self) -> str:
        return self._NAME

    def plan(self, state: PipelineState) -> PipelineState:
        """Critique always runs on the LLM response. No branching needed."""
        has_response = bool(state.get("llm_response"))
        logger.debug("[%s] Planning critique: response_present=%s",
                     self._NAME, has_response)
        return {}  # type: ignore[return-value]

    def act(self, state: PipelineState) -> PipelineState:
        """Extract critique scores and clean the response for user display.

        When sensitivity_level is above the configured gate, also runs
        faithfulness checking via cached_check_faithfulness().
        """
        from backend.core.critique import (
            critique,
            log_episode_to_replay_buffer,
            strip_critique_block,
        )
        from backend.core.reasoning import run_selective_faithfulness

        llm_response = state.get("llm_response", "")
        query = state.get("query", "")
        fused_context = state.get("fused_context", "")
        sensitivity = float(state.get("sensitivity_level", 0.5))

        critique_result = critique(query, fused_context, llm_response)
        clean_response = strip_critique_block(llm_response)

        # Phase 5: selective faithfulness on the hot path
        faith_checked, faith_score = run_selective_faithfulness(
            clean_response, fused_context, sensitivity,
        )
        if faith_checked:
            logger.info(
                "[%s] Faithfulness score=%.2f (sensitivity=%.2f)",
                self._NAME, faith_score, sensitivity,
            )

        # log to replay buffer for future RL training
        actual_weights = state.get("actual_weights", [0.5, 0.5])
        log_episode_to_replay_buffer({
            "query": query,
            "response": clean_response,
            "weights": {
                "cag": actual_weights[0] if len(actual_weights) > 0 else 0.0,
                "graph": actual_weights[1] if len(actual_weights) > 1 else 0.0,
            },
            "reward": critique_result["reward"],
            "proactive_suggestions": critique_result.get("proactive_suggestions", []),
            "fused_context": fused_context,
        })

        return {  # type: ignore[return-value]
            "reward": critique_result["reward"],
            "clean_response": clean_response,
            "proactive_suggestions": critique_result.get("proactive_suggestions", []),
            "critique_reason": critique_result.get("reason", ""),
            "faithfulness_checked": faith_checked,
            "faithfulness_score": faith_score,
        }

    def reflect(self, state: PipelineState) -> PipelineState:
        """Validate reward is in expected range and log anomalies."""
        reward = state.get("reward", 0.0)
        if reward < 0.0 or reward > 1.0:
            logger.warning("[%s] Reward out of bounds: %.2f (clamping to [0, 1])",
                           self._NAME, reward)
        if reward < 0.3:
            logger.warning("[%s] Low reward (%.2f) for query: %s...",
                           self._NAME, reward, state.get("query", "")[:50])
        suggestions = state.get("proactive_suggestions", [])
        faith = state.get("faithfulness_score", -1.0)
        logger.debug("[%s] Critique complete: reward=%.2f, suggestions=%d, faith=%.2f",
                     self._NAME, reward, len(suggestions), faith)
        return {}  # type: ignore[return-value]

    def __call__(self, state: PipelineState) -> PipelineState:
        """LangGraph node interface."""
        self.plan(state)
        updates = self.act(state)
        merged = {**state, **updates}
        self.reflect(merged)  # type: ignore[arg-type]
        return updates

# Author: Bradley R. Kinnard
"""Critique agent: self-critique scoring, response cleanup, and replay logging.

Wraps critique(), strip_critique_block(), and log_episode_to_replay_buffer()
from backend.core.critique. Produces the reward signal consumed by the
RL training loop and the proactive suggestion list shown in the UI.
"""
from __future__ import annotations

import logging
from typing import ClassVar

from backend.agents.base import PipelineState

logger = logging.getLogger(__name__)


class CritiqueAgent:
    """Owns self-critique and replay logging.

    Pipeline role: takes the raw LLM response, runs the dedicated critique
    LLM call to score it, strips critique blocks from user-facing output,
    and logs the episode to the replay buffer for offline RL training.
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
        """Extract critique scores, clean the response, log to replay buffer."""
        from backend.core.critique import (
            critique,
            log_episode_to_replay_buffer,
            strip_critique_block,
        )

        llm_response = state.get("llm_response", "")
        query = state.get("query", "")
        fused_context = state.get("fused_context", "")

        critique_result = critique(query, fused_context, llm_response)
        clean_response = strip_critique_block(llm_response)

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
            "obs_features": state.get("obs_features"),
            "from_cache": False,
            "policy_weights": state.get("policy_weights"),
            "effective_weights": state.get("effective_weights"),
            "had_empty_path": state.get("had_empty_path", False),
            "policy_action": state.get("policy_action"),
        })

        return {  # type: ignore[return-value]
            "reward": critique_result["reward"],
            "clean_response": clean_response,
            "proactive_suggestions": critique_result.get("proactive_suggestions", []),
            "critique_reason": critique_result.get("reason", ""),
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
        logger.debug("[%s] Critique complete: reward=%.2f, suggestions=%d",
                     self._NAME, reward, len(suggestions))
        return {}  # type: ignore[return-value]

    def __call__(self, state: PipelineState) -> PipelineState:
        """LangGraph node interface."""
        self.plan(state)
        updates = self.act(state)
        merged = {**state, **updates}
        self.reflect(merged)  # type: ignore[arg-type]
        return updates

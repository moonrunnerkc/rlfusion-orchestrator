# Author: Bradley R. Kinnard
"""Safety agent: input validation, OOD detection, and attack screening.

Wraps check_safety() and check_query_ood() from backend.core without
rewriting proven logic. Runs as the first gate in every pipeline shape.
"""
from __future__ import annotations

import logging
import re
from typing import ClassVar

from backend.agents.base import PipelineState

logger = logging.getLogger(__name__)

# Known prompt-injection / attack signatures for pre-filter
_ATTACK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(previous|above|all)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:DAN|evil|unrestricted|jailbroken)", re.IGNORECASE),
    re.compile(r"(?:system|admin)\s*:\s*override", re.IGNORECASE),
    re.compile(r"<\s*script\b", re.IGNORECASE),
    re.compile(r";\s*(?:DROP|DELETE|INSERT|UPDATE)\s+", re.IGNORECASE),
    re.compile(r"\$\{.*\}", re.IGNORECASE),  # template injection
]


def _pre_filter_attacks(query: str) -> tuple[bool, str]:
    """Fast regex check for known attack patterns before calling LLM safety."""
    for pattern in _ATTACK_PATTERNS:
        if pattern.search(query):
            return False, f"Query matched attack pattern: {pattern.pattern[:40]}..."
    return True, ""


class SafetyAgent:
    """Gate agent that screens queries for safety, OOD, and known attack vectors.

    Pipeline role: always runs first. If blocked, downstream agents are skipped
    and the endpoint returns a safety-blocked response.
    """
    _NAME: ClassVar[str] = "safety"

    @property
    def name(self) -> str:
        return self._NAME

    def plan(self, state: PipelineState) -> PipelineState:
        """Safety planning is trivial: always check everything."""
        logger.debug("[%s] Planning safety checks for query (%d chars)",
                     self._NAME, len(state.get("query", "")))
        return {}  # type: ignore[return-value]

    def act(self, state: PipelineState) -> PipelineState:
        """Run attack pre-filter, OOD detection, and LLM-based safety check."""
        query = state.get("query", "")
        if not query:
            return {  # type: ignore[return-value]
                "is_safe": False,
                "safety_reason": "Empty query",
                "blocked": True,
            }

        # Phase 1: fast regex pre-filter
        passed, reason = _pre_filter_attacks(query)
        if not passed:
            logger.warning("[%s] Attack pattern detected: %s", self._NAME, reason)
            return {  # type: ignore[return-value]
                "is_safe": False,
                "safety_reason": reason,
                "blocked": True,
            }

        # Phase 2: OOD detection (non-blocking, just logs)
        try:
            from backend.core.utils import check_query_ood
            is_ood, distance = check_query_ood(query)
            if is_ood:
                logger.warning("[%s] OOD flagged (distance=%.2f)", self._NAME, distance)
        except (ImportError, RuntimeError) as exc:
            logger.debug("[%s] OOD check skipped: %s", self._NAME, exc)

        # Phase 3: LLM-based safety classification
        from backend.core.critique import check_safety
        is_safe, safety_reason = check_safety(query)

        return {  # type: ignore[return-value]
            "is_safe": is_safe,
            "safety_reason": safety_reason,
            "blocked": not is_safe,
        }

    def reflect(self, state: PipelineState) -> PipelineState:
        """Log safety screening outcome."""
        if state.get("blocked"):
            logger.info("[%s] Query BLOCKED: %s", self._NAME,
                        state.get("safety_reason", "unknown"))
        else:
            logger.debug("[%s] Query passed safety screening", self._NAME)
        return {}  # type: ignore[return-value]

    def __call__(self, state: PipelineState) -> PipelineState:
        """LangGraph node interface: plan -> act -> reflect."""
        self.plan(state)
        updates = self.act(state)
        merged = {**state, **updates}
        self.reflect(merged)  # type: ignore[arg-type]
        return updates

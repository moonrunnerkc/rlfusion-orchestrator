# Author: Bradley R. Kinnard
"""ToolRegistry: registration, discovery, validation, and rate-limited dispatch.

Central hub for all tools. Validates conformance to BaseTool at registration,
enforces per-tool rate limits, routes queries to the best-matching tool(s)
based on keyword heuristics (LLM-based selection is Phase 3.2, layered on top).
"""
from __future__ import annotations

import logging
import time
import threading
from typing import ClassVar

from backend.tools.base import BaseTool, ToolInput, ToolOutput, make_output, timed_execute

logger = logging.getLogger(__name__)

# default per-tool rate limit: 10 calls per 60 seconds
_DEFAULT_MAX_CALLS = 10
_DEFAULT_WINDOW_SECS = 60.0


class _RateBucket:
    """Sliding-window rate limiter for a single tool."""

    def __init__(self, max_calls: int, window_secs: float) -> None:
        self._max_calls = max_calls
        self._window = window_secs
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = time.monotonic()
        with self._lock:
            self._timestamps = [t for t in self._timestamps if now - t < self._window]
            if len(self._timestamps) >= self._max_calls:
                return False
            self._timestamps.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._timestamps.clear()


class ToolRegistry:
    """Thread-safe registry for tool instances.

    Validates each tool at registration, tracks per-tool rate limits,
    and provides keyword-based tool selection for query routing.
    """
    _NAME: ClassVar[str] = "tool_registry"

    def __init__(
        self,
        max_calls_per_tool: int = _DEFAULT_MAX_CALLS,
        window_secs: float = _DEFAULT_WINDOW_SECS,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._rate_buckets: dict[str, _RateBucket] = {}
        self._max_calls = max_calls_per_tool
        self._window = window_secs

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def register(self, tool: BaseTool) -> None:
        """Add a tool to the registry. Validates protocol conformance first."""
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"Tool must satisfy BaseTool protocol. Got {type(tool).__name__} "
                f"which is missing required attributes/methods."
            )
        if not tool.name:
            raise ValueError("Tool name must be a non-empty string.")
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: '{tool.name}' is already registered.")

        self._tools[tool.name] = tool
        self._rate_buckets[tool.name] = _RateBucket(self._max_calls, self._window)
        logger.info("Registered tool: %s", tool.name)

    def unregister(self, tool_name: str) -> None:
        """Remove a tool from the registry."""
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' is not registered.")
        del self._tools[tool_name]
        del self._rate_buckets[tool_name]
        logger.info("Unregistered tool: %s", tool_name)

    def get(self, tool_name: str) -> BaseTool:
        """Retrieve a tool by exact name. Raises KeyError if not found."""
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' is not registered. Available: {self.tool_names}")
        return self._tools[tool_name]

    def list_tools(self) -> list[dict[str, str]]:
        """Return metadata for all registered tools (for selection prompts)."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "required_params": ", ".join(t.input_schema.get("required_params", [])),
            }
            for t in self._tools.values()
        ]

    def select_tools(self, query: str, max_tools: int = 3) -> list[str]:
        """Keyword-based tool selection. Returns ranked tool names matching the query.

        Scores each tool by: (1) exact word overlap between query and tool
        description/schema, (2) substring prefix matches (e.g. "calculate"
        matches "calculations"), and (3) tool name appearing in query.
        Falls back to returning all tools if nothing matches.
        """
        q_lower = query.lower()
        q_words = set(q_lower.split())
        scores: list[tuple[str, float]] = []

        for tool in self._tools.values():
            desc_text = tool.description.lower()
            schema_text = tool.input_schema.get("description", "").lower()
            corpus = f"{tool.name} {desc_text} {schema_text}"
            corpus_words = set(corpus.split())

            # exact word overlap
            score = float(len(q_words & corpus_words))

            # substring prefix matches (catches "calculate" vs "calculations")
            for qw in q_words:
                if len(qw) < 4:
                    continue
                for cw in corpus_words:
                    if cw.startswith(qw[:4]) and qw != cw:
                        score += 0.5
                        break

            # bonus if the tool name appears as substring in the query
            if tool.name.replace("_", " ") in q_lower or tool.name in q_lower:
                score += 2.0

            if score > 0:
                scores.append((tool.name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        if scores:
            return [name for name, _ in scores[:max_tools]]

        # no keyword match: return all available tools
        return self.tool_names[:max_tools]

    def invoke(self, tool_name: str, tool_input: ToolInput) -> ToolOutput:
        """Execute a tool with rate limiting and safety boundary.

        Returns a structured ToolOutput even on rate limit or missing tool.
        """
        if tool_name not in self._tools:
            return make_output(
                content=f"Tool '{tool_name}' not found.",
                confidence=0.0,
                source=tool_name,
                tool_name=tool_name,
                status="error",
            )

        bucket = self._rate_buckets[tool_name]
        if not bucket.allow():
            logger.warning("Rate limit hit for tool: %s", tool_name)
            return make_output(
                content=f"Tool '{tool_name}' rate-limited. Try again later.",
                confidence=0.0,
                source=tool_name,
                tool_name=tool_name,
                status="rate_limited",
            )

        tool = self._tools[tool_name]
        return timed_execute(tool, tool_input)

    def invoke_best(self, query: str, params: dict[str, str | int | float | bool] | None = None) -> ToolOutput:
        """Select the best tool for a query and execute it. Convenience wrapper."""
        candidates = self.select_tools(query, max_tools=1)
        if not candidates:
            return make_output(
                content="No tools available.",
                confidence=0.0,
                source="registry",
                tool_name="none",
                status="error",
            )
        tool_input = ToolInput(query=query, params=params or {})
        return self.invoke(candidates[0], tool_input)

    def reset_rate_limits(self) -> None:
        """Clear all rate limit state. Useful in tests."""
        for bucket in self._rate_buckets.values():
            bucket.reset()

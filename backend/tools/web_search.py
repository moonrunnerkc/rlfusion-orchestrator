# Author: Bradley R. Kinnard
"""Web search tool: wraps the existing tavily_search() in retrievers.py.

Does NOT duplicate or move tavily_search. Delegates to it and translates
the result into the ToolOutput contract. Safety check on output is the
caller's responsibility (registry or orchestrator layer).
"""
from __future__ import annotations

import logging
import time
from typing import ClassVar

from backend.tools.base import BaseTool, ToolInput, ToolOutput, ToolSchema, make_output

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Tavily-backed web search exposed as a BaseTool.

    Wraps backend.core.retrievers.tavily_search() without rewriting it.
    The existing function handles API key checks, config gating, and HTTP.
    """
    _NAME: ClassVar[str] = "web_search"
    _DESCRIPTION: ClassVar[str] = (
        "Search the web for current information, live data, prices, "
        "news, businesses, and real-time facts using Tavily API."
    )
    _SCHEMA: ClassVar[ToolSchema] = ToolSchema(
        required_params=["query"],
        optional_params=[],
        description="Web search for current events, live data, URLs, businesses, prices.",
    )

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def description(self) -> str:
        return self._DESCRIPTION

    @property
    def input_schema(self) -> ToolSchema:
        return self._SCHEMA

    def execute(self, tool_input: ToolInput) -> ToolOutput:
        """Delegate to tavily_search() and wrap the result."""
        from backend.core.retrievers import tavily_search

        query = tool_input["query"]
        if not query.strip():
            return make_output(
                content="Empty query provided to web search.",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="error",
            )

        start = time.monotonic()
        content, status = tavily_search(query)
        elapsed = (time.monotonic() - start) * 1000

        if status == "disabled":
            return make_output(
                content="Web search is disabled in configuration.",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="error",
                elapsed_ms=elapsed,
            )

        if status == "no_api_key":
            return make_output(
                content="Web search enabled but TAVILY_API_KEY not set.",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="error",
                elapsed_ms=elapsed,
            )

        if status == "error" or not content:
            return make_output(
                content=content or "Web search returned no results.",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="error",
                elapsed_ms=elapsed,
            )

        return make_output(
            content=content,
            confidence=0.85,
            source=self._NAME,
            tool_name=self._NAME,
            status="success",
            elapsed_ms=elapsed,
        )

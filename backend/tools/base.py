# Author: Bradley R. Kinnard
"""BaseTool protocol and shared types for dynamic tool orchestration (Phase 3).

Every tool implements name, description, input_schema, and execute().
The registry discovers and validates tools against this protocol at startup.
"""
from __future__ import annotations

import logging
import time
from typing import Literal, Protocol, TypedDict, runtime_checkable

logger = logging.getLogger(__name__)


class ToolInput(TypedDict):
    """Structured input to a tool invocation."""
    query: str
    params: dict[str, str | int | float | bool]


class ToolOutput(TypedDict):
    """Structured result from a tool execution."""
    content: str
    confidence: float
    source: str
    status: Literal["success", "error", "timeout", "rate_limited"]
    elapsed_ms: float
    tool_name: str


class ToolSchema(TypedDict, total=False):
    """JSON-schema-like descriptor of a tool's expected input parameters."""
    required_params: list[str]
    optional_params: list[str]
    description: str


@runtime_checkable
class BaseTool(Protocol):
    """Protocol for all tools in the dynamic tool orchestration layer.

    Tools wrap external capabilities (APIs, local computations, web search)
    behind a uniform interface. The registry validates conformance at
    registration time, and the orchestrator selects tools based on
    query decomposition output.
    """

    @property
    def name(self) -> str:
        """Short unique identifier for routing and logging."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description for tool selection prompts."""
        ...

    @property
    def input_schema(self) -> ToolSchema:
        """Declares expected parameters so the selector can match queries."""
        ...

    def execute(self, tool_input: ToolInput) -> ToolOutput:
        """Run the tool and return structured output.

        Must handle its own errors internally and return status='error'
        rather than raising. Timeout enforcement is the caller's job.
        """
        ...


def make_output(
    content: str,
    confidence: float,
    source: str,
    tool_name: str,
    status: Literal["success", "error", "timeout", "rate_limited"] = "success",
    elapsed_ms: float = 0.0,
) -> ToolOutput:
    """Factory for building ToolOutput dicts. Clamps confidence to [0, 1]."""
    return ToolOutput(
        content=content,
        confidence=max(0.0, min(1.0, confidence)),
        source=source,
        status=status,
        elapsed_ms=elapsed_ms,
        tool_name=tool_name,
    )


def timed_execute(tool: BaseTool, tool_input: ToolInput) -> ToolOutput:
    """Wrap a tool's execute() with timing and error boundary."""
    start = time.monotonic()
    try:
        result = tool.execute(tool_input)
        result["elapsed_ms"] = (time.monotonic() - start) * 1000
        return result
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        elapsed = (time.monotonic() - start) * 1000
        logger.warning("[%s] Unhandled error in tool execution: %s", tool.name, exc)
        return make_output(
            content=f"Tool execution failed: {exc}",
            confidence=0.0,
            source=tool.name,
            tool_name=tool.name,
            status="error",
            elapsed_ms=elapsed,
        )

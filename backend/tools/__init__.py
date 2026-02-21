# Author: Bradley R. Kinnard
"""Dynamic tool orchestration layer for RLFusion Orchestrator (Phase 3).

Exposes BaseTool protocol, ToolRegistry, and all built-in tool implementations.
Tools wrap external capabilities behind a uniform interface, selected by
query decomposition and dispatched with per-tool rate limiting.
"""

from backend.tools.base import (
    BaseTool,
    ToolInput,
    ToolOutput,
    ToolSchema,
    make_output,
    timed_execute,
)
from backend.tools.registry import ToolRegistry
from backend.tools.api_bridge import ApiBridgeTool
from backend.tools.calculator import CalculatorTool
from backend.tools.code_executor import CodeExecutorTool
from backend.tools.web_search import WebSearchTool

__all__ = [
    "BaseTool",
    "ToolInput",
    "ToolOutput",
    "ToolSchema",
    "ToolRegistry",
    "ApiBridgeTool",
    "CalculatorTool",
    "CodeExecutorTool",
    "WebSearchTool",
    "make_output",
    "timed_execute",
    "build_default_registry",
]


def build_default_registry(
    max_calls_per_tool: int = 10,
    window_secs: float = 60.0,
) -> ToolRegistry:
    """Create a registry pre-loaded with all built-in tools.

    Called during app startup to wire tools into the orchestration layer.
    """
    registry = ToolRegistry(
        max_calls_per_tool=max_calls_per_tool,
        window_secs=window_secs,
    )
    registry.register(WebSearchTool())
    registry.register(CalculatorTool())
    registry.register(CodeExecutorTool())
    registry.register(ApiBridgeTool())
    return registry

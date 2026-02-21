# Author: Bradley R. Kinnard
# test_tools.py - unit tests for Phase 3 dynamic tool orchestration layer
# Tests each tool, the registry, rate limiting, and safety constraints.

import os
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RLFUSION_DEVICE", "cpu")
os.environ.setdefault("RLFUSION_FORCE_CPU", "true")


# ---------------------------------------------------------------------------
# BaseTool protocol conformance
# ---------------------------------------------------------------------------

class TestBaseToolProtocol:
    """Verify all tools satisfy the BaseTool protocol."""

    def test_web_search_is_base_tool(self):
        from backend.tools.base import BaseTool
        from backend.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        assert isinstance(tool, BaseTool)
        assert tool.name == "web_search"

    def test_calculator_is_base_tool(self):
        from backend.tools.base import BaseTool
        from backend.tools.calculator import CalculatorTool
        tool = CalculatorTool()
        assert isinstance(tool, BaseTool)
        assert tool.name == "calculator"

    def test_code_executor_is_base_tool(self):
        from backend.tools.base import BaseTool
        from backend.tools.code_executor import CodeExecutorTool
        tool = CodeExecutorTool()
        assert isinstance(tool, BaseTool)
        assert tool.name == "code_executor"

    def test_api_bridge_is_base_tool(self):
        from backend.tools.base import BaseTool
        from backend.tools.api_bridge import ApiBridgeTool
        tool = ApiBridgeTool()
        assert isinstance(tool, BaseTool)
        assert tool.name == "api_bridge"

    def test_non_tool_fails_protocol(self):
        from backend.tools.base import BaseTool

        class NotATool:
            def foo(self):
                pass
        assert not isinstance(NotATool(), BaseTool)

    def test_all_tools_have_description(self):
        from backend.tools.web_search import WebSearchTool
        from backend.tools.calculator import CalculatorTool
        from backend.tools.code_executor import CodeExecutorTool
        from backend.tools.api_bridge import ApiBridgeTool

        for tool_cls in [WebSearchTool, CalculatorTool, CodeExecutorTool, ApiBridgeTool]:
            tool = tool_cls()
            assert len(tool.description) > 10, f"{tool.name} has too short a description"
            assert isinstance(tool.input_schema, dict)
            assert "required_params" in tool.input_schema

    def test_all_tools_have_unique_names(self):
        from backend.tools.web_search import WebSearchTool
        from backend.tools.calculator import CalculatorTool
        from backend.tools.code_executor import CodeExecutorTool
        from backend.tools.api_bridge import ApiBridgeTool

        names = [cls().name for cls in [WebSearchTool, CalculatorTool, CodeExecutorTool, ApiBridgeTool]]
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"


# ---------------------------------------------------------------------------
# ToolOutput helpers
# ---------------------------------------------------------------------------

class TestToolOutputHelpers:
    """Verify make_output and timed_execute."""

    def test_make_output_clamps_confidence(self):
        from backend.tools.base import make_output
        result = make_output("hi", 1.5, "src", "tool")
        assert result["confidence"] == 1.0
        result2 = make_output("hi", -0.5, "src", "tool")
        assert result2["confidence"] == 0.0

    def test_make_output_fields(self):
        from backend.tools.base import make_output
        result = make_output("content", 0.8, "src", "my_tool", status="success", elapsed_ms=42.5)
        assert result["content"] == "content"
        assert result["confidence"] == 0.8
        assert result["source"] == "src"
        assert result["tool_name"] == "my_tool"
        assert result["status"] == "success"
        assert result["elapsed_ms"] == 42.5

    def test_timed_execute_adds_elapsed(self):
        from backend.tools.base import timed_execute, BaseTool, ToolInput, make_output

        class SlowTool:
            @property
            def name(self):
                return "slow"

            @property
            def description(self):
                return "a slow tool"

            @property
            def input_schema(self):
                return {"required_params": [], "optional_params": []}

            def execute(self, tool_input):
                time.sleep(0.01)
                return make_output("done", 1.0, "slow", "slow")

        tool = SlowTool()
        result = timed_execute(tool, ToolInput(query="test", params={}))
        assert result["elapsed_ms"] > 0
        assert result["status"] == "success"

    def test_timed_execute_catches_exceptions(self):
        from backend.tools.base import timed_execute, ToolInput

        class BrokenTool:
            @property
            def name(self):
                return "broken"

            @property
            def description(self):
                return "always fails"

            @property
            def input_schema(self):
                return {"required_params": [], "optional_params": []}

            def execute(self, tool_input):
                raise RuntimeError("boom")

        tool = BrokenTool()
        result = timed_execute(tool, ToolInput(query="test", params={}))
        assert result["status"] == "error"
        assert "boom" in result["content"]


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------

class TestCalculatorTool:
    """Calculator: safe math eval and unit conversions."""

    def test_basic_arithmetic(self):
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        tool = CalculatorTool()
        result = tool.execute(ToolInput(query="2 + 3 * 4", params={}))
        assert result["status"] == "success"
        assert "14" in result["content"]

    def test_division(self):
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        tool = CalculatorTool()
        result = tool.execute(ToolInput(query="100 / 7", params={}))
        assert result["status"] == "success"
        assert result["confidence"] == 1.0

    def test_division_by_zero(self):
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        tool = CalculatorTool()
        result = tool.execute(ToolInput(query="10 / 0", params={}))
        assert result["status"] == "error"
        assert "zero" in result["content"].lower()

    def test_power(self):
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        tool = CalculatorTool()
        result = tool.execute(ToolInput(query="2 ** 10", params={}))
        assert result["status"] == "success"
        assert "1024" in result["content"]

    def test_sqrt_function(self):
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        tool = CalculatorTool()
        result = tool.execute(ToolInput(query="sqrt(144)", params={}))
        assert result["status"] == "success"
        assert "12" in result["content"]

    def test_pi_constant(self):
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        tool = CalculatorTool()
        result = tool.execute(ToolInput(query="pi * 2", params={}))
        assert result["status"] == "success"
        assert "6.28" in result["content"]

    def test_negative_values(self):
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        tool = CalculatorTool()
        result = tool.execute(ToolInput(query="-5 + 3", params={}))
        assert result["status"] == "success"
        assert "-2" in result["content"]

    def test_unknown_function(self):
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        tool = CalculatorTool()
        result = tool.execute(ToolInput(query="foobar(3)", params={}))
        assert result["status"] == "error"

    def test_empty_expression(self):
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        tool = CalculatorTool()
        result = tool.execute(ToolInput(query="", params={}))
        assert result["status"] == "error"

    def test_invalid_syntax(self):
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        tool = CalculatorTool()
        result = tool.execute(ToolInput(query="2 ++ 3 *** 4", params={}))
        assert result["status"] == "error"

    def test_unit_conversion_length(self):
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        tool = CalculatorTool()
        result = tool.execute(ToolInput(
            query="1000",
            params={"from_unit": "m", "to_unit": "km", "value": "1000"},
        ))
        assert result["status"] == "success"
        assert "1" in result["content"]
        assert "km" in result["content"]

    def test_unit_conversion_temperature(self):
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        tool = CalculatorTool()
        result = tool.execute(ToolInput(
            query="100",
            params={"from_unit": "C", "to_unit": "F", "value": "100"},
        ))
        assert result["status"] == "success"
        assert "212" in result["content"]

    def test_unit_conversion_data(self):
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        tool = CalculatorTool()
        result = tool.execute(ToolInput(
            query="1",
            params={"from_unit": "gb", "to_unit": "mb", "value": "1"},
        ))
        assert result["status"] == "success"
        assert "1024" in result["content"]

    def test_unit_conversion_invalid(self):
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        tool = CalculatorTool()
        result = tool.execute(ToolInput(
            query="100",
            params={"from_unit": "kg", "to_unit": "km", "value": "100"},
        ))
        assert result["status"] == "error"


class TestSafeEval:
    """Verify the AST-based safe evaluator rejects dangerous inputs."""

    def test_rejects_attribute_access(self):
        from backend.tools.calculator import _safe_eval
        with pytest.raises(ValueError):
            _safe_eval("__import__('os')")

    def test_rejects_unknown_names(self):
        from backend.tools.calculator import _safe_eval
        with pytest.raises(ValueError, match="Unknown name"):
            _safe_eval("os")

    def test_accepts_nested_calls(self):
        from backend.tools.calculator import _safe_eval
        result = _safe_eval("sqrt(abs(-16))")
        assert abs(result - 4.0) < 1e-6


class TestUnitConversion:
    """Standalone convert_units function tests."""

    def test_meters_to_feet(self):
        from backend.tools.calculator import convert_units
        result, category = convert_units(1.0, "m", "ft")
        assert abs(result - 3.28084) < 0.001
        assert category == "length"

    def test_kg_to_lb(self):
        from backend.tools.calculator import convert_units
        result, category = convert_units(1.0, "kg", "lb")
        assert abs(result - 2.20462) < 0.01
        assert category == "mass"

    def test_celsius_to_kelvin(self):
        from backend.tools.calculator import convert_units
        result, category = convert_units(0, "C", "K")
        assert abs(result - 273.15) < 0.01
        assert category == "temperature"

    def test_invalid_cross_category(self):
        from backend.tools.calculator import convert_units
        with pytest.raises(ValueError, match="same category"):
            convert_units(1.0, "m", "kg")

    def test_seconds_to_hours(self):
        from backend.tools.calculator import convert_units
        result, category = convert_units(3600, "s", "hr")
        assert abs(result - 1.0) < 1e-6
        assert category == "time"


# ---------------------------------------------------------------------------
# CodeExecutorTool
# ---------------------------------------------------------------------------

class TestCodeExecutorTool:
    """Code executor: sandboxed subprocess execution."""

    def test_simple_print(self):
        from backend.tools.code_executor import CodeExecutorTool
        from backend.tools.base import ToolInput
        tool = CodeExecutorTool(timeout_secs=5)
        result = tool.execute(ToolInput(query="print(2 + 2)", params={}))
        assert result["status"] == "success"
        assert "4" in result["content"]

    def test_multiline_code(self):
        from backend.tools.code_executor import CodeExecutorTool
        from backend.tools.base import ToolInput
        tool = CodeExecutorTool(timeout_secs=5)
        code = "x = [i**2 for i in range(5)]\nprint(x)"
        result = tool.execute(ToolInput(query=code, params={}))
        assert result["status"] == "success"
        assert "[0, 1, 4, 9, 16]" in result["content"]

    def test_empty_code(self):
        from backend.tools.code_executor import CodeExecutorTool
        from backend.tools.base import ToolInput
        tool = CodeExecutorTool()
        result = tool.execute(ToolInput(query="", params={}))
        assert result["status"] == "error"

    def test_banned_import_os(self):
        from backend.tools.code_executor import CodeExecutorTool
        from backend.tools.base import ToolInput
        tool = CodeExecutorTool(timeout_secs=5)
        result = tool.execute(ToolInput(query="import os\nprint(os.getcwd())", params={}))
        assert result["status"] == "error"

    def test_banned_import_subprocess(self):
        from backend.tools.code_executor import CodeExecutorTool
        from backend.tools.base import ToolInput
        tool = CodeExecutorTool(timeout_secs=5)
        result = tool.execute(ToolInput(query="import subprocess", params={}))
        assert result["status"] == "error"

    def test_banned_eval(self):
        from backend.tools.code_executor import CodeExecutorTool
        from backend.tools.base import ToolInput
        tool = CodeExecutorTool(timeout_secs=5)
        result = tool.execute(ToolInput(query="eval('1+1')", params={}))
        assert result["status"] == "error"

    def test_banned_open(self):
        from backend.tools.code_executor import CodeExecutorTool
        from backend.tools.base import ToolInput
        tool = CodeExecutorTool(timeout_secs=5)
        result = tool.execute(ToolInput(query="open('/etc/passwd')", params={}))
        assert result["status"] == "error"

    def test_banned_dunder_import(self):
        from backend.tools.code_executor import CodeExecutorTool
        from backend.tools.base import ToolInput
        tool = CodeExecutorTool(timeout_secs=5)
        result = tool.execute(ToolInput(query="__import__('os')", params={}))
        assert result["status"] == "error"

    def test_timeout_enforcement(self):
        from backend.tools.code_executor import CodeExecutorTool
        from backend.tools.base import ToolInput
        tool = CodeExecutorTool(timeout_secs=2)
        code = "import time\ntime.sleep(10)"
        # time is not in our banned list at the static check level,
        # but the subprocess will be killed by timeout
        result = tool.execute(ToolInput(query=code, params={}))
        assert result["status"] in ("timeout", "error")

    def test_math_computation(self):
        from backend.tools.code_executor import CodeExecutorTool
        from backend.tools.base import ToolInput
        tool = CodeExecutorTool(timeout_secs=5)
        code = "import math\nprint(f'{math.factorial(10)}')"
        result = tool.execute(ToolInput(query=code, params={}))
        assert result["status"] == "success"
        assert "3628800" in result["content"]

    def test_elapsed_time_recorded(self):
        from backend.tools.code_executor import CodeExecutorTool
        from backend.tools.base import ToolInput
        tool = CodeExecutorTool(timeout_secs=5)
        result = tool.execute(ToolInput(query="print('fast')", params={}))
        assert result["elapsed_ms"] > 0


class TestCodeValidation:
    """Standalone _validate_code function."""

    def test_clean_code_passes(self):
        from backend.tools.code_executor import _validate_code
        ok, reason = _validate_code("x = 1 + 2\nprint(x)")
        assert ok is True
        assert reason == ""

    def test_os_import_fails(self):
        from backend.tools.code_executor import _validate_code
        ok, reason = _validate_code("import os")
        assert ok is False
        assert "os" in reason

    def test_from_import_fails(self):
        from backend.tools.code_executor import _validate_code
        ok, reason = _validate_code("from pathlib import Path")
        assert ok is False
        assert "pathlib" in reason

    def test_exec_rejected(self):
        from backend.tools.code_executor import _validate_code
        ok, reason = _validate_code("exec('print(1)')")
        assert ok is False


# ---------------------------------------------------------------------------
# ApiBridgeTool
# ---------------------------------------------------------------------------

class TestApiBridgeTool:
    """API bridge: URL validation and error handling."""

    def test_no_url_returns_error(self):
        from backend.tools.api_bridge import ApiBridgeTool
        from backend.tools.base import ToolInput
        tool = ApiBridgeTool()
        result = tool.execute(ToolInput(query="fetch data", params={}))
        assert result["status"] == "error"
        assert "URL" in result["content"]

    def test_invalid_scheme_rejected(self):
        from backend.tools.api_bridge import ApiBridgeTool
        from backend.tools.base import ToolInput
        tool = ApiBridgeTool()
        result = tool.execute(ToolInput(query="fetch", params={"url": "ftp://example.com"}))
        assert result["status"] == "error"
        assert "scheme" in result["content"].lower()

    def test_private_url_blocked_localhost(self):
        from backend.tools.api_bridge import ApiBridgeTool
        from backend.tools.base import ToolInput
        tool = ApiBridgeTool()
        result = tool.execute(ToolInput(query="fetch", params={"url": "http://localhost:8000/data"}))
        assert result["status"] == "error"
        assert "private" in result["content"].lower()

    def test_private_url_blocked_10_net(self):
        from backend.tools.api_bridge import ApiBridgeTool
        from backend.tools.base import ToolInput
        tool = ApiBridgeTool()
        result = tool.execute(ToolInput(query="fetch", params={"url": "http://10.0.0.1/api"}))
        assert result["status"] == "error"
        assert "private" in result["content"].lower()

    def test_private_url_blocked_192_168(self):
        from backend.tools.api_bridge import ApiBridgeTool
        from backend.tools.base import ToolInput
        tool = ApiBridgeTool()
        result = tool.execute(ToolInput(query="fetch", params={"url": "http://192.168.1.1/api"}))
        assert result["status"] == "error"

    def test_unsupported_method_rejected(self):
        from backend.tools.api_bridge import ApiBridgeTool
        from backend.tools.base import ToolInput
        tool = ApiBridgeTool()
        result = tool.execute(ToolInput(
            query="delete", params={"url": "https://api.example.com/resource", "method": "DELETE"},
        ))
        assert result["status"] == "error"
        assert "GET and POST" in result["content"]

    def test_invalid_json_body(self):
        from backend.tools.api_bridge import ApiBridgeTool
        from backend.tools.base import ToolInput
        tool = ApiBridgeTool()
        result = tool.execute(ToolInput(
            query="post",
            params={"url": "https://api.example.com/data", "method": "POST", "body": "{invalid json"},
        ))
        assert result["status"] == "error"
        assert "JSON" in result["content"]


class TestPrivateUrlDetection:
    """Standalone _is_private_url checks."""

    def test_public_url_allowed(self):
        from backend.tools.api_bridge import _is_private_url
        assert _is_private_url("https://api.github.com/repos") is False

    def test_localhost_blocked(self):
        from backend.tools.api_bridge import _is_private_url
        assert _is_private_url("http://localhost:3000") is True

    def test_127_blocked(self):
        from backend.tools.api_bridge import _is_private_url
        assert _is_private_url("http://127.0.0.1/api") is True

    def test_172_private_blocked(self):
        from backend.tools.api_bridge import _is_private_url
        assert _is_private_url("http://172.16.0.1/api") is True

    def test_172_public_allowed(self):
        from backend.tools.api_bridge import _is_private_url
        assert _is_private_url("http://172.15.0.1/api") is False

    def test_0000_blocked(self):
        from backend.tools.api_bridge import _is_private_url
        assert _is_private_url("http://0.0.0.0:8080") is True


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------

class TestWebSearchTool:
    """Web search tool wraps tavily_search without duplicating logic."""

    def test_empty_query_returns_error(self):
        from backend.tools.web_search import WebSearchTool
        from backend.tools.base import ToolInput
        tool = WebSearchTool()
        result = tool.execute(ToolInput(query="", params={}))
        assert result["status"] == "error"
        assert "Empty" in result["content"]

    def test_disabled_web_returns_error(self):
        from backend.tools.web_search import WebSearchTool
        from backend.tools.base import ToolInput
        # web is disabled by default in config.yaml (web.enabled: false)
        tool = WebSearchTool()
        result = tool.execute(ToolInput(query="latest news", params={}))
        assert result["status"] == "error"
        # should indicate disabled or no_api_key
        assert result["confidence"] == 0.0

    def test_tool_name_and_source(self):
        from backend.tools.web_search import WebSearchTool
        from backend.tools.base import ToolInput
        tool = WebSearchTool()
        result = tool.execute(ToolInput(query="test", params={}))
        assert result["tool_name"] == "web_search"
        assert result["source"] == "web_search"


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    """Registry: registration, lookup, selection, rate limiting."""

    def test_register_and_get(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        reg = ToolRegistry()
        calc = CalculatorTool()
        reg.register(calc)
        assert reg.tool_count == 1
        assert reg.get("calculator") is calc

    def test_register_duplicate_raises(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        with pytest.raises(ValueError, match="Duplicate"):
            reg.register(CalculatorTool())

    def test_register_non_tool_raises(self):
        from backend.tools.registry import ToolRegistry

        class NotATool:
            pass
        reg = ToolRegistry()
        with pytest.raises(TypeError, match="BaseTool"):
            reg.register(NotATool())  # type: ignore[arg-type]

    def test_unregister(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        assert reg.tool_count == 1
        reg.unregister("calculator")
        assert reg.tool_count == 0

    def test_unregister_missing_raises(self):
        from backend.tools.registry import ToolRegistry
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.unregister("nonexistent")

    def test_get_missing_raises(self):
        from backend.tools.registry import ToolRegistry
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("nonexistent")

    def test_list_tools(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        from backend.tools.code_executor import CodeExecutorTool
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        reg.register(CodeExecutorTool())
        listing = reg.list_tools()
        assert len(listing) == 2
        names = {t["name"] for t in listing}
        assert names == {"calculator", "code_executor"}
        for t in listing:
            assert "description" in t

    def test_tool_names_property(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        assert "calculator" in reg.tool_names

    def test_select_tools_math_query(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        from backend.tools.web_search import WebSearchTool
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        reg.register(WebSearchTool())
        selected = reg.select_tools("calculate the square root of 144")
        assert "calculator" in selected

    def test_select_tools_web_query(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        from backend.tools.web_search import WebSearchTool
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        reg.register(WebSearchTool())
        selected = reg.select_tools("search the web for current news")
        assert "web_search" in selected

    def test_select_tools_no_match_returns_all(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        selected = reg.select_tools("zyxwvutsrqponmlkjihgfedcba")
        assert len(selected) >= 1  # falls back to returning available tools

    def test_invoke_executes_tool(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        result = reg.invoke("calculator", ToolInput(query="2 + 2", params={}))
        assert result["status"] == "success"
        assert "4" in result["content"]

    def test_invoke_missing_tool(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.base import ToolInput
        reg = ToolRegistry()
        result = reg.invoke("nonexistent", ToolInput(query="test", params={}))
        assert result["status"] == "error"
        assert "not found" in result["content"].lower()

    def test_invoke_best(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        result = reg.invoke_best("3 * 7")
        assert result["status"] == "success"


class TestRegistryRateLimiting:
    """Per-tool rate limit enforcement."""

    def test_rate_limit_blocks_after_max(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        reg = ToolRegistry(max_calls_per_tool=3, window_secs=60.0)
        reg.register(CalculatorTool())

        for _ in range(3):
            result = reg.invoke("calculator", ToolInput(query="1+1", params={}))
            assert result["status"] == "success"

        # 4th call should be rate-limited
        result = reg.invoke("calculator", ToolInput(query="1+1", params={}))
        assert result["status"] == "rate_limited"

    def test_rate_limit_reset(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        from backend.tools.base import ToolInput
        reg = ToolRegistry(max_calls_per_tool=2, window_secs=60.0)
        reg.register(CalculatorTool())

        reg.invoke("calculator", ToolInput(query="1+1", params={}))
        reg.invoke("calculator", ToolInput(query="1+1", params={}))
        result = reg.invoke("calculator", ToolInput(query="1+1", params={}))
        assert result["status"] == "rate_limited"

        reg.reset_rate_limits()
        result = reg.invoke("calculator", ToolInput(query="1+1", params={}))
        assert result["status"] == "success"

    def test_rate_limit_per_tool_isolation(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        from backend.tools.code_executor import CodeExecutorTool
        from backend.tools.base import ToolInput
        reg = ToolRegistry(max_calls_per_tool=2, window_secs=60.0)
        reg.register(CalculatorTool())
        reg.register(CodeExecutorTool())

        # exhaust calculator
        reg.invoke("calculator", ToolInput(query="1+1", params={}))
        reg.invoke("calculator", ToolInput(query="1+1", params={}))
        calc_result = reg.invoke("calculator", ToolInput(query="1+1", params={}))
        assert calc_result["status"] == "rate_limited"

        # code_executor should still work
        code_result = reg.invoke("code_executor", ToolInput(query="print('hi')", params={}))
        assert code_result["status"] == "success"


# ---------------------------------------------------------------------------
# build_default_registry
# ---------------------------------------------------------------------------

class TestBuildDefaultRegistry:
    """Verify the convenience factory wires up all built-in tools."""

    def test_all_tools_registered(self):
        from backend.tools import build_default_registry
        reg = build_default_registry()
        assert reg.tool_count == 4
        expected = {"web_search", "calculator", "code_executor", "api_bridge"}
        assert set(reg.tool_names) == expected

    def test_custom_rate_limits(self):
        from backend.tools import build_default_registry
        reg = build_default_registry(max_calls_per_tool=5, window_secs=30.0)
        assert reg.tool_count == 4


# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------

class TestToolsPackage:
    """Verify the backend.tools package exports all expected symbols."""

    def test_all_exports(self):
        import backend.tools as tools
        assert hasattr(tools, "BaseTool")
        assert hasattr(tools, "ToolInput")
        assert hasattr(tools, "ToolOutput")
        assert hasattr(tools, "ToolSchema")
        assert hasattr(tools, "ToolRegistry")
        assert hasattr(tools, "WebSearchTool")
        assert hasattr(tools, "CalculatorTool")
        assert hasattr(tools, "CodeExecutorTool")
        assert hasattr(tools, "ApiBridgeTool")
        assert hasattr(tools, "build_default_registry")
        assert hasattr(tools, "make_output")
        assert hasattr(tools, "timed_execute")

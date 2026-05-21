# Author: Bradley R. Kinnard
# test_tools.py - unit tests for the calculator tool and BaseTool protocol.
# The earlier dynamic tool layer (code_executor, api_bridge, registry) was
# removed in the v2 security hardening pass; bring those back when a real
# callsite needs them.

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
    """Verify the calculator satisfies the BaseTool protocol."""

    def test_calculator_is_base_tool(self):
        from backend.tools.base import BaseTool
        from backend.tools.calculator import CalculatorTool
        tool = CalculatorTool()
        assert isinstance(tool, BaseTool)
        assert tool.name == "calculator"

    def test_non_tool_fails_protocol(self):
        from backend.tools.base import BaseTool

        class NotATool:
            def foo(self):
                pass
        assert not isinstance(NotATool(), BaseTool)

    def test_calculator_has_description(self):
        from backend.tools.calculator import CalculatorTool
        tool = CalculatorTool()
        assert len(tool.description) > 10
        assert isinstance(tool.input_schema, dict)
        assert "required_params" in tool.input_schema


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
        from backend.tools.base import timed_execute, ToolInput, make_output

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
# Package surface
# ---------------------------------------------------------------------------

class TestToolsPackage:
    """Verify the backend.tools package exports the post-cleanup surface."""

    def test_all_exports(self):
        import backend.tools as tools
        assert hasattr(tools, "BaseTool")
        assert hasattr(tools, "ToolInput")
        assert hasattr(tools, "ToolOutput")
        assert hasattr(tools, "ToolSchema")
        assert hasattr(tools, "CalculatorTool")
        assert hasattr(tools, "make_output")
        assert hasattr(tools, "timed_execute")

    def test_removed_symbols_are_gone(self):
        import backend.tools as tools
        for name in ("ToolRegistry", "CodeExecutorTool", "ApiBridgeTool", "build_default_registry"):
            assert not hasattr(tools, name), (
                f"backend.tools should no longer export {name}; delete or update the consumer."
            )

# Author: Bradley R. Kinnard
"""Calculator tool: math evaluation and unit conversion.

Evaluates arithmetic expressions safely using Python's ast module (no eval).
Supports basic unit conversions for common engineering/science quantities.
"""
from __future__ import annotations

import ast
import logging
import math
import operator
import time
from typing import ClassVar

from backend.tools.base import BaseTool, ToolInput, ToolOutput, ToolSchema, make_output

logger = logging.getLogger(__name__)

# safe binary operators for ast-based evaluation
_OPS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# safe math functions available in expressions
_MATH_FUNCS: dict[str, object] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
    "pi": math.pi,
    "e": math.e,
}

# unit conversion factors (all relative to a base unit per category)
_CONVERSIONS: dict[str, dict[str, float]] = {
    "length": {
        "m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001,
        "mi": 1609.344, "ft": 0.3048, "in": 0.0254, "yd": 0.9144,
    },
    "mass": {
        "kg": 1.0, "g": 0.001, "mg": 1e-6, "lb": 0.453592,
        "oz": 0.0283495, "ton": 907.185, "tonne": 1000.0,
    },
    "temperature": {},  # handled separately (non-linear)
    "data": {
        "b": 1.0, "kb": 1024.0, "mb": 1048576.0, "gb": 1073741824.0,
        "tb": 1099511627776.0,
    },
    "time": {
        "s": 1.0, "ms": 0.001, "us": 1e-6, "ns": 1e-9,
        "min": 60.0, "hr": 3600.0, "day": 86400.0,
    },
}


def _safe_eval(expr: str) -> float:
    """Evaluate a math expression via AST walking. No exec/eval."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc
    return _eval_node(tree.body)


def _eval_node(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.Name):
        if node.id in _MATH_FUNCS:
            val = _MATH_FUNCS[node.id]
            if isinstance(val, float):
                return val
            raise ValueError(f"'{node.id}' is a function, not a constant. Use {node.id}(...).")
        raise ValueError(f"Unknown name: '{node.id}'")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        op_fn = _OPS[type(node.op)]
        return op_fn(_eval_node(node.operand))  # type: ignore[operator]

    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        op_fn = _OPS[type(node.op)]
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv)) and right == 0:
            raise ValueError("Division by zero.")
        result = op_fn(left, right)  # type: ignore[operator]
        if isinstance(result, complex):
            raise ValueError("Complex number result not supported.")
        return float(result)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed (e.g. sqrt(4)).")
        func_name = node.func.id
        if func_name not in _MATH_FUNCS:
            raise ValueError(f"Unknown function: '{func_name}'")
        func = _MATH_FUNCS[func_name]
        if not callable(func):
            raise ValueError(f"'{func_name}' is a constant, not a function.")
        args = [_eval_node(a) for a in node.args]
        return float(func(*args))  # type: ignore[operator]

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between C, F, and K."""
    f, t = from_unit.lower(), to_unit.lower()
    # normalize to Celsius first
    if f == "c":
        celsius = value
    elif f == "f":
        celsius = (value - 32) * 5 / 9
    elif f == "k":
        celsius = value - 273.15
    else:
        raise ValueError(f"Unknown temperature unit: '{from_unit}'")

    # then to target
    if t == "c":
        return celsius
    if t == "f":
        return celsius * 9 / 5 + 32
    if t == "k":
        return celsius + 273.15
    raise ValueError(f"Unknown temperature unit: '{to_unit}'")


def convert_units(value: float, from_unit: str, to_unit: str) -> tuple[float, str]:
    """Convert between units within the same category. Returns (result, category)."""
    fu = from_unit.lower().strip()
    tu = to_unit.lower().strip()

    # temperature is special
    if fu in ("c", "f", "k") and tu in ("c", "f", "k"):
        return _convert_temperature(value, fu, tu), "temperature"

    for category, table in _CONVERSIONS.items():
        if category == "temperature":
            continue
        if fu in table and tu in table:
            base_value = value * table[fu]
            return base_value / table[tu], category

    raise ValueError(
        f"Cannot convert '{from_unit}' to '{to_unit}'. "
        f"Units must belong to the same category."
    )


class CalculatorTool:
    """Safe math evaluation and unit conversion.

    Expression evaluation uses AST parsing (no eval/exec).
    Unit conversion covers length, mass, temperature, data sizes, and time.
    """
    _NAME: ClassVar[str] = "calculator"
    _DESCRIPTION: ClassVar[str] = (
        "Evaluate math expressions and convert between units. "
        "Supports arithmetic, trig, logarithms, and common unit conversions "
        "(length, mass, temperature, data, time)."
    )
    _SCHEMA: ClassVar[ToolSchema] = ToolSchema(
        required_params=["query"],
        optional_params=["from_unit", "to_unit", "value"],
        description="Math evaluation, arithmetic, unit conversion, calculations.",
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
        """Evaluate expression or convert units based on params."""
        start = time.monotonic()
        params = tool_input.get("params", {})
        query = tool_input["query"].strip()

        from_unit = str(params.get("from_unit", ""))
        to_unit = str(params.get("to_unit", ""))
        raw_value = params.get("value", "")

        # unit conversion mode
        if from_unit and to_unit:
            try:
                value = float(raw_value) if raw_value else _safe_eval(query)
                result, category = convert_units(value, from_unit, to_unit)
                elapsed = (time.monotonic() - start) * 1000
                content = f"{value} {from_unit} = {result:.6g} {to_unit} ({category})"
                return make_output(
                    content=content,
                    confidence=1.0,
                    source=self._NAME,
                    tool_name=self._NAME,
                    status="success",
                    elapsed_ms=elapsed,
                )
            except (ValueError, ZeroDivisionError, OverflowError) as exc:
                elapsed = (time.monotonic() - start) * 1000
                return make_output(
                    content=f"Conversion failed: {exc}",
                    confidence=0.0,
                    source=self._NAME,
                    tool_name=self._NAME,
                    status="error",
                    elapsed_ms=elapsed,
                )

        # expression evaluation mode
        if not query:
            return make_output(
                content="Empty expression.",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="error",
            )

        try:
            result = _safe_eval(query)
            elapsed = (time.monotonic() - start) * 1000
            return make_output(
                content=f"{query} = {result:.10g}",
                confidence=1.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="success",
                elapsed_ms=elapsed,
            )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
            elapsed = (time.monotonic() - start) * 1000
            return make_output(
                content=f"Evaluation failed: {exc}",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="error",
                elapsed_ms=elapsed,
            )

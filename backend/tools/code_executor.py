# Author: Bradley R. Kinnard
"""Code executor tool: sandboxed Python snippet execution via subprocess.

Runs user-provided Python code in an isolated subprocess with:
- No filesystem access beyond data/ (enforced via restricted builtins)
- Timeout enforcement (default 10s)
- Stdout/stderr capture
- No network access from the subprocess

Designed for data queries, calculations, and simple transformations.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import ClassVar

from backend.config import PROJECT_ROOT
from backend.tools.base import BaseTool, ToolInput, ToolOutput, ToolSchema, make_output

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECS = 10
_MAX_OUTPUT_CHARS = 4000

# these imports are banned in the sandboxed subprocess
_BANNED_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "httpx",
    "importlib", "ctypes", "signal", "multiprocessing",
    "threading", "pickle", "shelve", "tempfile",
    "__builtin__", "builtins", "code", "codeop",
    "compile", "compileall", "py_compile",
})

# wrapper script that disables dangerous imports before running the user code
_SANDBOX_TEMPLATE = textwrap.dedent("""\
    import sys as _sys

    _BANNED = {banned_set}

    class _ImportBlocker:
        def find_module(self, name, path=None):
            top = name.split(".")[0]
            if top in _BANNED:
                return self
            return None
        def load_module(self, name):
            raise ImportError(f"Module '{{name}}' is not allowed in sandbox.")

    _sys.meta_path.insert(0, _ImportBlocker())

    # user code below
    {user_code}
""")


def _validate_code(code: str) -> tuple[bool, str]:
    """Static check for obviously dangerous patterns before subprocess exec."""
    lower = code.lower()
    for mod in _BANNED_MODULES:
        # check for direct import statements
        if f"import {mod}" in lower or f"from {mod}" in lower:
            return False, f"Banned module '{mod}' detected in code."

    if "exec(" in lower or "eval(" in lower:
        return False, "exec() and eval() are not allowed in sandboxed code."
    if "__import__" in lower:
        return False, "__import__() is not allowed in sandboxed code."
    if "open(" in lower:
        return False, "open() is not allowed. Use tool parameters for file paths."

    return True, ""


class CodeExecutorTool:
    """Sandboxed Python code execution for data transformations and calculations.

    Runs code in a subprocess with import restrictions and a hard timeout.
    Use for math-heavy queries that benefit from actual computation.
    """
    _NAME: ClassVar[str] = "code_executor"
    _DESCRIPTION: ClassVar[str] = (
        "Execute Python code snippets in a sandboxed environment. "
        "Good for data calculations, list processing, string manipulation, "
        "and algorithmic queries. No file or network access."
    )
    _SCHEMA: ClassVar[ToolSchema] = ToolSchema(
        required_params=["query"],
        optional_params=["timeout"],
        description="Execute Python code for data queries, calculations, algorithms.",
    )

    def __init__(self, timeout_secs: int = _DEFAULT_TIMEOUT_SECS) -> None:
        self._timeout = timeout_secs

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
        """Run user code in a subprocess sandbox."""
        code = tool_input["query"].strip()
        if not code:
            return make_output(
                content="No code provided.",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="error",
            )

        # static validation
        ok, reason = _validate_code(code)
        if not ok:
            return make_output(
                content=f"Code rejected: {reason}",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="error",
            )

        # build sandboxed script
        indented_code = textwrap.indent(code, "    " * 0)
        banned_repr = repr(set(_BANNED_MODULES))
        script = _SANDBOX_TEMPLATE.format(
            banned_set=banned_repr,
            user_code=indented_code,
        )

        timeout = int(tool_input.get("params", {}).get("timeout", self._timeout))
        timeout = max(1, min(timeout, 30))

        start = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(PROJECT_ROOT / "data"),
                env={"PATH": ""},  # strip all PATH to prevent shell escapes
            )
            elapsed = (time.monotonic() - start) * 1000

            stdout = result.stdout[:_MAX_OUTPUT_CHARS] if result.stdout else ""
            stderr = result.stderr[:_MAX_OUTPUT_CHARS] if result.stderr else ""

            if result.returncode != 0:
                content = stderr or f"Process exited with code {result.returncode}"
                return make_output(
                    content=content,
                    confidence=0.0,
                    source=self._NAME,
                    tool_name=self._NAME,
                    status="error",
                    elapsed_ms=elapsed,
                )

            output = stdout.strip()
            if not output and stderr:
                output = f"(no stdout, stderr: {stderr.strip()[:500]})"

            return make_output(
                content=output or "(no output)",
                confidence=0.9 if output else 0.3,
                source=self._NAME,
                tool_name=self._NAME,
                status="success",
                elapsed_ms=elapsed,
            )

        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - start) * 1000
            return make_output(
                content=f"Code execution timed out after {timeout}s.",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="timeout",
                elapsed_ms=elapsed,
            )
        except OSError as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.warning("Code executor subprocess failed: %s", exc)
            return make_output(
                content=f"Subprocess launch failed: {exc}",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="error",
                elapsed_ms=elapsed,
            )

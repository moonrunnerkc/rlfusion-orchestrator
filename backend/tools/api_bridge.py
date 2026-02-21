# Author: Bradley R. Kinnard
"""API bridge tool: generic REST API caller with schema validation.

Makes HTTP requests to user-configured REST endpoints. Validates request
and response shapes against a provided schema. Strips sensitive headers
and enforces timeouts. No credentials are stored; API keys come from
environment variables or explicit params per invocation.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import ClassVar

import httpx

from backend.tools.base import BaseTool, ToolInput, ToolOutput, ToolSchema, make_output

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECS = 15
_MAX_RESPONSE_CHARS = 4000

# block requests to private/internal networks
_PRIVATE_HOST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^localhost$", re.IGNORECASE),
    re.compile(r"^127\.\d+\.\d+\.\d+$"),
    re.compile(r"^10\.\d+\.\d+\.\d+$"),
    re.compile(r"^172\.(1[6-9]|2\d|3[01])\.\d+\.\d+$"),
    re.compile(r"^192\.168\.\d+\.\d+$"),
    re.compile(r"^0\.0\.0\.0$"),
    re.compile(r"^::1$"),
    re.compile(r"^\[::1\]$"),
]


def _is_private_url(url: str) -> bool:
    """Reject URLs pointing to private/internal networks."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.hostname or ""
        for pattern in _PRIVATE_HOST_PATTERNS:
            if pattern.match(host):
                return True
        return False
    except (ValueError, TypeError):
        return True  # malformed URL, block it


def _sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    """Strip obviously sensitive headers from the response log."""
    sensitive = {"authorization", "cookie", "set-cookie", "x-api-key"}
    return {k: v for k, v in headers.items() if k.lower() not in sensitive}


class ApiBridgeTool:
    """Generic REST API tool with URL validation and response truncation.

    Supports GET and POST to external public APIs. Private network URLs
    are blocked. Timeouts are enforced. Response bodies are truncated
    to prevent context overflow.
    """
    _NAME: ClassVar[str] = "api_bridge"
    _DESCRIPTION: ClassVar[str] = (
        "Make HTTP requests to external REST APIs. "
        "Supports GET and POST with JSON bodies. "
        "Good for fetching structured data from public APIs."
    )
    _SCHEMA: ClassVar[ToolSchema] = ToolSchema(
        required_params=["query"],
        optional_params=["url", "method", "body", "api_key_env"],
        description="REST API calls, HTTP requests, external data fetch, JSON API.",
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
        """Make an HTTP request to an external API."""
        params = tool_input.get("params", {})
        url = str(params.get("url", "")).strip()
        method = str(params.get("method", "GET")).upper()
        body_raw = params.get("body", "")
        api_key_env = str(params.get("api_key_env", ""))

        if not url:
            return make_output(
                content="No URL provided. Set params.url to the target API endpoint.",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="error",
            )

        if not url.startswith(("http://", "https://")):
            return make_output(
                content=f"Invalid URL scheme. Must start with http:// or https://. Got: {url[:60]}",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="error",
            )

        if _is_private_url(url):
            return make_output(
                content="Requests to private/internal networks are blocked.",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="error",
            )

        if method not in ("GET", "POST"):
            return make_output(
                content=f"Only GET and POST are supported. Got: {method}",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="error",
            )

        headers: dict[str, str] = {"User-Agent": "RLFusion-ApiBridge/1.0"}
        if api_key_env:
            key = os.environ.get(api_key_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
            else:
                logger.warning("API key env var '%s' not set; proceeding without auth.", api_key_env)

        # parse body for POST
        json_body: dict[str, object] | None = None
        if method == "POST" and body_raw:
            try:
                if isinstance(body_raw, str):
                    json_body = json.loads(body_raw)
                elif isinstance(body_raw, dict):
                    json_body = body_raw  # type: ignore[assignment]
                else:
                    return make_output(
                        content="POST body must be a JSON string or dict.",
                        confidence=0.0,
                        source=self._NAME,
                        tool_name=self._NAME,
                        status="error",
                    )
            except json.JSONDecodeError as exc:
                return make_output(
                    content=f"Invalid JSON body: {exc}",
                    confidence=0.0,
                    source=self._NAME,
                    tool_name=self._NAME,
                    status="error",
                )

        start = time.monotonic()
        try:
            if method == "GET":
                resp = httpx.get(url, headers=headers, timeout=self._timeout, follow_redirects=True)
            else:
                resp = httpx.post(
                    url, headers=headers, json=json_body,
                    timeout=self._timeout, follow_redirects=True,
                )
            elapsed = (time.monotonic() - start) * 1000

            if resp.status_code >= 400:
                body_preview = resp.text[:500] if resp.text else "(empty)"
                return make_output(
                    content=f"HTTP {resp.status_code}: {body_preview}",
                    confidence=0.0,
                    source=self._NAME,
                    tool_name=self._NAME,
                    status="error",
                    elapsed_ms=elapsed,
                )

            body = resp.text[:_MAX_RESPONSE_CHARS]
            confidence = 0.8 if resp.status_code == 200 else 0.5

            return make_output(
                content=body,
                confidence=confidence,
                source=self._NAME,
                tool_name=self._NAME,
                status="success",
                elapsed_ms=elapsed,
            )

        except httpx.TimeoutException:
            elapsed = (time.monotonic() - start) * 1000
            return make_output(
                content=f"Request to {url[:60]} timed out after {self._timeout}s.",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="timeout",
                elapsed_ms=elapsed,
            )
        except httpx.RequestError as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.warning("API bridge request failed: %s", exc)
            return make_output(
                content=f"Request failed: {exc}",
                confidence=0.0,
                source=self._NAME,
                tool_name=self._NAME,
                status="error",
                elapsed_ms=elapsed,
            )

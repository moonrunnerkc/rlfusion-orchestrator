# Author: Bradley R. Kinnard
"""Admin auth dependency for protected endpoints.

Boot-time check enforces a non-empty key of at least 32 characters so the
process refuses to start with a weak or missing RLFUSION_ADMIN_KEY. Per-
request check uses hmac.compare_digest to avoid timing side channels and
raises HTTPException(401) on failure (never returns a 200 with an "error"
body, since that breaks monitoring).
"""
from __future__ import annotations

import hmac
import logging
import os

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

_MIN_ADMIN_KEY_LEN = 32


def get_admin_key() -> str:
    """Return the configured admin key. Empty string if unset."""
    return os.environ.get("RLFUSION_ADMIN_KEY", "")


def enforce_admin_key_at_boot() -> None:
    """Refuse to start if RLFUSION_ADMIN_KEY is missing or too short.

    Tests may set RLFUSION_ALLOW_WEAK_ADMIN_KEY=1 to bypass the length floor
    so they can exercise the auth path with short fixture keys.
    """
    key = get_admin_key()
    allow_weak = os.environ.get("RLFUSION_ALLOW_WEAK_ADMIN_KEY", "").lower() in (
        "1", "true", "yes",
    )
    if not key:
        raise RuntimeError(
            "RLFUSION_ADMIN_KEY is required. Set it to a random string of at "
            "least 32 characters before starting the server."
        )
    if len(key) < _MIN_ADMIN_KEY_LEN and not allow_weak:
        raise RuntimeError(
            f"RLFUSION_ADMIN_KEY is too short ({len(key)} chars, need "
            f">= {_MIN_ADMIN_KEY_LEN}). Generate a longer one, e.g. "
            "`python -c \"import secrets; print(secrets.token_urlsafe(48))\"`."
        )


def _extract_bearer(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return ""
    return header[len("Bearer "):].strip()


def require_admin(request: Request) -> None:
    """FastAPI dependency: raise 401 unless the request carries a valid bearer.

    Use as `dependencies=[Depends(require_admin)]` on protected routes. Never
    returns a 200 with an error body. Constant-time comparison defeats timing
    attacks that would otherwise reveal partial-key information.
    """
    admin_key = get_admin_key()
    if not admin_key:
        logger.warning("Admin endpoint hit but RLFUSION_ADMIN_KEY is unset")
        raise HTTPException(status_code=401, detail="Admin authentication required.")

    presented = _extract_bearer(request)
    if not presented or not hmac.compare_digest(presented, admin_key):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")

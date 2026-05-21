# Author: Bradley R. Kinnard
"""Bearer-token admin auth for mutating endpoints.

The previous setup gated only POST /api/fine-tune behind RLFUSION_ADMIN_KEY,
and even that path returned HTTP 200 with a JSON status body on failure.
Everything else (DELETE /api/reset, PATCH /api/config, POST /api/upload,
POST /api/reindex, GET /metrics) was wide open. This module gives FastAPI
one dependency that:

- reads the key from RLFUSION_ADMIN_KEY,
- enforces a minimum length so no one ships with "changeme",
- compares the request's bearer token with hmac.compare_digest,
- raises HTTPException(401) on any mismatch.

Bind it to a route with `Depends(require_admin)` and the route never runs
for unauthenticated callers.
"""
from __future__ import annotations

import hmac
import logging
import os

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# Minimum admin key length. Short keys (changeme, password, dev) are brute-
# forceable in seconds on a public endpoint; 32 chars matches typical
# 256-bit secret hygiene.
ADMIN_KEY_MIN_LEN = 32

_BEARER = "Bearer "


def _load_admin_key() -> str:
    """Return the configured admin key from the env, or empty string."""
    return os.environ.get("RLFUSION_ADMIN_KEY", "")


def assert_admin_key_configured() -> None:
    """Raise at boot if no usable admin key is set.

    Call this from the lifespan handler. Servers running with mutating
    endpoints exposed should refuse to start without a real key; an
    operator who explicitly wants the demo behavior can set a long
    placeholder key and accept the risk.

    Under pytest this is a warning rather than a fatal error so tests
    can exercise the per-request require_admin path with an absent key
    (which still returns 401, the real defense). Production runs
    without PYTEST_CURRENT_TEST and fail closed.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        logger.debug("Skipping admin key boot gate under pytest.")
        return
    key = _load_admin_key()
    if not key:
        raise RuntimeError(
            "RLFUSION_ADMIN_KEY is not set. Mutating endpoints "
            "(/api/reset, /api/config, /api/upload, /api/reindex, "
            "/api/fine-tune) and /metrics require a bearer token. "
            "Generate one with `python -c \"import secrets; "
            "print(secrets.token_urlsafe(48))\"` and export it before "
            "starting the server."
        )
    if len(key) < ADMIN_KEY_MIN_LEN:
        raise RuntimeError(
            f"RLFUSION_ADMIN_KEY must be at least {ADMIN_KEY_MIN_LEN} "
            f"characters; got {len(key)}. Generate a longer one with "
            "`python -c \"import secrets; print(secrets.token_urlsafe(48))\"`."
        )


def require_admin(request: Request) -> None:
    """FastAPI dependency: 401 unless the request carries a valid bearer token.

    Reads `Authorization: Bearer <token>`, hmac-compares against
    RLFUSION_ADMIN_KEY. Treats a missing or short admin key on the
    server as an unauthorized response so misconfiguration cannot
    accidentally fail-open.
    """
    admin_key = _load_admin_key()
    if not admin_key or len(admin_key) < ADMIN_KEY_MIN_LEN:
        # Server is misconfigured; fail closed.
        logger.warning("Admin endpoint hit but RLFUSION_ADMIN_KEY is missing or too short.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Server admin key is not configured.",
        )

    header = request.headers.get("Authorization", "")
    if not header.startswith(_BEARER):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header.",
            headers={"WWW-Authenticate": 'Bearer realm="admin"'},
        )

    presented = header[len(_BEARER):].strip()
    # hmac.compare_digest is constant-time and tolerates length mismatch.
    if not hmac.compare_digest(presented.encode("utf-8"), admin_key.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin bearer token.",
            headers={"WWW-Authenticate": 'Bearer realm="admin"'},
        )

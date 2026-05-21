# Author: Bradley R. Kinnard
"""SHA-256 model integrity verification.

models/CHECKSUMS.txt is a flat list of `<hex digest>  <basename>` lines
(the same format `sha256sum` emits). At boot we verify every GGUF listed
matches; mismatches refuse to start the server with a named error.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CHECKSUMS_FILENAME = "CHECKSUMS.txt"
_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(_CHUNK_SIZE)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _parse_checksums(text: str) -> dict[str, str]:
    """Parse `sha256sum`-style lines into {basename: digest}."""
    expected: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, name = parts[0], parts[-1].lstrip("*")
        expected[name] = digest.lower()
    return expected


class ChecksumError(RuntimeError):
    """Raised when a tracked model file is missing or hash-mismatched."""


def verify_model_checksums(models_dir: Path) -> None:
    """Verify every entry in models/CHECKSUMS.txt against disk.

    No-ops cleanly if the manifest is missing (developers running off-mainline
    builds should drop the manifest entirely rather than pin a wrong hash).
    Tracked files that are missing or mismatched raise ChecksumError.
    """
    manifest = models_dir / _CHECKSUMS_FILENAME
    if not manifest.exists():
        logger.warning(
            "Model checksum manifest not found at %s; skipping integrity check. "
            "Create one with: cd models && sha256sum *.gguf > CHECKSUMS.txt",
            manifest,
        )
        return

    expected = _parse_checksums(manifest.read_text())
    if not expected:
        logger.warning("Model checksum manifest at %s is empty; skipping check.", manifest)
        return

    for name, want in expected.items():
        target = models_dir / name
        if not target.exists():
            raise ChecksumError(
                f"Tracked model file missing: {target}. "
                f"CHECKSUMS.txt lists it but it is not on disk."
            )
        got = _sha256_of(target)
        if got != want:
            raise ChecksumError(
                f"Checksum mismatch for {target}: expected {want}, got {got}. "
                "Re-download the model or update CHECKSUMS.txt if this is intentional."
            )
        logger.info("Model integrity OK: %s", name)

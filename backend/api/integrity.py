# Author: Bradley R. Kinnard
"""Startup integrity check for GGUF model artifacts.

`models/CHECKSUMS.txt` holds one `sha256  filename` line per pinned model
file, in the same format as `sha256sum`. At boot we verify every listed
file matches; mismatch refuses to start so a tampered weight file can't
quietly ride into production via `torch.load` or llama-cpp.

The CQL policy `models/rl_policy_cql.d3` is locally produced after
training and intentionally not pinned here; we set `weights_only=True`
on its torch.load instead.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CHECKSUMS_FILENAME = "CHECKSUMS.txt"
# Read in 1 MiB blocks so multi-GB GGUFs don't pin the whole file in RAM.
_BLOCK_SIZE = 1 << 20


def _read_manifest(manifest_path: Path) -> list[tuple[str, str]]:
    """Parse a sha256sum-style manifest into a list of (sha256, filename) tuples."""
    entries: list[tuple[str, str]] = []
    for raw in manifest_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Format: "<hex digest><space*>  <filename>" (sha256sum default)
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(
                f"Bad line in {manifest_path}: {line!r} (expected '<sha256>  <filename>')"
            )
        digest, name = parts[0].lower(), parts[1].lstrip("*").strip()
        entries.append((digest, name))
    return entries


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_BLOCK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_model_checksums(models_dir: Path) -> None:
    """Verify every entry in models/CHECKSUMS.txt; raise on mismatch.

    No-op (with a warning) if the manifest is absent. The manifest is
    optional in dev, mandatory in production by way of operators putting
    one in place; we don't want to block local hacking when the file is
    intentionally missing.
    """
    manifest_path = models_dir / CHECKSUMS_FILENAME
    if not manifest_path.exists():
        logger.warning(
            "models/%s missing; skipping GGUF integrity verification. "
            "Generate one with `cd models && sha256sum *.gguf > CHECKSUMS.txt` "
            "and commit the result for production.",
            CHECKSUMS_FILENAME,
        )
        return

    entries = _read_manifest(manifest_path)
    if not entries:
        logger.warning("models/%s is empty; nothing to verify.", CHECKSUMS_FILENAME)
        return

    failures: list[str] = []
    for expected, name in entries:
        target = (models_dir / name).resolve()
        if not target.is_relative_to(models_dir.resolve()):
            failures.append(f"{name}: manifest path escapes models/")
            continue
        if not target.exists():
            failures.append(f"{name}: file not found at {target}")
            continue
        actual = _sha256_file(target)
        if actual != expected:
            failures.append(
                f"{name}: sha256 mismatch (expected {expected[:12]}…, "
                f"got {actual[:12]}…)"
            )
            continue
        logger.info("integrity: %s sha256 verified", name)

    if failures:
        raise RuntimeError(
            "Model integrity check failed:\n  - "
            + "\n  - ".join(failures)
            + f"\nFix the listed files or regenerate {CHECKSUMS_FILENAME}."
        )

# Author: Bradley R. Kinnard
"""Screen retrieved chunks for prompt-injection markers before they hit the LLM.

The safety agent today only inspects the user's own query. A retrieved
document chunk, a PDF page, or a cached CAG entry can carry the same
"ignore previous instructions / you are now DAN" payload and reach the
generator inside the fused context. This module exposes one function
that filters chunks and another that wraps the surviving text in
unambiguous trust delimiters so the generator system prompt can refer to
them.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Mirrors backend.agents.safety_agent._ATTACK_PATTERNS. Kept in a separate
# module so retrievers / fusion can import without dragging the rest of
# the agents package along.
_ATTACK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(previous|above|all)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:DAN|evil|unrestricted|jailbroken)", re.IGNORECASE),
    re.compile(r"(?:system|admin)\s*:\s*override", re.IGNORECASE),
    re.compile(r"<\s*script\b", re.IGNORECASE),
    re.compile(r";\s*(?:DROP|DELETE|INSERT|UPDATE)\s+", re.IGNORECASE),
    re.compile(r"<\s*\|im_start\|\s*>", re.IGNORECASE),
    re.compile(r"<\s*\|im_end\|\s*>", re.IGNORECASE),
    re.compile(r"###\s*system\s*:", re.IGNORECASE),
]

CHUNK_BEGIN_MARKER = "<<<BEGIN UNTRUSTED RETRIEVED CONTEXT>>>"
CHUNK_END_MARKER = "<<<END UNTRUSTED RETRIEVED CONTEXT>>>"


def text_looks_adversarial(text: str) -> tuple[bool, str]:
    """Return (True, reason) if `text` matches any known injection pattern."""
    if not text:
        return False, ""
    for pat in _ATTACK_PATTERNS:
        if pat.search(text):
            return True, f"matched {pat.pattern[:60]}"
    return False, ""


def filter_adversarial_chunks(
    chunks: list[dict[str, object]],
    *,
    text_key: str = "text",
    source_label: str = "chunk",
) -> list[dict[str, object]]:
    """Drop any chunk whose text body trips the injection regex.

    Logs the source / id of dropped chunks so an operator can investigate
    a poisoned document. Returns a new list; never mutates the input.
    """
    kept: list[dict[str, object]] = []
    for chunk in chunks:
        raw = chunk.get(text_key, "")
        text = raw if isinstance(raw, str) else str(raw)
        bad, reason = text_looks_adversarial(text)
        if bad:
            ident = chunk.get("id", chunk.get("source", "?"))
            logger.warning(
                "Filtered adversarial %s '%s': %s",
                source_label, ident, reason,
            )
            continue
        kept.append(chunk)
    return kept


def wrap_untrusted_context(body: str) -> str:
    """Wrap `body` in BEGIN/END markers used by the generator system prompt.

    The markers are deliberately verbose so a model that misreads the
    delimiters as plain text still has a clear cue that what follows is
    third-party content, not an instruction.
    """
    if not body:
        return body
    return f"{CHUNK_BEGIN_MARKER}\n{body}\n{CHUNK_END_MARKER}"

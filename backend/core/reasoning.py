# Author: Bradley R. Kinnard
"""Tree-structured reasoning with Outcome-Refining Process Supervision.

Phase 5 of the RLFusion upgrade plan. Generates N candidate responses,
scores each with critique(), prunes low-reward branches, optionally refines
the top-K candidates, then selects the best by composite reward. Exploration
trees are logged to the replay buffer for RL training.

Also provides selective faithfulness gating: check_faithfulness() runs on
the hot path only for high-sensitivity queries (as flagged by decomposition).
A TTL cache avoids redundant LLM calls for repeated claim verification.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from typing import TypedDict

from backend.config import cfg
from backend.core.critique import (
    check_faithfulness,
    critique,
    log_episode_to_replay_buffer,
    strip_critique_block,
)

logger = logging.getLogger(__name__)

# Config section with safe fallbacks
_reasoning_cfg = cfg.get("reasoning", {})
_DEFAULT_BEAM_WIDTH = int(_reasoning_cfg.get("beam_width", 3))
_DEFAULT_PRUNE_THRESHOLD = float(_reasoning_cfg.get("prune_threshold", 0.3))
_DEFAULT_MAX_REFINEMENT = int(_reasoning_cfg.get("max_refinement_passes", 1))
_FAITHFULNESS_HOT = bool(_reasoning_cfg.get("faithfulness_on_hot_path", True))
_FAITHFULNESS_GATE = float(_reasoning_cfg.get("faithfulness_sensitivity_gate", 0.7))
_FAITHFULNESS_CACHE_TTL = int(_reasoning_cfg.get("faithfulness_cache_ttl_secs", 300))
_LOG_TREE = bool(_reasoning_cfg.get("log_exploration_tree", True))


# ── Typed structures ────────────────────────────────────────────────────

class CandidateResponse(TypedDict):
    """Single candidate in the ORPS exploration tree."""
    text: str
    reward: float
    factual: float
    proactivity: float
    helpfulness: float
    suggestions: list[str]
    reason: str
    refinement_pass: int


class ExplorationTree(TypedDict):
    """Full tree logged to replay buffer for offline RL training."""
    query: str
    fused_context: str
    beam_width: int
    candidates: list[CandidateResponse]
    selected_index: int
    selected_reward: float
    pruned_count: int
    refined_count: int
    faithfulness_checked: bool
    faithfulness_score: float
    elapsed_ms: float


class ReasoningResult(TypedDict):
    """Return value from run_orps_reasoning() consumed by the orchestrator."""
    response: str
    reward: float
    proactive_suggestions: list[str]
    reason: str
    candidates_explored: int
    pruned_count: int
    faithfulness_score: float


# ── Faithfulness cache ──────────────────────────────────────────────────

_faithfulness_cache: dict[str, tuple[float, bool, float]] = {}


def _cache_key(claim: str, context_hash: str) -> str:
    """Build a stable key for faithfulness cache lookups."""
    raw = f"{claim.strip().lower()}|{context_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _context_hash(context: str) -> str:
    """Quick hash of the fused context for cache keying."""
    return hashlib.sha256(context.encode()).hexdigest()[:16]


def cached_check_faithfulness(
    claim: str,
    chunks: list[dict[str, str]],
    context: str = "",
) -> tuple[bool, float]:
    """Check faithfulness with a TTL cache around the LLM call.

    Returns (supported, confidence) just like check_faithfulness(), but
    reuses recent results to avoid repeated LLM calls for the same claim
    within the same context window. The context param is used purely for
    cache keying (so results are scoped to the same retrieval context).
    """
    ctx_h = _context_hash(context or str(chunks)[:200])
    key = _cache_key(claim, ctx_h)
    now = time.time()

    # cache hit?
    if key in _faithfulness_cache:
        ts, supported, confidence = _faithfulness_cache[key]
        if now - ts < _FAITHFULNESS_CACHE_TTL:
            logger.debug("Faithfulness cache hit for claim hash %s", key[:8])
            return supported, confidence

    # cache miss, delegate to the proven LLM-backed check
    supported, confidence = check_faithfulness(claim, chunks)
    _faithfulness_cache[key] = (now, supported, confidence)

    # evict stale entries periodically (cheap, inline)
    if len(_faithfulness_cache) > 200:
        stale_keys = [
            k for k, (ts, _, _) in _faithfulness_cache.items()
            if now - ts > _FAITHFULNESS_CACHE_TTL
        ]
        for sk in stale_keys:
            del _faithfulness_cache[sk]

    return supported, confidence


def clear_faithfulness_cache() -> int:
    """Flush the faithfulness cache. Returns the number of entries cleared."""
    count = len(_faithfulness_cache)
    _faithfulness_cache.clear()
    return count


# ── Candidate generation and scoring ───────────────────────────────────

def _generate_candidates(
    query: str,
    system_prompt: str,
    user_prompt: str,
    beam_width: int,
) -> list[str]:
    """Generate beam_width candidate responses from the LLM.

    Uses progressively higher temperature to encourage diversity across
    candidates while keeping the first candidate conservative.
    """
    from backend.core.model_router import get_engine

    engine = get_engine()
    candidates: list[str] = []

    for i in range(beam_width):
        # first candidate is conservative, subsequent ones explore more
        temp = 0.2 + (i * 0.15)
        temp = min(temp, 0.8)

        content = engine.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp, num_ctx=8192,
        )
        candidates.append(content)

    return candidates


def _score_candidate(
    query: str,
    fused_context: str,
    raw_response: str,
    refinement_pass: int = 0,
) -> CandidateResponse:
    """Score a single candidate via critique(). Cheap wrapper for tree nodes."""
    result = critique(query, fused_context, raw_response)
    clean = strip_critique_block(raw_response)

    return CandidateResponse(
        text=clean,
        reward=result["reward"],
        factual=result.get("factual", result["reward"]),
        proactivity=result.get("proactivity", result["reward"]),
        helpfulness=result.get("helpfulness", result["reward"]),
        suggestions=result.get("proactive_suggestions", []),
        reason=result.get("reason", ""),
        refinement_pass=refinement_pass,
    )


def _refine_candidate(
    query: str,
    system_prompt: str,
    candidate: CandidateResponse,
    fused_context: str,
) -> str:
    """Generate a refined version of a candidate using critique feedback.

    Feeds the original response + critique scores back to the LLM with
    instructions to improve weak areas.
    """
    from backend.core.model_router import get_engine

    engine = get_engine()
    refinement_prompt = (
        f"You previously answered this question:\n\n"
        f"QUESTION: {query}\n\n"
        f"YOUR PREVIOUS ANSWER:\n{candidate['text'][:1500]}\n\n"
        f"EVALUATION:\n"
        f"- Factual accuracy: {candidate['factual']:.2f}\n"
        f"- Helpfulness: {candidate['helpfulness']:.2f}\n"
        f"- Proactivity: {candidate['proactivity']:.2f}\n"
        f"- Feedback: {candidate['reason']}\n\n"
        f"CONTEXT (for grounding):\n{fused_context[:1200]}\n\n"
        f"Write an improved answer that addresses the weaknesses. "
        f"Be more precise, cite specifics from the context, and anticipate follow-up questions."
    )

    return engine.generate(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": refinement_prompt},
        ],
        temperature=0.15, num_ctx=8192,
    )


# ── Main ORPS entry point ──────────────────────────────────────────────

def run_orps_reasoning(
    query: str,
    fused_context: str,
    system_prompt: str,
    user_prompt: str,
    sensitivity_level: float = 0.5,
    beam_width: int | None = None,
    prune_threshold: float | None = None,
    max_refinement_passes: int | None = None,
) -> ReasoningResult:
    """Execute tree-structured reasoning with ORPS.

    1. Generate beam_width candidate responses
    2. Score each candidate with critique()
    3. Prune branches below prune_threshold
    4. Refine top candidate(s) if max_refinement_passes > 0
    5. Select best response by composite reward
    6. Optionally check faithfulness for high-sensitivity queries
    7. Log exploration tree to replay buffer

    Falls back to single-candidate mode if beam_width <= 1 or if
    the LLM is unreachable (returns first candidate with default scores).
    """
    t0 = time.time()
    bw = beam_width if beam_width is not None else _DEFAULT_BEAM_WIDTH
    pt = prune_threshold if prune_threshold is not None else _DEFAULT_PRUNE_THRESHOLD
    mr = max_refinement_passes if max_refinement_passes is not None else _DEFAULT_MAX_REFINEMENT

    # clamp beam width to sane range
    bw = max(1, min(bw, 8))

    # Step 1: generate candidates
    try:
        raw_candidates = _generate_candidates(query, system_prompt, user_prompt, bw)
    except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
        logger.warning("Candidate generation failed: %s; falling back to empty", exc)
        return ReasoningResult(
            response="",
            reward=0.0,
            proactive_suggestions=[],
            reason=f"Generation failed: {exc}",
            candidates_explored=0,
            pruned_count=0,
            faithfulness_score=-1.0,
        )

    # Step 2: score all candidates
    scored: list[CandidateResponse] = []
    for raw in raw_candidates:
        scored.append(_score_candidate(query, fused_context, raw, refinement_pass=0))

    # Step 3: prune low-reward branches
    surviving = [c for c in scored if c["reward"] >= pt]
    pruned_count = len(scored) - len(surviving)

    if not surviving:
        # all pruned, keep the best of the originals
        surviving = [max(scored, key=lambda c: c["reward"])]
        pruned_count = len(scored) - 1

    # Step 4: refine top candidate(s)
    refined_count = 0
    for pass_num in range(1, mr + 1):
        best = max(surviving, key=lambda c: c["reward"])
        # skip refinement if already very good
        if best["reward"] >= 0.9:
            break
        try:
            refined_text = _refine_candidate(query, system_prompt, best, fused_context)
            refined_candidate = _score_candidate(
                query, fused_context, refined_text, refinement_pass=pass_num
            )
            surviving.append(refined_candidate)
            refined_count += 1
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
            logger.warning("Refinement pass %d failed: %s", pass_num, exc)

    # Step 5: select best
    all_candidates = scored + [c for c in surviving if c not in scored]
    best = max(surviving, key=lambda c: c["reward"])
    selected_idx = all_candidates.index(best) if best in all_candidates else 0

    # Step 6: faithfulness check (selective, hot path)
    faith_score = -1.0
    faith_checked = False
    if _FAITHFULNESS_HOT and sensitivity_level >= _FAITHFULNESS_GATE:
        faith_checked = True
        # split response into claim-sized sentences for verification
        sentences = [
            s.strip() for s in best["text"].replace("\n", " ").split(".")
            if len(s.strip()) > 20
        ]
        if sentences:
            chunks = [{"text": fused_context[:2000]}]
            supported_count = 0
            for sentence in sentences[:5]:
                supported, conf = cached_check_faithfulness(
                    sentence, chunks, fused_context
                )
                if supported:
                    supported_count += 1
            faith_score = supported_count / len(sentences[:5])
            logger.info(
                "Faithfulness check: %d/%d claims supported (score=%.2f)",
                supported_count, len(sentences[:5]), faith_score,
            )

    elapsed = (time.time() - t0) * 1000

    # Step 7: log exploration tree to replay buffer
    if _LOG_TREE:
        tree = ExplorationTree(
            query=query,
            fused_context=fused_context[:500],
            beam_width=bw,
            candidates=all_candidates,
            selected_index=selected_idx,
            selected_reward=best["reward"],
            pruned_count=pruned_count,
            refined_count=refined_count,
            faithfulness_checked=faith_checked,
            faithfulness_score=faith_score,
            elapsed_ms=elapsed,
        )
        _log_exploration_tree(tree, query, best)

    logger.info(
        "ORPS complete: %d candidates, %d pruned, %d refined, "
        "best_reward=%.2f, faith=%.2f, took=%.0fms",
        len(all_candidates), pruned_count, refined_count,
        best["reward"], faith_score, elapsed,
    )

    return ReasoningResult(
        response=best["text"],
        reward=best["reward"],
        proactive_suggestions=best["suggestions"],
        reason=best["reason"],
        candidates_explored=len(all_candidates),
        pruned_count=pruned_count,
        faithfulness_score=faith_score,
    )


def _log_exploration_tree(
    tree: ExplorationTree,
    query: str,
    best: CandidateResponse,
) -> None:
    """Persist the ORPS tree to the replay buffer as an enriched episode."""
    try:
        log_episode_to_replay_buffer({
            "query": query,
            "response": best["text"],
            "reward": best["reward"],
            "weights": {"rag": 0.0, "cag": 0.0, "graph": 0.0},
            "proactive_suggestions": best["suggestions"],
            "fused_context": tree["fused_context"],
        })
    except (OSError, sqlite3.Error, KeyError, TypeError) as exc:
        logger.warning("Failed to log exploration tree: %s", exc)


# ── Selective faithfulness for the critique agent ───────────────────────

def should_check_faithfulness(sensitivity_level: float) -> bool:
    """Gate for hot-path faithfulness: only for high-sensitivity queries."""
    return _FAITHFULNESS_HOT and sensitivity_level >= _FAITHFULNESS_GATE


def run_selective_faithfulness(
    response: str,
    fused_context: str,
    sensitivity_level: float,
) -> tuple[bool, float]:
    """Run faithfulness on the hot path if sensitivity exceeds the gate.

    Returns (checked, score). If not checked, returns (False, -1.0).
    """
    if not should_check_faithfulness(sensitivity_level):
        return False, -1.0

    sentences = [
        s.strip() for s in response.replace("\n", " ").split(".")
        if len(s.strip()) > 20
    ]
    if not sentences:
        return True, 1.0

    chunks = [{"text": fused_context[:2000]}]
    supported = 0
    for sentence in sentences[:5]:
        ok, _ = cached_check_faithfulness(sentence, chunks, fused_context)
        if ok:
            supported += 1

    score = supported / len(sentences[:5])
    return True, score

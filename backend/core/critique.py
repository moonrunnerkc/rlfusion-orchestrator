# Author: Bradley R. Kinnard
# critique.py - inline self-critique + episode logging for RL training
# Originally built for personal offline use, now open-sourced for public benefit.

import hashlib
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple
from backend.config import cfg, PROJECT_ROOT

logger = logging.getLogger(__name__)

# Inline critique instruction - appended to system prompt for single-pass scoring
INLINE_CRITIQUE_INSTRUCTION = """
After your answer, add a self-critique in this exact format:

<critique>
Factual accuracy: X.XX/1.00
Proactivity score: X.XX/1.00
Helpfulness: X.XX/1.00
Citation coverage: X.XX/1.00
Final reward: X.XX
Proactive suggestions:
• [Follow-up question user might ask]
• [Another topic to explore]
</critique>

CITATION RULES:
- Cite sources as [1], [2], [3] inline
- Every factual claim needs a citation
- Citation coverage = cited claims / total claims

Write suggestions as USER questions, not self-instructions.
Be harsh but fair.
"""

# Regex patterns - match with or without closing tag (LLM often omits </critique>)
_CRITIQUE_BLOCK_RE = re.compile(r"<critique>(.*?)(?:</critique>|$)", re.DOTALL | re.IGNORECASE)
_CITATION_RE = re.compile(r'\[(\d+)\]')
_NUM = r"-?[0-9]+\.?[0-9]*"
_SCORE_RE = re.compile(rf"(?:final reward|reward)[:\s]*({_NUM})", re.IGNORECASE)
_FACTUAL_RE = re.compile(rf"factual accuracy[:\s]*({_NUM})", re.IGNORECASE)
_PROACTIVE_RE = re.compile(rf"proactivity score[:\s]*({_NUM})", re.IGNORECASE)
_HELPFULNESS_RE = re.compile(rf"helpfulness[:\s]*({_NUM})", re.IGNORECASE)
_SUGGESTION_RE = re.compile(r"[•\-\*]\s*(.+)")

# Internal tag patterns to strip from output
_SOURCE_TAG_PATTERNS = [
    r'\[RAG:[^\]]+\]\s*',
    r'\[CAG:[^\]]+\]\s*',
    r'\[GRAPH:[^\]]+\]\s*',
    r'\[WEB:[^\]]+\]\s*',
]


def get_critique_instruction() -> str:
    """No longer injected into system prompts. Critique runs as a dedicated
    post-generation LLM call via _run_critique_llm(). Returning empty string
    prevents the model from outputting raw score placeholders in responses."""
    return ""


def count_citations(response: str) -> dict:
    citations = _CITATION_RE.findall(response)
    sentences = [s.strip() for s in re.split(r'[.!?]+', response) if len(s.strip()) > 20]
    coverage = len(citations) / max(len(sentences), 1)
    logger.debug("Citations: %d across %d sentences (coverage=%.2f)", len(citations), len(sentences), coverage)
    return {
        "total_citations": len(citations),
        "unique_sources": len(set(citations)),
        "sentence_count": len(sentences),
        "coverage_ratio": min(coverage, 1.0)
    }


def parse_inline_critique(response: str) -> Tuple[str, Dict[str, Any]]:
    result = {
        "reward": 0.75, "factual": 0.75, "proactivity": 0.75, "helpfulness": 0.75,
        "citation_coverage": 0.0, "proactive_suggestions": ["Tell me more about this topic"],
        "reason": "No critique block found"
    }

    citation_stats = count_citations(response)
    result["citation_coverage"] = citation_stats["coverage_ratio"]

    match = _CRITIQUE_BLOCK_RE.search(response)
    if not match:
        return response.strip(), result

    critique_text = match.group(1)
    cleaned = _CRITIQUE_BLOCK_RE.sub("", response).strip()

    # parse scores
    for pattern, key in [(_FACTUAL_RE, "factual"), (_PROACTIVE_RE, "proactivity"), (_HELPFULNESS_RE, "helpfulness")]:
        m = pattern.search(critique_text)
        if m:
            try:
                result[key] = max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                pass

    reward_match = _SCORE_RE.search(critique_text)
    if reward_match:
        try:
            result["reward"] = max(0.0, min(1.0, float(reward_match.group(1))))
        except ValueError:
            pass
    else:
        result["reward"] = (result["factual"] + result["proactivity"] + result["helpfulness"]) / 3.0

    suggestions = _SUGGESTION_RE.findall(critique_text)
    # Filter out generic filler suggestions
    _GENERIC = {"tell me more", "learn more", "know more", "elaborate", "explain further"}
    filtered = [s.strip() for s in suggestions
                if s.strip() and not any(g in s.strip().lower() for g in _GENERIC)]
    result["proactive_suggestions"] = filtered[:3] if filtered else ["Tell me more about this topic"]
    result["reason"] = f"Self-critique: {result['reward']:.2f}"

    return cleaned, result


# Prompt for the dedicated critique LLM call. JSON is far more reliable
# than asking the model to embed <critique> XML inside its own response.
_CRITIQUE_EVAL_PROMPT = """You are a strict response evaluator. Score the AI response on three axes.
Return ONLY valid JSON with no other text.

{{"factual_accuracy": 0.0-1.0, "proactivity": 0.0-1.0, "helpfulness": 0.0-1.0, "follow_up_questions": ["<specific question 1>", "<specific question 2>", "<specific question 3>"]}}

Scoring guide:
- factual_accuracy: Are claims grounded in the provided context? Penalize fabrication.
- proactivity: Does the response anticipate the user's next need?
- helpfulness: Is it directly useful, well-structured, and complete?
- follow_up_questions: Write 3 specific questions the USER would logically ask next.
  Each question MUST reference a concrete detail from the response (a name, concept, or term).
  NEVER return placeholder text or generic filler.

User query: {query}
Context (abbreviated): {context}
AI response (abbreviated): {response}

JSON only:"""

# JSON extraction: greedy match for the first { ... } block
_JSON_BLOCK_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)


def _run_critique_llm(query: str, fused_context: str, response: str) -> Dict[str, Any]:
    """Dedicated LLM call to evaluate a response. Returns parsed scores + suggestions."""
    import json as _json
    try:
        from backend.core.model_router import get_engine
        engine = get_engine()

        prompt = _CRITIQUE_EVAL_PROMPT.format(
            query=query[:500],
            context=fused_context[:800],
            response=response[:1200],
        )

        # critique timeout is configurable per host; defaults to 60s because
        # local CPU inference of a 30B model takes longer than a vLLM call would
        critique_timeout = float(cfg.get("critique", {}).get("timeout_secs", 60))
        content = engine.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, num_predict=300, num_ctx=4096, timeout=critique_timeout,
        ).strip()

        # try JSON parse, fall back to regex extraction from partial output
        parsed: dict[str, Any] = {}
        try:
            raw_parsed = _json.loads(content)
            if isinstance(raw_parsed, dict):
                parsed = raw_parsed
        except _json.JSONDecodeError:
            pass

        if not parsed:
            m = _JSON_BLOCK_RE.search(content)
            if m:
                try:
                    raw_parsed = _json.loads(m.group(0))
                    if isinstance(raw_parsed, dict):
                        parsed = raw_parsed
                except _json.JSONDecodeError:
                    pass

        if not parsed:
            logger.debug("Critique LLM returned unparseable output: %.200s", content)
            # last resort: pull numbers with the existing score regexes
            factual = _FACTUAL_RE.search(content)
            proactive = _PROACTIVE_RE.search(content)
            helpful = _HELPFULNESS_RE.search(content)
            f_val = float(factual.group(1)) if factual else 0.70
            p_val = float(proactive.group(1)) if proactive else 0.70
            h_val = float(helpful.group(1)) if helpful else 0.70
            return {
                "factual": max(0.0, min(1.0, f_val)),
                "proactivity": max(0.0, min(1.0, p_val)),
                "helpfulness": max(0.0, min(1.0, h_val)),
                "follow_up_questions": [],
            }

        def _unwrap_score(val: Any) -> float:
            """Extract a float score from flat or nested LLM JSON output.

            Handles: 0.9, {"@value": 0.9}, {"value": 0.9}, {"score": 0.9}
            dolphin-llama3:8b sometimes wraps scores in nested objects.
            """
            if isinstance(val, (int, float)):
                return max(0.0, min(1.0, float(val)))
            if isinstance(val, dict):
                for key in ("@value", "value", "score"):
                    if key in val:
                        try:
                            return max(0.0, min(1.0, float(val[key])))
                        except (TypeError, ValueError):
                            pass
            try:
                return max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError):
                return 0.70

        def _unwrap_questions(val: Any) -> list[str]:
            """Extract follow-up questions from flat or nested LLM JSON.

            Handles: ["q1", "q2"], {"@list": ["q1", "q2"]},
            {"items": ["q1"]}, or nested dicts with string values.
            """
            if isinstance(val, list):
                return [s for s in val if isinstance(s, str) and s.strip()]
            if isinstance(val, dict):
                for key in ("@list", "items", "questions"):
                    if key in val and isinstance(val[key], list):
                        return [s for s in val[key] if isinstance(s, str) and s.strip()]
                # last resort: collect all string values from the dict
                return [v for v in val.values() if isinstance(v, str) and len(v) > 10]
            return []

        return {
            "factual": _unwrap_score(parsed.get("factual_accuracy", 0.70)),
            "proactivity": _unwrap_score(parsed.get("proactivity", 0.70)),
            "helpfulness": _unwrap_score(parsed.get("helpfulness", 0.70)),
            "follow_up_questions": _unwrap_questions(parsed.get("follow_up_questions", []))[:3],
        }

    except Exception as exc:
        logger.warning("Critique LLM call failed: %s", exc)
        return {"factual": 0.70, "proactivity": 0.70, "helpfulness": 0.70, "follow_up_questions": []}


def _build_fallback_suggestions(query: str, response: str) -> list[str]:
    """Extract follow-up suggestions from the response when LLM critique fails.

    Pulls key nouns/concepts from the response and builds natural questions.
    Much better than the old 'What are the key considerations for {query}?'
    which dumped the entire query verbatim into a generic template.
    """
    # extract capitalized multi-word phrases as candidate topics
    candidates = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', response)
    # also grab technical terms (acronyms, terms with numbers/underscores)
    tech_terms = re.findall(r'\b([A-Z]{2,}[a-z]*(?:\s+[A-Z][a-z]+)*)\b', response)
    # filter noise: short, generic, or already in query
    q_lower = query.lower()
    seen: set[str] = set()
    topics: list[str] = []
    for term in candidates + tech_terms:
        t = term.strip()
        if len(t) < 4 or t.lower() in q_lower or t.lower() in seen:
            continue
        # skip articles, pronouns, generic starters
        if re.match(r'^(?:The|This|That|These|Also|More|Your|Each)\b', t):
            continue
        seen.add(t.lower())
        topics.append(t)
        if len(topics) >= 6:
            break

    # build natural questions from extracted topics
    templates = [
        "How does {topic} work in practice?",
        "What role does {topic} play in the pipeline?",
        "Can you explain {topic} in more detail?",
    ]
    suggestions: list[str] = []
    for i, topic in enumerate(topics[:3]):
        template = templates[i % len(templates)]
        suggestions.append(template.format(topic=topic))

    # last resort: short, generic but not hideous
    if not suggestions:
        suggestions = ["What would you like to explore next?"]

    return suggestions


def critique(query: str, fused_context: str, response: str) -> Dict[str, Any]:
    """Evaluate response quality via a dedicated LLM call. Model-agnostic.

    Always runs the dedicated _run_critique_llm scorer. The legacy
    'inline critique block' path is gone: v2.0.0 stopped injecting the
    critique instruction into system prompts, so any <critique> block
    that appears in a response is the model hallucinating after seeing
    the schema in retrieved context. Trusting it would let the reward
    signal be set by the same model that produced the answer.
    """
    scale = cfg.get("critique", {}).get("reward_scale", 1.0)

    scores = _run_critique_llm(query, fused_context, response)
    reward = (scores["factual"] + scores["proactivity"] + scores["helpfulness"]) / 3.0

    # filter generic filler and echoed template placeholders
    _JUNK = {"tell me more", "learn more", "know more", "elaborate", "explain further",
            "q1", "q2", "q3", "<specific question 1>", "<specific question 2>",
            "<specific question 3>", "specific question"}
    suggestions = [
        s.strip() for s in scores.get("follow_up_questions", [])
        if s.strip() and not any(g in s.strip().lower() for g in _JUNK)
        and len(s.strip()) > 10  # reject very short placeholder-like strings
    ]
    if not suggestions:
        suggestions = _build_fallback_suggestions(query, response)

    return {
        "reward": min(reward * scale, 1.0),
        "reason": f"LLM critique: factual={scores['factual']:.2f} proactivity={scores['proactivity']:.2f} helpful={scores['helpfulness']:.2f}",
        "proactive_suggestions": suggestions[:3],
    }


# Well-formed critique blocks only. A bare opening tag with no closing is
# a hallucination, not a signal: strip_critique_block leaves it alone.
_PAIRED_CRITIQUE_RE = re.compile(
    r"<critique>.*?</critique>", re.DOTALL | re.IGNORECASE,
)


def strip_critique_block(response: str) -> str:
    """Remove well-formed <critique>...</critique> blocks and source tags.

    Two invariants:

    1. Only well-formed pairs are stripped. A bare <critique> with no
       closing tag is treated as content, not a marker. The old behavior
       (strip to EOF on any opening tag) silently deleted entire
       responses when the model emitted one stray token.

    2. Stripping never produces an empty string. If removing the
       critique pair would leave nothing, the original is returned
       verbatim. The user has to see something; the alternative is an
       empty bubble that hides the bug. The reward scorer is a separate
       LLM call that does not depend on this text being clean.
    """
    text = response
    stripped = _PAIRED_CRITIQUE_RE.sub("", text)
    for pattern in _SOURCE_TAG_PATTERNS:
        stripped = re.sub(pattern, '', stripped)
    stripped = stripped.strip()
    if not stripped:
        return text.strip()
    return stripped


def compute_reward(query: str, fused_context: str, response: str) -> float:
    try:
        return float(critique(query, fused_context, response)["reward"])
    except Exception:
        return 0.75


def log_episode_to_replay_buffer(episode: dict) -> bool:
    """Log an episode to the replay buffer database for RL training.

    Episode dict shape:
        query, response, reward            — turn outcome
        weights                            — {"cag": ..., "graph": ...}
        fused_context, proactive_suggestions
        obs_features                       — list[float] (10) from obs_builder
        from_cache                         — bool, CAG fast-path hit
        policy_weights / effective_weights — list[float] (2) for F1.7
        had_empty_path                     — bool, true when rebalancer fired
        policy_action                      — list[float] (2) pre-softmax

    The trainer reads `policy_weights` so it learns from the policy's actual
    decision, not the empty-path override.
    """
    import json as _json

    from backend.config import cfg as _cfg

    db_path = PROJECT_ROOT / "db" / "rlfo_cache.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(str(db_path))
        cols = {r[1] for r in conn.execute("PRAGMA table_info(episodes)").fetchall()}

        weights = episode.get("weights", {}) or {}
        cag_w = float(weights.get("cag", 0.0))
        graph_w = float(weights.get("graph", 0.0))
        reward = float(episode.get("reward", 0.0))
        query_val = str(episode.get("query", ""))
        response_val = str(episode.get("response", ""))
        fused_ctx = str(episode.get("fused_context", ""))
        proactive = " | ".join(episode.get("proactive_suggestions", []) or [])

        def _maybe(name: str, value: Any) -> tuple[str, Any] | None:
            if name in cols:
                return name, value
            return None

        columns: list[str] = ["query", "response", "reward", "cag_weight", "graph_weight",
                              "fused_context", "proactive_suggestions"]
        values: list[Any] = [query_val, response_val, reward, cag_w, graph_w,
                             fused_ctx, proactive]

        for entry in (
            _maybe("obs_features", _json.dumps(episode["obs_features"]) if episode.get("obs_features") is not None else None),
            _maybe("from_cache", 1 if episode.get("from_cache") else 0),
            _maybe("policy_weights", _json.dumps(episode["policy_weights"]) if episode.get("policy_weights") is not None else None),
            _maybe("effective_weights", _json.dumps(episode["effective_weights"]) if episode.get("effective_weights") is not None else None),
            _maybe("had_empty_path", 1 if episode.get("had_empty_path") else 0),
            _maybe("policy_action", _json.dumps(episode["policy_action"]) if episode.get("policy_action") is not None else None),
        ):
            if entry is None:
                continue
            name, value = entry
            columns.append(name)
            values.append(value)

        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT INTO episodes ({', '.join(columns)}) VALUES ({placeholders})"
        cursor = conn.execute(sql, values)
        episode_id = cursor.lastrowid

        reinsert_thresh = float(_cfg.get("cag", {}).get("reinsert_reward_threshold", 0.70))
        cache_thresh = float(_cfg.get("cag", {}).get("cache_threshold", 0.85))
        if reward >= reinsert_thresh and not episode.get("from_cache"):
            query_key = query_val.strip()
            if query_key and response_val:
                cache_score = max(cache_thresh + 0.05, reward)
                key_hash = hashlib.sha256(query_key.strip().lower().encode("utf-8")).hexdigest()
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache (key, key_hash, value, score) VALUES (?, ?, ?, ?)",
                        (query_key, key_hash, response_val, cache_score),
                    )
                except sqlite3.OperationalError:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache (key, value, score) VALUES (?, ?, ?)",
                        (query_key, response_val, cache_score),
                    )
                logger.info(
                    "High-quality episode cached in CAG (reward=%.2f, cached as %.2f)",
                    reward, cache_score,
                )

        conn.commit()
        conn.close()
        logger.info(
            "Episode #%d logged | reward=%.2f | from_cache=%s | query='%s...'",
            episode_id, reward, bool(episode.get("from_cache")), query_val[:50],
        )
        return True

    except Exception as e:
        logger.error("Failed to log episode: %s", e)
        return False


# Unsafe content categories for fast blocklist check
_UNSAFE_KEYWORDS: frozenset[str] = frozenset({
    "how to make a bomb", "how to hack", "how to kill",
    "synthesize drugs", "make methamphetamine", "build a weapon",
    "child exploitation", "child abuse", "csam",
    "create malware", "write ransomware", "ddos attack",
    "suicide method", "self-harm instructions",
    "terrorist attack", "biological weapon", "chemical weapon",
})


def _keyword_blocklist_check(query: str) -> tuple[bool, str]:
    """Fast set-membership check against known unsafe content categories."""
    q_lower = query.lower()
    for phrase in _UNSAFE_KEYWORDS:
        if phrase in q_lower:
            return (False, f"Query matched unsafe content category")
    return (True, "")


def check_safety(query: str) -> tuple:
    """Three-tier safety without LLM on the hot path. ~20 ms total.

    Tier 1: Regex attack pre-filter (0 ms, in SafetyAgent)
    Tier 2: Keyword blocklist (~0.1 ms)
    Tier 3: OOD detector via Mahalanobis distance (~20 ms)
    LLM fallback only for amber-zone OOD distances (async, <5% of queries).
    """
    # Tier 2: keyword blocklist
    safe, reason = _keyword_blocklist_check(query)
    if not safe:
        logger.warning("Safety blocklist triggered: '%s...'", query[:50])
        return (False, reason)

    # Tier 3: OOD detection
    try:
        from backend.core.utils import embed_text, mahalanobis_distance
        q_emb = embed_text(query)
        dist = mahalanobis_distance(q_emb)
        if dist > 50.0:
            logger.warning("Safety OOD flagged (distance=%.2f): '%s...'", dist, query[:50])
            return (False, "Query flagged as out-of-distribution (potential adversarial input)")
        # amber zone: OOD detector is uncertain, but not blocking
        if dist > 20.0:
            logger.info("Safety amber zone (distance=%.2f): '%s...'", dist, query[:50])
    except (ImportError, RuntimeError) as exc:
        logger.debug("OOD check unavailable in check_safety: %s", exc)

    return (True, "Safe")


def check_faithfulness(claim: str, chunks: list) -> tuple:
    if not chunks:
        return (False, 0.0)

    try:
        from backend.core.model_router import get_engine
        engine = get_engine()

        chunk_text = "\n---\n".join([c.get("text", str(c))[:500] for c in chunks[:5]])
        prompt = f"""Sources:\n{chunk_text}\n\nIs this claim SUPPORTED? (one word)\nClaim: "{claim}" """

        result = engine.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, num_ctx=2048,
        ).upper()
        supported = "SUPPORTED" in result and "NOT" not in result
        return (supported, 0.9 if supported else 0.1)

    except Exception:
        return (False, 0.5)



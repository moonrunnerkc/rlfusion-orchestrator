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
_SCORE_RE = re.compile(r"(?:final reward|reward)[:\s]*([0-9]+\.?[0-9]*)", re.IGNORECASE)
_FACTUAL_RE = re.compile(r"factual accuracy[:\s]*([0-9]+\.?[0-9]*)", re.IGNORECASE)
_PROACTIVE_RE = re.compile(r"proactivity score[:\s]*([0-9]+\.?[0-9]*)", re.IGNORECASE)
_HELPFULNESS_RE = re.compile(r"helpfulness[:\s]*([0-9]+\.?[0-9]*)", re.IGNORECASE)
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

    Falls back to inline <critique> parsing if the LLM call returns nothing,
    and to static defaults if both fail. Return shape is frozen.
    """
    scale = cfg.get("critique", {}).get("reward_scale", 1.0)

    # first try: inline parse (cheap, no LLM call) for models that do produce the block
    _, inline = parse_inline_critique(response)
    if inline["reason"] != "No critique block found":
        return {
            "reward": min(inline["reward"] * scale, 1.0),
            "reason": inline["reason"],
            "proactive_suggestions": inline["proactive_suggestions"],
        }

    # second try: dedicated LLM evaluation call
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


def strip_critique_block(response: str) -> str:
    text = response
    text = _CRITIQUE_BLOCK_RE.sub("", text)

    for pattern in _SOURCE_TAG_PATTERNS:
        text = re.sub(pattern, '', text)

    # remove "Critique" or "Self-Critique" header and everything after
    for marker in [r'\*{0,2}Self[- ]?Critique\*{0,2}:?\s*.*$',
                   r'\*{0,2}Critique\*{0,2}:?\s*.*$',
                   r'#{1,3}\s*(?:Self[- ]?)?Critique.*$',
                   r'\nSelf-\s*$']:
        text = re.sub(marker, '', text, flags=re.IGNORECASE | re.DOTALL)

    # remove standalone score lines (digits, X.XX placeholders, or any value)
    score_patterns = [
        r'Factual accuracy:?\s*[\dXx\./_]+.*?(?=\n|$)',
        r'Proactivity score:?\s*[\dXx\./_]+.*?(?=\n|$)',
        r'Helpfulness:?\s*[\dXx\./_]+.*?(?=\n|$)',
        r'Citation coverage:?\s*[\dXx\./_]+.*?(?=\n|$)',
        r'Final reward:?\s*[\dXx\./_]+.*?(?=\n|$)',
        r'Proactive suggestions:.*$',
    ]
    for pattern in score_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    return text.strip()


def compute_reward(query: str, fused_context: str, response: str) -> float:
    try:
        return float(critique(query, fused_context, response)["reward"])
    except Exception:
        return 0.75


def log_episode_to_replay_buffer(episode: dict) -> bool:
    """Log an episode to the replay buffer database for RL training.

    Writes to the 2-path episodes schema (cag_weight, graph_weight). If
    an older 4-path schema is present, falls back to writing rag_weight=0
    so the insert still succeeds — migration drops the column on the
    next ./scripts/init_db.sh run.
    """
    db_path = PROJECT_ROOT / "db" / "rlfo_cache.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT, response TEXT, reward REAL,
                cag_weight REAL, graph_weight REAL,
                fused_context TEXT, proactive_suggestions TEXT
            )
        """)

        cols = [r[1] for r in conn.execute("PRAGMA table_info(episodes)").fetchall()]
        has_rag = "rag_weight" in cols

        if has_rag:
            cursor = conn.execute("""
                INSERT INTO episodes (query, response, reward, rag_weight, cag_weight, graph_weight, fused_context, proactive_suggestions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.get("query", ""),
                episode.get("response", ""),
                episode.get("reward", 0.0),
                0.0,
                episode.get("weights", {}).get("cag", 0.0),
                episode.get("weights", {}).get("graph", 0.0),
                episode.get("fused_context", ""),
                " | ".join(episode.get("proactive_suggestions", [])),
            ))
        else:
            cursor = conn.execute("""
                INSERT INTO episodes (query, response, reward, cag_weight, graph_weight, fused_context, proactive_suggestions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.get("query", ""),
                episode.get("response", ""),
                episode.get("reward", 0.0),
                episode.get("weights", {}).get("cag", 0.0),
                episode.get("weights", {}).get("graph", 0.0),
                episode.get("fused_context", ""),
                " | ".join(episode.get("proactive_suggestions", [])),
            ))

        episode_id = cursor.lastrowid
        reward = episode.get("reward", 0.0)

        # cache high-quality responses for instant future retrieval
        cag_reinsert_threshold = float(
            cfg.get("cag", {}).get("reinsert_reward_threshold", 0.70)
        )
        if reward >= cag_reinsert_threshold:
            query_key = episode.get("query", "").strip()
            response_val = episode.get("response", "")
            if query_key and response_val:
                cache_score = max(0.90, reward)
                key_hash = hashlib.sha256(query_key.strip().lower().encode("utf-8")).hexdigest()
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache (key, key_hash, value, score) VALUES (?, ?, ?, ?)",
                        (query_key, key_hash, response_val, cache_score),
                    )
                except sqlite3.OperationalError:
                    # fallback if key_hash column doesn't exist yet
                    conn.execute(
                        "INSERT OR REPLACE INTO cache (key, value, score) VALUES (?, ?, ?)",
                        (query_key, response_val, cache_score),
                    )
                logger.info("High-quality episode cached in CAG (reward=%.2f, cached as %.2f)", reward, cache_score)

        conn.commit()
        conn.close()
        logger.info("Episode #%d logged | reward=%.2f | query='%s...'", episode_id, reward, episode.get('query', '')[:50])
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



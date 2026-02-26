# Author: Bradley R. Kinnard
# critique.py - inline self-critique + episode logging for RL training
# Originally built for personal offline use, now open-sourced for public benefit.

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

        content = engine.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, num_predict=300, num_ctx=4096, timeout=15.0,
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
    """Log an episode to the replay buffer database for RL training."""
    db_path = PROJECT_ROOT / "db" / "rlfo_cache.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT, response TEXT, reward REAL,
                rag_weight REAL, cag_weight REAL, graph_weight REAL,
                fused_context TEXT, proactive_suggestions TEXT
            )
        """)

        cursor = conn.execute("""
            INSERT INTO episodes (query, response, reward, rag_weight, cag_weight, graph_weight, fused_context, proactive_suggestions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode.get("query", ""),
            episode.get("response", ""),
            episode.get("reward", 0.0),
            episode.get("weights", {}).get("rag", 0.0),
            episode.get("weights", {}).get("cag", 0.0),
            episode.get("weights", {}).get("graph", 0.0),
            episode.get("fused_context", ""),
            " | ".join(episode.get("proactive_suggestions", []))
        ))

        episode_id = cursor.lastrowid
        reward = episode.get("reward", 0.0)

        # cache high-quality responses for instant future retrieval
        if reward >= 0.70:
            query_key = episode.get("query", "").strip()
            response_val = episode.get("response", "")
            if query_key and response_val:
                cache_score = max(0.90, reward)
                conn.execute("INSERT OR REPLACE INTO cache (key, value, score) VALUES (?, ?, ?)",
                           (query_key, response_val, cache_score))
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


# ---------------------------------------------------------------------------
# STIS Contradiction Trigger (Phase 2)
# ---------------------------------------------------------------------------
# Hard threshold: if RAG content contradicts authoritative ontology facts
# AND the best CSWR score drops below this value, abort Ollama generation
# and route to the STIS engine for mathematically forced consensus.

STIS_CSWR_THRESHOLD = 0.70
# Minimum query-to-ontology relevance before we bother checking for contradictions.
# Set high (0.75+) to avoid false positives on tangentially-related queries.
STIS_RELEVANCE_THRESHOLD = 0.75


def _load_ontology_facts() -> list[dict[str, str]]:
    """Load authoritative facts from ontology.json. Cached in module scope."""
    import json
    from backend.config import PROJECT_ROOT

    onto_path = PROJECT_ROOT / "data" / "ontology.json"
    if not onto_path.exists():
        return []
    try:
        data = json.loads(onto_path.read_text())
        facts = []
        for node in data.get("nodes", []):
            desc = node.get("description", "").strip()
            if desc and node.get("type") == "fact":
                facts.append({
                    "id": node.get("id", ""),
                    "label": node.get("label", ""),
                    "text": desc,
                    "confidence": node.get("confidence", 1.0),
                })
        return facts
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Failed to load ontology facts: %s", exc)
        return []


# cache ontology between calls (reloaded on import)
_ontology_facts: list[dict[str, str]] | None = None


def _get_ontology_facts() -> list[dict[str, str]]:
    """Return cached ontology facts, loading once on first call."""
    global _ontology_facts
    if _ontology_facts is None:
        _ontology_facts = _load_ontology_facts()
    return _ontology_facts


def detect_contradiction(
    rag_results: list[dict[str, object]],
    graph_results: list[dict[str, object]],
    query: str = "",
) -> dict[str, object]:
    """Detect factual contradictions between RAG results and authoritative ontology.

    Uses a two-stage approach:
    1. Embedding similarity to find ontology facts RELEVANT to the query
    2. LLM-based claim comparison to detect if RAG contradicts those facts

    Pure cosine similarity cannot distinguish "uses model A" from "uses model B"
    because they share vocabulary. Only the LLM can detect factual disagreement
    between topically-similar statements.

    Returns a dict with:
        contradicted (bool): True if a factual conflict was detected
        similarity (float): relevance score of the matched ontology fact
        rag_claim (str): text of the top RAG chunk
        graph_claim (str): text of the contradicted ontology fact
    """
    from backend.core.utils import embed_text

    empty_result: dict[str, object] = {
        "contradicted": False,
        "similarity": 1.0,
        "rag_claim": "",
        "graph_claim": "",
    }

    if not rag_results:
        return empty_result

    top_rag = max(rag_results, key=lambda r: float(r.get("score", 0)))
    rag_text = str(top_rag.get("text", ""))
    if not rag_text.strip():
        return empty_result

    # Stage 1: find ontology facts relevant to the query via embedding
    ontology_facts = _get_ontology_facts()
    if not ontology_facts or not query:
        return empty_result

    query_emb = embed_text(query)
    norm_q = float((query_emb @ query_emb) ** 0.5)

    relevant_facts: list[tuple[float, dict[str, str]]] = []
    for fact in ontology_facts:
        fact_emb = embed_text(fact["text"])
        norm_f = float((fact_emb @ fact_emb) ** 0.5)
        denom = norm_q * norm_f
        relevance = float(query_emb @ fact_emb) / denom if denom > 1e-9 else 0.0
        if relevance >= STIS_RELEVANCE_THRESHOLD:
            relevant_facts.append((relevance, fact))

    if not relevant_facts:
        logger.debug("No ontology facts relevant to query (threshold=%.2f)", STIS_RELEVANCE_THRESHOLD)
        return empty_result

    # sort by relevance, check the most relevant fact first
    relevant_facts.sort(key=lambda x: x[0], reverse=True)
    best_relevance, best_fact = relevant_facts[0]

    logger.info(
        "Ontology fact relevant to query: '%s' (relevance=%.4f)",
        best_fact["label"], best_relevance,
    )

    # Stage 2: Fast embedding-based contradiction check (~0.1 ms)
    # Compare RAG text directly against the ontology fact.
    # High query-fact relevance + low rag-fact similarity = likely contradiction.
    rag_claim_short = rag_text[:600]
    onto_claim = best_fact["text"]

    rag_emb = embed_text(rag_claim_short)
    fact_emb = embed_text(onto_claim)
    norm_r = float((rag_emb @ rag_emb) ** 0.5)
    norm_f = float((fact_emb @ fact_emb) ** 0.5)
    denom = norm_r * norm_f
    rag_fact_sim = float(rag_emb @ fact_emb) / denom if denom > 1e-9 else 0.0

    # Both RAG and fact are relevant to the query, but they disagree with each other.
    # Threshold: if rag-fact similarity < 0.6 while query-fact relevance > 0.7,
    # the RAG chunk likely says something different from the verified fact.
    contradicted = rag_fact_sim < 0.6 and best_relevance > 0.7

    if contradicted:
        logger.warning(
            "Embedding contradiction: RAG vs ontology '%s' "
            "(query-fact relevance=%.4f, rag-fact sim=%.4f) | RAG='%.80s...'",
            best_fact["label"], best_relevance, rag_fact_sim, rag_text,
        )
    else:
        logger.debug(
            "No contradiction: RAG vs ontology '%s' (rag-fact sim=%.4f)",
            best_fact["label"], rag_fact_sim,
        )

    return {
        "contradicted": contradicted,
        "similarity": round(1.0 - best_relevance, 6) if contradicted else round(best_relevance, 6),
        "rag_claim": rag_text[:500],
        "graph_claim": f"[ONTOLOGY: {best_fact['label']}] {onto_claim}" if contradicted else "",
    }


def should_route_to_stis(
    rag_results: list[dict[str, object]],
    graph_results: list[dict[str, object]],
    cswr_scores: list[float] | None = None,
    query: str = "",
) -> dict[str, object]:
    """Evaluate whether to abort Ollama and fall back to the STIS engine.

    LLM-verified contradiction against ontology facts is the primary trigger.
    When a contradiction IS confirmed by the LLM, the CSWR score acts as a
    secondary gate: if CSWR is high, the retrieval may still be trustworthy
    despite the contradiction.

    Returns a dict with:
        route_to_stis (bool): True if Ollama should be aborted
        reason (str): human-readable explanation of the decision
        contradiction (dict): output of detect_contradiction()
        best_cswr (float): highest CSWR score observed
    """
    contradiction = detect_contradiction(rag_results, graph_results, query=query)

    # compute best CSWR from provided scores or from result metadata
    if cswr_scores and len(cswr_scores) > 0:
        best_cswr = max(cswr_scores)
    else:
        all_scores: list[float] = []
        for r in rag_results:
            csw = r.get("csw_score") or r.get("score", 0)
            all_scores.append(float(csw))
        for r in graph_results:
            csw = r.get("csw_score") or r.get("score", 0)
            all_scores.append(float(csw))
        best_cswr = max(all_scores) if all_scores else 0.0

    contradicted = bool(contradiction["contradicted"])
    cswr_below = best_cswr < STIS_CSWR_THRESHOLD

    # LLM-verified contradiction is a strong signal. A well-written
    # disinformation doc can score high on CSWR while being factually wrong.
    # If the LLM confirms a contradiction against the ontology, route to STIS
    # unless CSWR is exceptionally high (> 0.95, near-perfect retrieval).
    route = contradicted and (cswr_below or best_cswr < 0.95)

    if route:
        reason = (
            f"LLM-verified contradiction against ontology "
            f"with weak CSWR confidence (best={best_cswr:.4f}, threshold={STIS_CSWR_THRESHOLD}). "
            f"Routing to STIS for forced consensus."
        )
        logger.info("STIS routing triggered: %s", reason)
    elif contradicted:
        reason = (
            f"Contradiction detected but "
            f"CSWR score sufficient (best={best_cswr:.4f} >= {STIS_CSWR_THRESHOLD}). "
            f"Proceeding with Ollama."
        )
    elif cswr_below:
        reason = (
            f"CSWR below threshold (best={best_cswr:.4f} < {STIS_CSWR_THRESHOLD}) but "
            f"no contradiction detected. Proceeding with Ollama."
        )
    else:
        reason = "No contradiction, CSWR confidence adequate. Proceeding with Ollama."

    return {
        "route_to_stis": route,
        "reason": reason,
        "contradiction": contradiction,
        "best_cswr": round(best_cswr, 6),
    }

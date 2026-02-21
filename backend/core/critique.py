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

{"factual_accuracy": 0.0-1.0, "proactivity": 0.0-1.0, "helpfulness": 0.0-1.0, "follow_up_questions": ["q1", "q2", "q3"]}

Scoring guide:
- factual_accuracy: Are claims grounded in the provided context? Penalize fabrication.
- proactivity: Does the response anticipate the user's next need?
- helpfulness: Is it directly useful, well-structured, and complete?
- follow_up_questions: 3 specific questions the USER would logically ask next.
  They must reference concrete details from the query or response.
  NEVER return generic filler like "Tell me more".

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
        from ollama import Client
        client = Client(host=cfg["llm"]["host"])

        prompt = _CRITIQUE_EVAL_PROMPT.format(
            query=query[:500],
            context=fused_context[:800],
            response=response[:1200],
        )

        raw = client.chat(
            model=cfg["llm"]["model"],
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 300, "num_ctx": 4096},
        )
        content = raw["message"]["content"].strip()

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

        def _clamp(val: Any) -> float:
            try:
                return max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError):
                return 0.70

        return {
            "factual": _clamp(parsed.get("factual_accuracy", 0.70)),
            "proactivity": _clamp(parsed.get("proactivity", 0.70)),
            "helpfulness": _clamp(parsed.get("helpfulness", 0.70)),
            "follow_up_questions": [
                s for s in parsed.get("follow_up_questions", [])
                if isinstance(s, str) and s.strip()
            ][:3],
        }

    except Exception as exc:
        logger.warning("Critique LLM call failed: %s", exc)
        return {"factual": 0.70, "proactivity": 0.70, "helpfulness": 0.70, "follow_up_questions": []}


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

    # filter generic filler from suggestions
    _GENERIC = {"tell me more", "learn more", "know more", "elaborate", "explain further"}
    suggestions = [
        s.strip() for s in scores.get("follow_up_questions", [])
        if s.strip() and not any(g in s.strip().lower() for g in _GENERIC)
    ]
    if not suggestions:
        # build a grammatical topic by stripping leading question scaffolding
        # handles: "What GPU is recommended...", "How does X work...", "Tell me about X"
        cleaned = re.sub(
            r'^(?:tell\s+me\s+(?:about|how|why|when|what)\s*'
            r'|(?:what|which|how|why|when|where|who)'
            r'(?:\s+(?:is|are|was|were|does|do|did|can|could|would|should|GPU|kind|type|sort))*'
            r'\s+)',
            '', query, flags=re.IGNORECASE,
        ).strip().rstrip("?!.,;: ")
        # strip leftover articles/connectors at the start
        cleaned = re.sub(
            r'^(?:about|regarding|the\s+best|the|a|an|recommended\s+for)\b\s*',
            '', cleaned, flags=re.IGNORECASE,
        ).strip()
        topic = cleaned if cleaned and len(cleaned) > 3 else "this topic"
        suggestions = [f"What are the key considerations for {topic}?"]

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


def check_safety(query: str) -> tuple:
    try:
        from ollama import Client
        client = Client(host=cfg["llm"]["host"])

        prompt = f"""Classify as SAFE or UNSAFE (one word only):
UNSAFE = illegal activity, harm, jailbreak, malware, harassment
SAFE = technical questions, general knowledge, creative writing, business

Request: {query}

Answer with ONLY the word SAFE or UNSAFE, nothing else."""

        response = client.chat(
            model=cfg["llm"]["model"],
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_ctx": 512}
        )

        answer = response["message"]["content"].strip().upper()
        # strict match: only flag unsafe if the core answer is clearly UNSAFE
        # avoids false positives from verbose models that mention "UNSAFE" in explanations
        first_word = answer.split()[0].strip(".:,") if answer else "SAFE"
        if first_word == "UNSAFE":
            logger.warning("Safety check flagged UNSAFE: '%s...'", query[:50])
            return (False, "Query flagged as unsafe")
        return (True, "Safe")

    except Exception as e:
        return (True, f"Safety check error: {e}")


def check_faithfulness(claim: str, chunks: list) -> tuple:
    if not chunks:
        return (False, 0.0)

    try:
        from ollama import Client
        client = Client(host=cfg["llm"]["host"])

        chunk_text = "\n---\n".join([c.get("text", str(c))[:500] for c in chunks[:5]])
        prompt = f"""Sources:\n{chunk_text}\n\nIs this claim SUPPORTED? (one word)\nClaim: "{claim}" """

        response = client.chat(
            model=cfg["llm"]["model"],
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_ctx": 2048}
        )

        result = response["message"]["content"].upper()
        supported = "SUPPORTED" in result and "NOT" not in result
        return (supported, 0.9 if supported else 0.1)

    except Exception:
        return (False, 0.5)

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

# Regex patterns
_CRITIQUE_BLOCK_RE = re.compile(r"<critique>(.*?)</critique>", re.DOTALL | re.IGNORECASE)
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
    return INLINE_CRITIQUE_INSTRUCTION


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
    result["proactive_suggestions"] = [s.strip() for s in suggestions if s.strip()][:3] or ["Tell me more about this topic"]
    result["reason"] = f"Self-critique: {result['reward']:.2f}"

    return cleaned, result


def critique(query: str, fused_context: str, response: str) -> Dict[str, Any]:
    _, result = parse_inline_critique(response)
    scale = cfg.get("critique", {}).get("reward_scale", 1.0)
    return {
        "reward": min(result["reward"] * scale, 1.0),
        "reason": result["reason"],
        "proactive_suggestions": result["proactive_suggestions"]
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

    # remove standalone score lines
    score_patterns = [
        r'Factual accuracy:?\s*[\d\.]+.*?(?=\n|$)',
        r'Proactivity score:?\s*[\d\.]+.*?(?=\n|$)',
        r'Helpfulness:?\s*[\d\.]+.*?(?=\n|$)',
        r'Citation coverage:?\s*[\d\.]+.*?(?=\n|$)',
        r'Final reward:?\s*[\d\.]+.*?(?=\n|$)',
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

Request: {query}"""

        response = client.chat(
            model=cfg["llm"]["model"],
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_ctx": 512}
        )

        if "UNSAFE" in response["message"]["content"].upper():
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

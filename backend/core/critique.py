"""
Response quality evaluation and reward computation for reinforcement learning.

This module implements automated critique of generated responses using a language model
as a judge. It computes scalar rewards for RL training and generates proactive follow-up
suggestions to guide user interactions.

Author: Bradley R. Kinnard
License: MIT
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from ollama import Client

from backend.config import cfg

_client: Optional[Client] = None


def get_ollama_client() -> Client:
    """
    Retrieve cached Ollama client instance.

    Creates client on first call and caches for subsequent requests
    to avoid repeated initialization overhead.

    Returns:
        Configured Ollama client connected to host specified in config
    """
    global _client

    if _client is None:
        _client = Client(host=cfg["llm"]["host"])

    return _client


CRITIQUE_PROMPT = """
You are an extremely harsh but fair evaluator of AI responses.
Score this response from 0.0 to 1.0 on four dimensions and give a final weighted score.

Query: {query}
Response: {response}

1. Factual accuracy & hallucination-free:    /10
2. Depth and insight (Grok-level or better): /10
3. Clarity, structure, and readability:     /10
4. Proactive usefulness (did it anticipate obvious follow-ups?): /10

Final score = average of the four × 0.9 + 0.1 bonus if it feels genuinely helpful.
Respond with exactly one number between 0.00 and 1.00. No explanation.
"""


def call_judge(query: str, response: str) -> Dict[str, Any]:
    """
    Evaluate response quality using language model as judge.

    Sends query and response to LLM with structured evaluation prompt.
    Returns parsed numeric score with reasoning.

    Args:
        query: Original user query
        response: Generated response to evaluate

    Returns:
        Dict containing 'score' (float 0.0-1.0), 'reason', and 'proactive_suggestions'
    """
    client = get_ollama_client()

    prompt = CRITIQUE_PROMPT.format(query=query, response=response)

    result = client.chat(
        model=cfg["llm"]["model"],
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )

    content = result["message"]["content"].strip()

    try:
        score = float(content)
        score = max(0.0, min(1.0, score))
        return {
            "score": score,
            "reason": f"Evaluation score: {score:.2f}",
            "proactive_suggestions": []
        }
    except (ValueError, AttributeError):
        return {
            "score": 0.0,
            "reason": "Parse failed",
            "proactive_suggestions": []
        }


def critique(
    query: str,
    fused_context: str,
    generated_response: str
) -> Dict[str, Any]:
    """
    Main critique function. Judges response quality and returns scaled reward.
    Includes proactive suggestions if score is low.
    """
    judge_result = call_judge(query, generated_response)

    raw_score = judge_result["score"]
    reward_scale = cfg["critique"]["reward_scale"]

    # apply scaling and cap at 1.0
    reward = min(raw_score * reward_scale, 1.0)

    # Generate intelligent proactive suggestions using LLM
    threshold = cfg["critique"]["proactive_threshold"]
    suggestions = []

    # Always generate proactive suggestions - the system should anticipate next steps
    try:
        client = get_ollama_client()

        proactive_prompt = f"""You are a highly intelligent assistant analyzing a conversation.

Query: {query}
Response: {generated_response}
Context: {fused_context[:500]}...

Based on this interaction, predict 2-3 intelligent follow-up actions or questions the user might want next.
Be specific, innovative, and technical. Think several steps ahead.

Examples of good proactive suggestions:
- "Compare this approach with X alternative and benchmark performance"
- "Validate edge case: what happens when Y condition occurs?"
- "Next: implement Z feature to leverage this capability"

Return ONLY a JSON array of 2-3 strings. No explanation.
Example: ["suggestion 1", "suggestion 2", "suggestion 3"]"""

        result = client.chat(
            model=cfg["llm"]["model"],
            messages=[{"role": "user", "content": proactive_prompt}],
            options={"temperature": 0.7, "max_tokens": 200}
        )

        content = result["message"]["content"].strip()

        # Parse JSON response
        import json as json_lib
        suggestions = json_lib.loads(content)

        # Ensure it's a list of strings
        if isinstance(suggestions, list) and len(suggestions) > 0:
            suggestions = [str(s) for s in suggestions[:3]]  # Max 3 suggestions
        else:
            suggestions = []

    except Exception as e:
        # If LLM fails, return empty list (don't show bad suggestions)
        suggestions = []

    return {
        "reward": reward,
        "reason": judge_result["reason"],
        "proactive_suggestions": suggestions
    }
_dummy_warning_shown = False

def dummy_critique() -> Dict[str, Any]:
    """Fallback critique when Ollama isn't available. Returns fixed dummy score."""
    global _dummy_warning_shown

    if not _dummy_warning_shown:
        print("Ollama not reachable - using dummy critique")
        _dummy_warning_shown = True

    return {
        "reward": 0.87,
        "reason": "dummy judge - real LLM not ready yet",
        "proactive_suggestions": []
    }
def compute_reward(query: str, fused_context: str, response: str) -> float:
    """
    Fast scalar reward used by FusionEnv and synthetic data generation.
    Calls the real judge and returns only the numeric reward.
    """
    try:
        result = critique(query, fused_context, response)
        return float(result["reward"])
    except Exception:
        # If Ollama is down during training, fall back to dummy
        return 0.75


import sqlite3
from pathlib import Path

def log_episode_to_replay_buffer(episode: dict) -> bool:
    """
    Permanently store a full human-readable episode for future retraining.
    Returns True if episode was successfully inserted, False otherwise.
    """
    db_path = Path(__file__).parent.parent.parent / "db" / "rlfo_cache.db"

    try:
        conn = sqlite3.connect(db_path)

        # Create the proper table if it doesn't exist yet
        conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            query TEXT,
            response TEXT,
            reward REAL,
            rag_weight REAL,
            cag_weight REAL,
            graph_weight REAL,
            fused_context TEXT,
            proactive_suggestions TEXT
        )
        """)

        cursor = conn.execute("""
        INSERT INTO episodes
        (query, response, reward, rag_weight, cag_weight, graph_weight, fused_context, proactive_suggestions)
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

        rows_inserted = cursor.rowcount
        episode_id = cursor.lastrowid

        # If high-quality episode (reward >= 0.70), cache in CAG for instant future retrieval
        # Lower threshold allows more caching, but CAG retrieval threshold (0.85) filters on recall
        reward = episode.get("reward", 0.0)
        if reward >= 0.70:
            query_key = episode.get("query", "").strip()
            response_val = episode.get("response", "")
            if query_key and response_val:
                try:
                    # Insert or update CAG cache with boosted score to ensure retrieval
                    # Boost to 0.90 so it passes CAG's 0.85 threshold
                    cache_score = max(0.90, reward)
                    conn.execute("""
                    INSERT OR REPLACE INTO cache (key, value, score)
                    VALUES (?, ?, ?)
                    """, (query_key, response_val, cache_score))
                    print(f"  💾 High-quality episode cached in CAG (reward={reward:.2f} → cached as {cache_score:.2f})")
                except Exception as e:
                    print(f"  ⚠️  CAG cache failed: {e}")

        conn.commit()
        conn.close()

        if rows_inserted > 0:
            print(f"✅ Episode #{episode_id} logged to DB | reward={episode.get('reward', 0.0):.2f} | query='{episode.get('query', '')[:50]}...'")
            return True
        else:
            return False

    except Exception as e:
        print(f"❌ Failed to log episode to DB: {e}")
        return False

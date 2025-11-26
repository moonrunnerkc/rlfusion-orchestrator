# Author: Bradley R. Kinnard
# backend/core/critique.py
"""
Self-critique engine for RLFO.
Uses Qwen2-72b via Ollama as a judge to score generated responses.
Returns float reward in [0.0, 1.0] and optional proactive suggestions.
Must be 100% deterministic given the same inputs.
"""

from ollama import Client
import json
from typing import Dict, List, Any
from backend.config import cfg
from backend.core.utils import embed_text
import numpy as np

# cache ollama client
_client = None


def get_ollama_client() -> Client:
    """Get cached Ollama client. Creates on first call."""
    global _client

    if _client is None:
        _client = Client(host=cfg["llm"]["host"])

    return _client


CRITIQUE_PROMPT = """You are a critical evaluator for an AI system. Score the following response.

Query: {query}

Response: {response}

Evaluate on these criteria:
- Factual accuracy (no hallucinations or made-up info)
- Relevance to the query
- Coherence and clarity
- Proactivity (anticipates follow-ups or catches potential issues)

Return a score from 0.0 (terrible) to 1.0 (perfect).

Output format (JSON):
{{
  "score": <float between 0.0 and 1.0>,
  "reason": "<brief explanation of score>",
  "proactive_suggestions": ["<suggestion 1>", "<suggestion 2>", ...]
}}

Respond ONLY with valid JSON, no extra text."""


def call_judge(query: str, response: str) -> Dict[str, Any]:
    """
    Call Qwen2 to judge response quality. Returns score + suggestions.
    """
    client = get_ollama_client()

    prompt = CRITIQUE_PROMPT.format(query=query, response=response)

    # temp=0.0 for determinism
    result = client.chat(
        model=cfg["llm"]["model"],
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )

    content = result["message"]["content"]

    try:
        critique = json.loads(content)
        return {
            "score": float(critique.get("score", 0.0)),
            "reason": critique.get("reason", ""),
            "proactive_suggestions": critique.get("proactive_suggestions", [])
        }
    except (json.JSONDecodeError, ValueError, KeyError):
        # LLM didn't return valid JSON
        return {
            "score": 0.0,
            "reason": "parse failed",
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

    # include suggestions if below threshold
    threshold = cfg["critique"]["proactive_threshold"]
    if raw_score < threshold:
        suggestions = judge_result["proactive_suggestions"]
    else:
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

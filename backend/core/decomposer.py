# Author: Bradley R. Kinnard
# decomposer.py - query decomposition for retrieval routing
# Originally built for personal offline use, now open-sourced for public benefit.

import json
import logging
import re
from ollama import Client
from backend.config import cfg

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a query analyzer. Output ONLY valid JSON with these exact keys:
{
  "primary_intent": "explain|compare|troubleshoot|list|design|summarize",
  "required_facts": ["fact1", "fact2"],
  "key_entities": ["entity1", "entity2"],
  "temporal_focus": "past|current|future|null",
  "expected_shape": "definition|list|step-by-step|insight|code|comparison|troubleshoot",
  "sensitivity_level": 0.0-1.0
}
Respond with ONLY the JSON object. No explanation."""


def decompose_query(query: str, mode: str = "chat") -> dict:
    """Decompose a query into structured profile for CSWR scoring.
    Uses heuristic by default (0.1 ms). LLM path available via config."""
    use_llm = cfg.get("decomposer", {}).get("use_llm", False)
    if not use_llm:
        return _heuristic_decompose(query)

    return _llm_decompose(query, mode)


def _llm_decompose(query: str, mode: str = "chat") -> dict:
    """Full LLM roundtrip for structured decomposition. Slow (~4.8s) but precise."""
    try:
        client = Client(host=cfg["llm"]["host"])
        response = client.chat(
            model=cfg["llm"]["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'Mode: {mode}\nQuery: "{query}"\n\nAnalyze and respond with JSON only:'}
            ],
            options={"temperature": 0.1, "num_predict": 200}
        )

        content = response["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        parsed = json.loads(content)
        return {
            "original_query": query,
            "primary_intent": parsed.get("primary_intent", "explain"),
            "required_facts": parsed.get("required_facts", []),
            "key_entities": parsed.get("key_entities", []),
            "temporal_focus": parsed.get("temporal_focus") if parsed.get("temporal_focus") != "null" else None,
            "expected_shape": parsed.get("expected_shape", "insight"),
            "sensitivity_level": float(parsed.get("sensitivity_level", 0.5))
        }

    except (json.JSONDecodeError, KeyError, ValueError, ConnectionError) as e:
        logger.warning("LLM decomposition failed (%s), using heuristic fallback", e)
        return _heuristic_decompose(query)


def _heuristic_decompose(query: str) -> dict:
    q = query.lower()

    # intent detection
    if any(w in q for w in ["why", "how does", "what is", "explain", "define"]):
        intent, shape = "explain", "definition"
    elif any(w in q for w in ["compare", "difference", "vs", "versus"]):
        intent, shape = "compare", "comparison"
    elif any(w in q for w in ["fix", "error", "broken", "debug", "issue", "problem"]):
        intent, shape = "troubleshoot", "troubleshoot"
    elif any(w in q for w in ["list", "show me", "all", "enumerate"]):
        intent, shape = "list", "list"
    elif any(w in q for w in ["build", "create", "design", "implement"]):
        intent, shape = "design", "step-by-step"
    elif any(w in q for w in ["summarize", "summary", "overview"]):
        intent, shape = "summarize", "insight"
    else:
        intent, shape = "explain", "insight"

    # entity extraction
    entities = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', query)
    entities.extend(re.findall(r'\b[A-Z]{2,6}\b', query))
    entities = list(set(entities))[:5]

    # temporal focus
    temporal = None
    if any(w in q for w in ["was", "used to", "previously", "history"]):
        temporal = "past"
    elif any(w in q for w in ["will", "future", "plan", "upcoming"]):
        temporal = "future"
    elif any(w in q for w in ["currently", "now", "today", "latest"]):
        temporal = "current"

    return {
        "original_query": query,
        "primary_intent": intent,
        "required_facts": [],
        "key_entities": entities,
        "temporal_focus": temporal,
        "expected_shape": shape,
        "sensitivity_level": 0.8 if intent == "troubleshoot" else 0.5
    }

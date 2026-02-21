# Author: Bradley R. Kinnard
# hotpotqa.py - Multi-hop QA ground-truth benchmark for RLFO.
# Measures exact-match and F1 on questions that require chaining facts.

from __future__ import annotations

import logging
import re
import string
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np

logger = logging.getLogger(__name__)


class HotpotSample(TypedDict):
    """A single multi-hop QA evaluation item."""
    question: str
    answer: str
    supporting_titles: list[str]
    difficulty: str  # "easy", "medium", "hard"


class HotpotResult(TypedDict):
    """Evaluation result for one HotpotQA sample."""
    question: str
    predicted: str
    gold: str
    exact_match: bool
    f1: float
    latency_ms: float


@dataclass
class HotpotReport:
    """Aggregated HotpotQA benchmark metrics."""
    total_samples: int = 0
    exact_match_rate: float = 0.0
    mean_f1: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    individual_results: list[HotpotResult] = field(default_factory=list)
    elapsed_secs: float = 0.0


# Built-in RLFO-domain multi-hop samples for offline testing
_BUILTIN_SAMPLES: list[HotpotSample] = [
    {
        "question": "Which component of RLFO uses Mahalanobis distance, and what covariance method does it use?",
        "answer": "OOD detection uses Mahalanobis distance with Ledoit-Wolf shrinkage",
        "supporting_titles": ["utils", "safety"],
        "difficulty": "medium",
    },
    {
        "question": "What RL algorithm trains the fusion weights offline, and what is the observation dim?",
        "answer": "CQL trains fusion weights offline with 396-dimension observations",
        "supporting_titles": ["rl_policy", "fusion_env"],
        "difficulty": "medium",
    },
    {
        "question": "How many retrieval paths does RLFO have, and which one uses FAISS?",
        "answer": "Four retrieval paths; RAG uses FAISS IndexFlatL2",
        "supporting_titles": ["retrievers", "architecture"],
        "difficulty": "easy",
    },
    {
        "question": "What scoring method does CSWR use, and what are its four components?",
        "answer": "CSWR uses vector weight, local stability, question fit, and drift penalty",
        "supporting_titles": ["retrievers", "cswr_doc"],
        "difficulty": "hard",
    },
    {
        "question": "What model does RLFO use for generation, and what is the max context window?",
        "answer": "Llama 3.1 8B via Ollama with 8192 token context window",
        "supporting_titles": ["config", "main"],
        "difficulty": "easy",
    },
    {
        "question": "Which agent runs first in the pipeline for adversarial queries, and why?",
        "answer": "Safety agent runs first as a gate for adversarial/OOD queries to enable early exit",
        "supporting_titles": ["orchestrator", "safety_agent"],
        "difficulty": "hard",
    },
    {
        "question": "What is the critique reward formula's proactive bonus, and what threshold triggers it?",
        "answer": "0.25 proactive reward bonus triggered at 0.75 threshold",
        "supporting_titles": ["critique", "config"],
        "difficulty": "medium",
    },
    {
        "question": "How does the entity graph support retrieval, and what similarity threshold is used?",
        "answer": "Entity graph enables graph retrieval path with 0.92 entity similarity threshold",
        "supporting_titles": ["graph_engine", "config"],
        "difficulty": "hard",
    },
]


def _normalize_answer(text: str) -> str:
    """Lower, strip articles/punctuation/whitespace for fair comparison."""
    text = text.lower().strip()
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    text = " ".join(text.split())
    return text


def exact_match(predicted: str, gold: str) -> bool:
    """Normalized exact match between predicted and gold answers."""
    return _normalize_answer(predicted) == _normalize_answer(gold)


def token_f1(predicted: str, gold: str) -> float:
    """Token-level F1 between predicted and gold answers."""
    pred_tokens = _normalize_answer(predicted).split()
    gold_tokens = _normalize_answer(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def evaluate_sample(
    sample: HotpotSample,
    answer_fn: object,
) -> HotpotResult:
    """Evaluate a single HotpotQA sample.

    answer_fn: callable(question: str) -> str, returns the model's answer.
    """
    t0 = time.perf_counter()

    if callable(answer_fn):
        try:
            predicted = str(answer_fn(sample["question"]))
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.warning("Answer generation failed: %s", exc)
            predicted = ""
    else:
        predicted = ""

    latency_ms = (time.perf_counter() - t0) * 1000

    return HotpotResult(
        question=sample["question"],
        predicted=predicted,
        gold=sample["answer"],
        exact_match=exact_match(predicted, sample["answer"]),
        f1=token_f1(predicted, sample["answer"]),
        latency_ms=round(latency_ms, 2),
    )


def run_hotpotqa(
    samples: list[HotpotSample] | None = None,
    answer_fn: object = None,
    max_samples: int = 200,
) -> HotpotReport:
    """Run HotpotQA benchmark across provided or built-in samples."""
    if samples is None:
        samples = _BUILTIN_SAMPLES
        logger.info("Using %d built-in HotpotQA samples", len(samples))

    samples = samples[:max_samples]
    t0 = time.time()
    results: list[HotpotResult] = []

    for sample in samples:
        result = evaluate_sample(sample, answer_fn)
        results.append(result)

    elapsed = time.time() - t0

    # aggregate by difficulty
    by_diff: dict[str, dict[str, float]] = {}
    for diff in ("easy", "medium", "hard"):
        diff_results = [r for r, s in zip(results, samples) if s["difficulty"] == diff]
        if diff_results:
            by_diff[diff] = {
                "exact_match_rate": round(sum(1 for r in diff_results if r["exact_match"]) / len(diff_results), 4),
                "mean_f1": round(sum(r["f1"] for r in diff_results) / len(diff_results), 4),
                "count": len(diff_results),
            }

    latencies = [r["latency_ms"] for r in results]

    return HotpotReport(
        total_samples=len(results),
        exact_match_rate=round(sum(1 for r in results if r["exact_match"]) / max(1, len(results)), 4),
        mean_f1=round(sum(r["f1"] for r in results) / max(1, len(results)), 4),
        latency_p50_ms=round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
        latency_p95_ms=round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        by_difficulty=by_diff,
        individual_results=results,
        elapsed_secs=round(elapsed, 2),
    )


def load_hotpotqa_dataset(max_samples: int = 200) -> list[HotpotSample]:
    """Load HotpotQA from HuggingFace. Falls back to built-in samples."""
    try:
        from datasets import load_dataset
        ds = load_dataset("hotpot_qa", "fullwiki", split="validation", trust_remote_code=True)
        samples: list[HotpotSample] = []
        for row in ds.select(range(min(max_samples, len(ds)))):
            sf = row.get("supporting_facts", {})
            sf_dict: dict = sf if isinstance(sf, dict) else {}
            titles = list(set(sf_dict.get("title", [])))
            level = str(row.get("level", "medium"))
            samples.append(HotpotSample(
                question=row["question"],
                answer=row["answer"],
                supporting_titles=titles,
                difficulty=level,
            ))
        logger.info("Loaded %d HotpotQA samples from HuggingFace", len(samples))
        return samples
    except (ImportError, OSError, ValueError, ConnectionError) as exc:
        logger.warning("HuggingFace load failed, using built-in samples: %s", exc)
        return list(_BUILTIN_SAMPLES)

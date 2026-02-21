# Author: Bradley R. Kinnard
# ragchecker.py - RAG evaluation using retrieval precision/recall metrics.
# Measures how well the retrieval pipeline surfaces relevant chunks.

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np

logger = logging.getLogger(__name__)


class RagSample(TypedDict):
    """Single RAG evaluation sample."""
    query: str
    relevant_doc_ids: list[str]
    answer: str


class RagCheckResult(TypedDict):
    """Result of evaluating a single RAG sample."""
    query: str
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    retrieved_ids: list[str]
    relevant_ids: list[str]
    latency_ms: float


@dataclass
class RagCheckerConfig:
    """Settings for RAG evaluation run."""
    top_k: int = 10
    relevance_threshold: float = 0.5
    max_samples: int = 200


@dataclass
class RagCheckerReport:
    """Aggregated RAG evaluation metrics across all samples."""
    total_samples: int = 0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    individual_results: list[RagCheckResult] = field(default_factory=list)
    elapsed_secs: float = 0.0


# Built-in synthetic samples for offline testing (no HuggingFace download)
_BUILTIN_SAMPLES: list[RagSample] = [
    {
        "query": "What is CSWR in RLFusion?",
        "relevant_doc_ids": ["cswr_doc", "retriever_doc"],
        "answer": "Chunk Stability Weighted Retrieval filters RAG chunks by neighbor similarity, question fit, and drift.",
    },
    {
        "query": "How does the CQL policy select fusion weights?",
        "relevant_doc_ids": ["rl_policy_doc", "fusion_doc"],
        "answer": "CQL uses offline conservative Q-learning to produce 4-dim action logits for RAG/CAG/Graph/Web weights.",
    },
    {
        "query": "What is the observation space of FusionEnv?",
        "relevant_doc_ids": ["fusion_env_doc", "rl_policy_doc"],
        "answer": "Box(shape=(396,), dtype=float32) with 384 embedding dims plus 12 engineered features.",
    },
    {
        "query": "How does OOD detection work in RLFO?",
        "relevant_doc_ids": ["utils_doc", "safety_doc"],
        "answer": "Mahalanobis distance with Ledoit-Wolf shrinkage covariance, fitted on in-distribution embeddings.",
    },
    {
        "query": "What retrieval paths does RLFO support?",
        "relevant_doc_ids": ["retriever_doc", "fusion_doc", "architecture_doc"],
        "answer": "Four paths: RAG (FAISS), CAG (exact cache), Graph (entity ontology), Web (Tavily search).",
    },
]


def _compute_precision_recall(retrieved: list[str], relevant: set[str], k: int) -> tuple[float, float, float]:
    """Compute precision@k, recall@k, and F1@k."""
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)

    precision = hits / max(1, len(top_k))
    recall = hits / max(1, len(relevant))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def evaluate_sample(
    sample: RagSample,
    retrieval_fn: object,
    config: RagCheckerConfig,
) -> RagCheckResult:
    """Evaluate a single RAG sample against the retrieval pipeline.

    retrieval_fn should accept (query: str, top_k: int) and return
    a list of dicts with at least a 'doc_id' or 'text' field.
    """
    relevant_set = set(sample["relevant_doc_ids"])
    t0 = time.perf_counter()

    # Call retrieval function if it's actually callable
    if callable(retrieval_fn):
        try:
            results = retrieval_fn(sample["query"], config.top_k)
            retrieved_ids = [
                r.get("doc_id", r.get("id", r.get("text", "")[:32]))
                for r in results
            ]
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as exc:
            logger.warning("Retrieval failed for query '%s': %s", sample["query"][:40], exc)
            retrieved_ids = []
    else:
        retrieved_ids = []

    latency_ms = (time.perf_counter() - t0) * 1000
    precision, recall, f1 = _compute_precision_recall(retrieved_ids, relevant_set, config.top_k)

    return RagCheckResult(
        query=sample["query"],
        precision_at_k=precision,
        recall_at_k=recall,
        f1_at_k=f1,
        retrieved_ids=retrieved_ids,
        relevant_ids=list(relevant_set),
        latency_ms=round(latency_ms, 2),
    )


def run_ragchecker(
    samples: list[RagSample] | None = None,
    retrieval_fn: object = None,
    config: RagCheckerConfig | None = None,
) -> RagCheckerReport:
    """Run RAG evaluation across a set of samples.

    Falls back to built-in synthetic samples when none provided.
    """
    if config is None:
        config = RagCheckerConfig()

    if samples is None:
        samples = _BUILTIN_SAMPLES
        logger.info("Using %d built-in RAG evaluation samples", len(samples))

    samples = samples[:config.max_samples]
    t0 = time.time()
    results: list[RagCheckResult] = []

    for sample in samples:
        result = evaluate_sample(sample, retrieval_fn, config)
        results.append(result)

    elapsed = time.time() - t0
    latencies = [r["latency_ms"] for r in results]

    return RagCheckerReport(
        total_samples=len(results),
        mean_precision=round(float(np.mean([r["precision_at_k"] for r in results])), 4) if results else 0.0,
        mean_recall=round(float(np.mean([r["recall_at_k"] for r in results])), 4) if results else 0.0,
        mean_f1=round(float(np.mean([r["f1_at_k"] for r in results])), 4) if results else 0.0,
        latency_p50_ms=round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
        latency_p95_ms=round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        individual_results=results,
        elapsed_secs=round(elapsed, 2),
    )


def load_ragchecker_dataset(max_samples: int = 200) -> list[RagSample]:
    """Attempt to load RAGCHECKER-style dataset from HuggingFace.

    Falls back to built-in samples if datasets library isn't available
    or the download fails.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("hotpot_qa", "fullwiki", split="validation", trust_remote_code=True)
        samples: list[RagSample] = []
        for row in ds.select(range(min(max_samples, len(ds)))):
            # hotpot_qa has 'question', 'answer', 'supporting_facts'
            sf = row.get("supporting_facts", {})
            sf_dict: dict = sf if isinstance(sf, dict) else {}
            doc_ids = list(set(sf_dict.get("title", [])))
            samples.append(RagSample(
                query=row["question"],
                relevant_doc_ids=doc_ids,
                answer=row["answer"],
            ))
        logger.info("Loaded %d samples from HuggingFace hotpot_qa for RAG evaluation", len(samples))
        return samples
    except (ImportError, OSError, ValueError, ConnectionError) as exc:
        logger.warning("Could not load HuggingFace dataset, using built-in samples: %s", exc)
        return list(_BUILTIN_SAMPLES)

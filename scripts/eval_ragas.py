# Author: Bradley R. Kinnard
# eval_ragas.py - RAGAs-style evaluation of the 2-path retrieval pipeline
# across three fusion strategies (uniform, heuristic, learned CQL).
#
# Produces a results table with three metrics per strategy:
#   * context_relevance  -- does the fused context contain the ground-truth
#                           span? (substring + embedding cosine, max of the two)
#   * answer_relevance   -- embedding cosine between generated answer and the
#                           reference answer
#   * faithfulness       -- proportion of reference-answer tokens that also
#                           appear in the fused context (n-gram overlap)
#
# These are simplified, local proxies for the RAGAs metrics in
# Es et al. 2023 (arXiv 2309.15217); they do not require an LLM judge and
# they are reproducible from a single command. Honest, small, and local.
#
# Usage:
#     python3 scripts/eval_ragas.py
#     python3 scripts/eval_ragas.py --strategies uniform,heuristic
#     python3 scripts/eval_ragas.py --dry-run        # skip the LLM, score
#                                                    # the fused context as
#                                                    # the "answer"
#     python3 scripts/eval_ragas.py --limit 10
#
# Output: a markdown table on stdout and a JSON dump at
# `tests/results/ragas_<timestamp>.json`.

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("eval_ragas")

QA_PATH = PROJECT_ROOT / "data" / "benchmarks" / "ragas_qa.jsonl"
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
STRATEGIES = ("uniform", "heuristic", "learned")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class QARow:
    qid: str
    query: str
    reference_answer: str
    context_span: str


@dataclass
class PerQueryResult:
    qid: str
    strategy: str
    weights: tuple[float, float]
    context_relevance: float
    answer_relevance: float
    faithfulness: float
    fused_context_chars: int
    answer_chars: int


def _load_qa(limit: int | None) -> list[QARow]:
    rows: list[QARow] = []
    with QA_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(QARow(
                qid=str(obj["id"]),
                query=str(obj["query"]),
                reference_answer=str(obj["reference_answer"]),
                context_span=str(obj["context_span"]),
            ))
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"No Q&A rows loaded from {QA_PATH}")
    return rows


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text or "")}


def _context_relevance(fused: str, span: str, embed_fn) -> float:
    """Max of substring presence (1.0/0.0) and embedding cosine vs the span."""
    if not fused.strip() or not span.strip():
        return 0.0
    substring = 1.0 if span.lower() in fused.lower() else 0.0
    a = embed_fn(span)
    b = embed_fn(fused[:2000])
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return max(substring, max(0.0, cos))


def _answer_relevance(answer: str, reference: str, embed_fn) -> float:
    if not answer.strip() or not reference.strip():
        return 0.0
    a = embed_fn(answer)
    b = embed_fn(reference)
    return max(0.0, float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))


def _faithfulness(answer: str, fused: str) -> float:
    """Fraction of reference-answer content tokens that also appear in the
    fused context. A weak proxy for the RAGAs faithfulness metric — it is
    biased toward extractive answers but is reproducible without a judge."""
    ans_toks = _tokens(answer)
    if not ans_toks:
        return 0.0
    ctx_toks = _tokens(fused)
    overlap = ans_toks & ctx_toks
    return len(overlap) / len(ans_toks)


def _weights_for_strategy(strategy: str, policy, query: str) -> np.ndarray:
    from backend.agents.fusion_agent import _heuristic_weights, compute_rl_weights

    if strategy == "uniform":
        return np.array([0.5, 0.5], dtype=np.float32)
    if strategy == "heuristic":
        return _heuristic_weights(query).astype(np.float32)
    if strategy == "learned":
        if policy is None:
            return np.array([0.5, 0.5], dtype=np.float32)
        return compute_rl_weights(query, policy).astype(np.float32)
    raise ValueError(f"Unknown strategy: {strategy}")


def _try_load_policy():
    """Return a loaded CQL wrapper or None if no checkpoint exists."""
    from backend.config import PROJECT_ROOT as _PR

    path = Path(_PR) / "models" / "rl_policy_cql.d3"
    if not path.exists():
        return None
    try:
        import torch
        from backend.main import CQLPolicyWrapper
        state = torch.load(str(path), weights_only=True, map_location="cpu")
        return CQLPolicyWrapper(state["policy"])
    except Exception as exc:
        logger.warning("Could not load CQL policy: %s", exc)
        return None


def _generate(query: str, fused: str, dry_run: bool) -> str:
    if dry_run:
        return fused
    try:
        from backend.core.model_router import get_engine
        engine = get_engine()
        return engine.generate(
            messages=[
                {"role": "system", "content": "Use the context to answer. Be concise."},
                {"role": "user", "content": f"Context:\n{fused}\n\nQuestion: {query}"},
            ],
            temperature=0.0, num_predict=200, num_ctx=4096,
        )
    except Exception as exc:
        logger.warning("Generator failed (%s); falling back to context as answer", exc)
        return fused


def run_eval(strategies: Iterable[str], dry_run: bool, limit: int | None) -> dict[str, Any]:
    from backend.agents.fusion_agent import build_fusion_context
    from backend.core.retrievers import retrieve
    from backend.core.utils import embed_text

    rows = _load_qa(limit)
    logger.info("Loaded %d Q&A rows from %s", len(rows), QA_PATH)

    policy = None
    if "learned" in strategies:
        policy = _try_load_policy()
        logger.info("Learned policy: %s", "loaded" if policy is not None else "missing (skipped)")

    per_query: list[PerQueryResult] = []
    for row in rows:
        retrieval = retrieve(row.query)
        for strategy in strategies:
            weights = _weights_for_strategy(strategy, policy, row.query)
            fused = build_fusion_context(retrieval, weights, query=row.query)
            answer = _generate(row.query, fused, dry_run)
            per_query.append(PerQueryResult(
                qid=row.qid,
                strategy=strategy,
                weights=(float(weights[0]), float(weights[1])),
                context_relevance=_context_relevance(fused, row.context_span, embed_text),
                answer_relevance=_answer_relevance(answer, row.reference_answer, embed_text),
                faithfulness=_faithfulness(row.reference_answer, fused),
                fused_context_chars=len(fused),
                answer_chars=len(answer),
            ))
            logger.info(
                "[%s | %s] w=(%.2f,%.2f) ctx=%.2f ans=%.2f faith=%.2f",
                row.qid, strategy, *weights, per_query[-1].context_relevance,
                per_query[-1].answer_relevance, per_query[-1].faithfulness,
            )

    # aggregate per strategy
    summary: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        items = [p for p in per_query if p.strategy == strategy]
        if not items:
            continue
        summary[strategy] = {
            "n": len(items),
            "context_relevance": float(np.mean([p.context_relevance for p in items])),
            "answer_relevance": float(np.mean([p.answer_relevance for p in items])),
            "faithfulness": float(np.mean([p.faithfulness for p in items])),
        }

    return {
        "qa_path": str(QA_PATH),
        "n_queries": len(rows),
        "strategies": list(strategies),
        "summary": summary,
        "per_query": [p.__dict__ for p in per_query],
        "dry_run": dry_run,
    }


def _format_table(summary: dict[str, dict[str, float]]) -> str:
    lines = [
        "| strategy   |   n | context_relevance | answer_relevance | faithfulness |",
        "|------------|----:|------------------:|-----------------:|-------------:|",
    ]
    for strat in STRATEGIES:
        if strat not in summary:
            continue
        s = summary[strat]
        lines.append(
            f"| {strat:<10s} | {int(s['n']):>3d} | "
            f"{s['context_relevance']:>17.3f} | "
            f"{s['answer_relevance']:>16.3f} | "
            f"{s['faithfulness']:>12.3f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(STRATEGIES),
        help=f"Comma-separated subset of {STRATEGIES}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the LLM and use fused context as answer (CI-friendly).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N queries from the dataset.",
    )
    args = parser.parse_args()

    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    for s in strategies:
        if s not in STRATEGIES:
            print(f"Unknown strategy: {s}", file=sys.stderr)
            return 1

    if args.dry_run:
        os.environ["RLFUSION_ENV_DRY_RUN"] = "1"

    result = run_eval(strategies, args.dry_run, args.limit)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ragas_{int(time.time())}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote raw results to %s", out_path)

    table = _format_table(result["summary"])
    print()
    print(f"Dataset: {QA_PATH}  ({result['n_queries']} queries)")
    print(f"Dry-run: {result['dry_run']}")
    print()
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Author: Bradley R. Kinnard
# runner.py - Unified benchmark runner for RLFO evaluation.
# Orchestrates RAGChecker, HotpotQA, and TruthfulQA with standardized output.

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from tests.benchmarks.hotpotqa import HotpotReport, run_hotpotqa, HotpotSample
from tests.benchmarks.ragchecker import RagCheckerConfig, RagCheckerReport, RagSample, run_ragchecker
from tests.benchmarks.truthfulqa import TruthfulReport, TruthfulSample, run_truthfulqa

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path("tests/results")


class BenchmarkResult(TypedDict):
    """Standardized output from a single benchmark run."""
    benchmark: str
    timestamp: str
    passed: bool
    summary: dict[str, float]
    elapsed_secs: float


@dataclass
class RunnerConfig:
    """Configuration for the unified benchmark runner."""
    max_samples: int = 200
    rag_top_k: int = 10
    regression_threshold: float = 0.05
    results_dir: str = "tests/results"
    phase_tag: str = ""


@dataclass
class MasterReport:
    """Aggregated report across all benchmark suites."""
    timestamp: str = ""
    total_benchmarks: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    overall_passed: bool = False
    results: dict[str, BenchmarkResult] = field(default_factory=dict)
    regressions: list[str] = field(default_factory=list)
    elapsed_secs: float = 0.0


class BenchmarkRunner:
    """Coordinates all benchmark suites and checks for regressions."""

    def __init__(
        self,
        config: RunnerConfig | None = None,
        retrieval_fn: object = None,
        answer_fn: object = None,
    ) -> None:
        self._config = config or RunnerConfig()
        self._retrieval_fn = retrieval_fn
        self._answer_fn = answer_fn
        self._results_dir = Path(self._config.results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def run_ragchecker(
        self,
        samples: list[RagSample] | None = None,
    ) -> BenchmarkResult:
        """Run RAGChecker benchmark and return standardized result."""
        rag_config = RagCheckerConfig(
            top_k=self._config.rag_top_k,
            max_samples=self._config.max_samples,
        )
        report = run_ragchecker(
            samples=samples,
            retrieval_fn=self._retrieval_fn,
            config=rag_config,
        )
        passed = report.mean_f1 >= 0.0  # any non-error run passes baseline
        result = BenchmarkResult(
            benchmark="ragchecker",
            timestamp=datetime.now().isoformat(),
            passed=passed,
            summary={
                "mean_precision": report.mean_precision,
                "mean_recall": report.mean_recall,
                "mean_f1": report.mean_f1,
                "latency_p50_ms": report.latency_p50_ms,
                "latency_p95_ms": report.latency_p95_ms,
                "total_samples": report.total_samples,
            },
            elapsed_secs=report.elapsed_secs,
        )
        self._save("ragchecker", result)
        return result

    def run_hotpotqa(
        self,
        samples: list[HotpotSample] | None = None,
    ) -> BenchmarkResult:
        """Run HotpotQA multi-hop benchmark and return standardized result."""
        report = run_hotpotqa(
            samples=samples,
            answer_fn=self._answer_fn,
            max_samples=self._config.max_samples,
        )
        passed = report.mean_f1 >= 0.0
        result = BenchmarkResult(
            benchmark="hotpotqa",
            timestamp=datetime.now().isoformat(),
            passed=passed,
            summary={
                "exact_match_rate": report.exact_match_rate,
                "mean_f1": report.mean_f1,
                "latency_p50_ms": report.latency_p50_ms,
                "latency_p95_ms": report.latency_p95_ms,
                "total_samples": report.total_samples,
            },
            elapsed_secs=report.elapsed_secs,
        )
        self._save("hotpotqa", result)
        return result

    def run_truthfulqa(
        self,
        samples: list[TruthfulSample] | None = None,
    ) -> BenchmarkResult:
        """Run TruthfulQA hallucination benchmark and return standardized result."""
        report = run_truthfulqa(
            samples=samples,
            answer_fn=self._answer_fn,
            max_samples=self._config.max_samples,
        )
        passed = report.truthful_rate >= 0.5
        result = BenchmarkResult(
            benchmark="truthfulqa",
            timestamp=datetime.now().isoformat(),
            passed=passed,
            summary={
                "truthful_rate": report.truthful_rate,
                "informative_rate": report.informative_rate,
                "truthful_informative_rate": report.truthful_informative_rate,
                "latency_p50_ms": report.latency_p50_ms,
                "latency_p95_ms": report.latency_p95_ms,
                "total_samples": report.total_samples,
            },
            elapsed_secs=report.elapsed_secs,
        )
        self._save("truthfulqa", result)
        return result

    def run_all(self) -> MasterReport:
        """Execute all benchmark suites and produce a master report."""
        t0 = time.time()
        results: dict[str, BenchmarkResult] = {}

        for name, run_fn in [
            ("ragchecker", self.run_ragchecker),
            ("hotpotqa", self.run_hotpotqa),
            ("truthfulqa", self.run_truthfulqa),
        ]:
            try:
                logger.info("Running benchmark: %s", name)
                results[name] = run_fn()
            except (RuntimeError, ValueError, TypeError, OSError, ImportError) as exc:
                logger.error("Benchmark %s failed: %s", name, exc)
                results[name] = BenchmarkResult(
                    benchmark=name,
                    timestamp=datetime.now().isoformat(),
                    passed=False,
                    summary={"error": str(exc)},
                    elapsed_secs=0.0,
                )

        elapsed = time.time() - t0
        passed_count = sum(1 for r in results.values() if r["passed"])
        total = len(results)
        regressions = self._check_regressions(results)

        report = MasterReport(
            timestamp=datetime.now().isoformat(),
            total_benchmarks=total,
            passed=passed_count,
            failed=total - passed_count,
            pass_rate=round(passed_count / max(1, total), 3),
            overall_passed=passed_count == total and len(regressions) == 0,
            results=results,
            regressions=regressions,
            elapsed_secs=round(elapsed, 2),
        )

        self._save_master(report)
        return report

    def _check_regressions(
        self,
        current: dict[str, BenchmarkResult],
    ) -> list[str]:
        """Compare current results against trailing 7-day average. Flag >5% drops."""
        regressions: list[str] = []
        threshold = self._config.regression_threshold

        for bench_name, bench_result in current.items():
            history = self._load_history(bench_name, days=7)
            if len(history) < 2:
                continue  # not enough data for regression check

            # compare key summary metrics
            for metric_key, current_val in bench_result["summary"].items():
                if not isinstance(current_val, (int, float)):
                    continue
                historical_vals = [
                    h.get("summary", {}).get(metric_key)
                    for h in history
                    if isinstance(h.get("summary", {}).get(metric_key), (int, float))
                ]
                if not historical_vals:
                    continue

                avg_hist = sum(historical_vals) / len(historical_vals)
                if avg_hist == 0:
                    continue

                # drop detection (for metrics where higher = better)
                if metric_key.endswith("_ms"):
                    # latency: higher is worse, so check for increase
                    if current_val > avg_hist * (1 + threshold):
                        msg = f"{bench_name}.{metric_key}: {current_val:.3f} > trailing avg {avg_hist:.3f} (+{threshold*100:.0f}%)"
                        regressions.append(msg)
                        logger.warning("Regression detected: %s", msg)
                else:
                    # accuracy-like: lower is worse
                    if current_val < avg_hist * (1 - threshold):
                        msg = f"{bench_name}.{metric_key}: {current_val:.3f} < trailing avg {avg_hist:.3f} (-{threshold*100:.0f}%)"
                        regressions.append(msg)
                        logger.warning("Regression detected: %s", msg)

        return regressions

    def _load_history(self, benchmark_name: str, days: int = 7) -> list[dict]:
        """Load benchmark results from the last N days."""
        import glob
        pattern = str(self._results_dir / f"{benchmark_name}_*.json")
        files = sorted(glob.glob(pattern), reverse=True)
        results: list[dict] = []

        cutoff = time.time() - (days * 86400)
        for fpath in files[:50]:  # cap to avoid scanning thousands
            try:
                stat = os.stat(fpath)
                if stat.st_mtime < cutoff:
                    break
                with open(fpath) as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
        return results

    def _save(self, name: str, result: BenchmarkResult) -> str:
        """Persist a single benchmark result as timestamped JSON."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{self._config.phase_tag}" if self._config.phase_tag else ""
        path = self._results_dir / f"{name}{tag}_{ts}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Saved benchmark result: %s", path)
        return str(path)

    def _save_master(self, report: MasterReport) -> str:
        """Persist the master benchmark report."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{self._config.phase_tag}" if self._config.phase_tag else ""
        path = self._results_dir / f"benchmark_master{tag}_{ts}.json"
        with open(path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        logger.info("Saved master benchmark report: %s", path)
        return str(path)

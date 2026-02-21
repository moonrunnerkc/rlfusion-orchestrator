# Author: Bradley R. Kinnard
# test_phase9_benchmarks.py - tests for Phase 9 automated evaluation & monitoring
# No external services required. Tests run against built-in samples.

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RLFUSION_DEVICE", "cpu")
os.environ.setdefault("RLFUSION_FORCE_CPU", "true")


# ---------------------------------------------------------------------------
# ragchecker.py
# ---------------------------------------------------------------------------

class TestRagcheckerPrecisionRecall:
    """Validate precision@k, recall@k, and F1 computation."""

    def test_perfect_retrieval(self):
        from tests.benchmarks.ragchecker import _compute_precision_recall
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        p, r, f1 = _compute_precision_recall(retrieved, relevant, k=3)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_partial_retrieval(self):
        from tests.benchmarks.ragchecker import _compute_precision_recall
        retrieved = ["a", "b", "x", "y"]
        relevant = {"a", "b", "c"}
        p, r, f1 = _compute_precision_recall(retrieved, relevant, k=4)
        assert p == 0.5  # 2 / 4
        assert abs(r - 0.6667) < 0.01  # 2 / 3
        assert f1 > 0

    def test_no_hits(self):
        from tests.benchmarks.ragchecker import _compute_precision_recall
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        p, r, f1 = _compute_precision_recall(retrieved, relevant, k=3)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_k_truncation(self):
        from tests.benchmarks.ragchecker import _compute_precision_recall
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "d", "e"}
        # only look at top-2
        p, r, f1 = _compute_precision_recall(retrieved, relevant, k=2)
        assert p == 0.5  # 1/2 (only "a" in top-2)
        assert abs(r - 0.3333) < 0.01  # 1/3


class TestRagcheckerEvaluate:
    """Evaluate single samples with mock retrieval function."""

    def test_builtin_samples_exist(self):
        from tests.benchmarks.ragchecker import _BUILTIN_SAMPLES
        assert len(_BUILTIN_SAMPLES) >= 5
        for s in _BUILTIN_SAMPLES:
            assert "query" in s
            assert "relevant_doc_ids" in s
            assert "answer" in s

    def test_evaluate_with_mock_retriever(self):
        from tests.benchmarks.ragchecker import evaluate_sample, RagCheckerConfig

        sample = {
            "query": "test query",
            "relevant_doc_ids": ["doc_a", "doc_b"],
            "answer": "test answer",
        }

        def mock_retriever(query: str, top_k: int) -> list:
            return [{"doc_id": "doc_a"}, {"doc_id": "doc_c"}]

        config = RagCheckerConfig(top_k=5)
        result = evaluate_sample(sample, mock_retriever, config)
        assert result["query"] == "test query"
        assert result["precision_at_k"] == 0.5
        assert result["recall_at_k"] == 0.5
        assert result["latency_ms"] >= 0

    def test_evaluate_with_no_retriever(self):
        from tests.benchmarks.ragchecker import evaluate_sample, RagCheckerConfig

        sample = {
            "query": "test",
            "relevant_doc_ids": ["x"],
            "answer": "y",
        }
        config = RagCheckerConfig()
        result = evaluate_sample(sample, None, config)
        assert result["precision_at_k"] == 0.0
        assert result["recall_at_k"] == 0.0


class TestRagcheckerRun:
    """End-to-end ragchecker benchmark run."""

    def test_run_with_builtins(self):
        from tests.benchmarks.ragchecker import run_ragchecker

        report = run_ragchecker()
        assert report.total_samples == 5
        assert report.elapsed_secs >= 0
        assert 0 <= report.mean_precision <= 1.0
        assert 0 <= report.mean_recall <= 1.0

    def test_run_with_custom_samples(self):
        from tests.benchmarks.ragchecker import run_ragchecker, RagCheckerConfig

        samples = [{
            "query": "q",
            "relevant_doc_ids": ["d1"],
            "answer": "a",
        }]
        config = RagCheckerConfig(max_samples=1)
        report = run_ragchecker(samples=samples, config=config)
        assert report.total_samples == 1


# ---------------------------------------------------------------------------
# hotpotqa.py
# ---------------------------------------------------------------------------

class TestHotpotNormalization:
    """Answer normalization for fair comparison."""

    def test_normalize_strips_articles(self):
        from tests.benchmarks.hotpotqa import _normalize_answer
        assert "cat" in _normalize_answer("The cat")

    def test_normalize_strips_punctuation(self):
        from tests.benchmarks.hotpotqa import _normalize_answer
        result = _normalize_answer("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_normalize_lowercases(self):
        from tests.benchmarks.hotpotqa import _normalize_answer
        assert _normalize_answer("UPPER") == "upper"


class TestHotpotExactMatch:
    """Exact match scoring."""

    def test_identical_strings(self):
        from tests.benchmarks.hotpotqa import exact_match
        assert exact_match("hello world", "hello world")

    def test_case_insensitive(self):
        from tests.benchmarks.hotpotqa import exact_match
        assert exact_match("Hello World", "hello world")

    def test_different_answers(self):
        from tests.benchmarks.hotpotqa import exact_match
        assert not exact_match("apples", "oranges")


class TestHotpotF1:
    """Token-level F1 scoring."""

    def test_perfect_match(self):
        from tests.benchmarks.hotpotqa import token_f1
        assert token_f1("the quick brown fox", "the quick brown fox") == 1.0

    def test_partial_overlap(self):
        from tests.benchmarks.hotpotqa import token_f1
        f1 = token_f1("quick brown fox", "slow brown fox")
        assert 0 < f1 < 1.0

    def test_no_overlap(self):
        from tests.benchmarks.hotpotqa import token_f1
        assert token_f1("alpha beta", "gamma delta") == 0.0

    def test_empty_gold(self):
        from tests.benchmarks.hotpotqa import token_f1
        assert token_f1("", "") == 1.0


class TestHotpotRun:
    """End-to-end HotpotQA benchmark run."""

    def test_run_with_builtins(self):
        from tests.benchmarks.hotpotqa import run_hotpotqa

        report = run_hotpotqa()
        assert report.total_samples == 8
        assert report.elapsed_secs >= 0
        assert 0 <= report.exact_match_rate <= 1.0
        assert 0 <= report.mean_f1 <= 1.0

    def test_run_with_mock_answer_fn(self):
        from tests.benchmarks.hotpotqa import run_hotpotqa

        def echo_fn(q: str) -> str:
            return q  # just echo the question

        report = run_hotpotqa(answer_fn=echo_fn, max_samples=3)
        assert report.total_samples == 3
        assert report.elapsed_secs >= 0

    def test_by_difficulty_keys(self):
        from tests.benchmarks.hotpotqa import run_hotpotqa

        report = run_hotpotqa()
        # built-in samples span easy/medium/hard
        assert "easy" in report.by_difficulty or "medium" in report.by_difficulty


# ---------------------------------------------------------------------------
# truthfulqa.py
# ---------------------------------------------------------------------------

class TestTruthfulChecks:
    """Truthfulness and informativeness scoring."""

    def test_detects_correct_refusal(self):
        from tests.benchmarks.truthfulqa import _check_truthfulness

        sample = {
            "question": "Is X true?",
            "correct_answers": ["no", "that is not correct"],
            "incorrect_answers": ["yes X is true"],
            "category": "false_premise",
        }
        is_truthful, score = _check_truthfulness(
            "That is not correct. X is actually false.", sample
        )
        assert is_truthful
        assert score > 0

    def test_detects_hallucination(self):
        from tests.benchmarks.truthfulqa import _check_truthfulness

        sample = {
            "question": "Does RLFO use quantum?",
            "correct_answers": ["no"],
            "incorrect_answers": ["quantum entanglement synchronizes"],
            "category": "false_premise",
        }
        is_truthful, score = _check_truthfulness(
            "Yes, quantum entanglement synchronizes the retrieval paths.", sample
        )
        # contains incorrect answer fragment -> should flag
        assert score < 0.5

    def test_informativeness_check(self):
        from tests.benchmarks.truthfulqa import _check_informativeness
        assert _check_informativeness("This is a detailed and informative response about the topic.")
        assert not _check_informativeness("I don't know")
        assert not _check_informativeness("")


class TestTruthfulRun:
    """End-to-end TruthfulQA benchmark run."""

    def test_run_with_builtins(self):
        from tests.benchmarks.truthfulqa import run_truthfulqa

        report = run_truthfulqa()
        assert report.total_samples == 8
        assert report.elapsed_secs >= 0
        assert 0 <= report.truthful_rate <= 1.0

    def test_builtin_samples_structure(self):
        from tests.benchmarks.truthfulqa import _BUILTIN_SAMPLES
        for s in _BUILTIN_SAMPLES:
            assert "question" in s
            assert "correct_answers" in s
            assert "incorrect_answers" in s
            assert "category" in s
            assert len(s["correct_answers"]) >= 1
            assert len(s["incorrect_answers"]) >= 1


# ---------------------------------------------------------------------------
# runner.py
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    """Unified benchmark runner orchestration."""

    def test_runner_config_defaults(self):
        from tests.benchmarks.runner import RunnerConfig
        config = RunnerConfig()
        assert config.max_samples == 200
        assert config.rag_top_k == 10
        assert config.regression_threshold == 0.05

    def test_runner_instantiation(self):
        from tests.benchmarks.runner import BenchmarkRunner, RunnerConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunnerConfig(results_dir=tmpdir, max_samples=3)
            runner = BenchmarkRunner(config=config)
            assert runner._results_dir.exists()

    def test_run_ragchecker(self):
        from tests.benchmarks.runner import BenchmarkRunner, RunnerConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunnerConfig(results_dir=tmpdir, max_samples=2)
            runner = BenchmarkRunner(config=config)
            result = runner.run_ragchecker()
            assert result["benchmark"] == "ragchecker"
            assert result["passed"] is True  # baseline criterion
            assert "mean_f1" in result["summary"]

    def test_run_hotpotqa(self):
        from tests.benchmarks.runner import BenchmarkRunner, RunnerConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunnerConfig(results_dir=tmpdir, max_samples=2)
            runner = BenchmarkRunner(config=config)
            result = runner.run_hotpotqa()
            assert result["benchmark"] == "hotpotqa"
            assert "exact_match_rate" in result["summary"]

    def test_run_truthfulqa(self):
        from tests.benchmarks.runner import BenchmarkRunner, RunnerConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunnerConfig(results_dir=tmpdir, max_samples=2)
            runner = BenchmarkRunner(config=config)
            result = runner.run_truthfulqa()
            assert result["benchmark"] == "truthfulqa"
            assert "truthful_rate" in result["summary"]

    def test_run_all_produces_master_report(self):
        from tests.benchmarks.runner import BenchmarkRunner, RunnerConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunnerConfig(results_dir=tmpdir, max_samples=2, phase_tag="test")
            runner = BenchmarkRunner(config=config)
            report = runner.run_all()
            assert report.total_benchmarks == 3
            assert report.passed + report.failed == 3
            assert 0 <= report.pass_rate <= 1.0
            # check that files were written
            result_files = list(Path(tmpdir).glob("*.json"))
            assert len(result_files) >= 4  # 3 individual + 1 master

    def test_regression_detection(self):
        from tests.benchmarks.runner import BenchmarkRunner, RunnerConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunnerConfig(results_dir=tmpdir, max_samples=2, regression_threshold=0.05)
            runner = BenchmarkRunner(config=config)
            # no history -> no regressions
            current = {
                "test_bench": {
                    "benchmark": "test_bench",
                    "timestamp": "",
                    "passed": True,
                    "summary": {"mean_f1": 0.5},
                    "elapsed_secs": 0.1,
                }
            }
            regressions = runner._check_regressions(current)
            assert regressions == []

    def test_save_and_load(self):
        from tests.benchmarks.runner import BenchmarkRunner, RunnerConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunnerConfig(results_dir=tmpdir)
            runner = BenchmarkRunner(config=config)
            result = {
                "benchmark": "test",
                "timestamp": "2026-01-01",
                "passed": True,
                "summary": {"score": 0.9},
                "elapsed_secs": 1.0,
            }
            path = runner._save("test", result)
            assert Path(path).exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["benchmark"] == "test"


# ---------------------------------------------------------------------------
# Prometheus metrics (backend/main.py instrumentation)
# ---------------------------------------------------------------------------

class TestPrometheusMetrics:
    """Verify Prometheus metric definitions import cleanly."""

    def test_metrics_import(self):
        from prometheus_client import Counter, Gauge, Histogram
        # just verify they're importable
        assert Counter is not None
        assert Gauge is not None
        assert Histogram is not None

    def test_generate_latest(self):
        from prometheus_client import generate_latest
        output = generate_latest()
        assert isinstance(output, bytes)
        assert len(output) > 0


# ---------------------------------------------------------------------------
# Config: monitoring section
# ---------------------------------------------------------------------------

class TestMonitoringConfig:
    """Phase 9 monitoring config keys exist with safe defaults."""

    def test_monitoring_config_present(self):
        from backend.config import cfg
        monitoring = cfg.get("monitoring", {})
        assert monitoring.get("prometheus_enabled") is True
        assert monitoring.get("correlation_id_header") == "X-Correlation-ID"
        assert monitoring.get("regression_threshold") == 0.05
        assert monitoring.get("benchmark_results_dir") == "tests/results"
        assert monitoring.get("metrics_history_max") == 10000


# ---------------------------------------------------------------------------
# Grafana dashboard structure
# ---------------------------------------------------------------------------

class TestGrafanaDashboard:
    """Validate the Grafana dashboard JSON is well-formed."""

    def test_dashboard_loads(self):
        dashboard_path = PROJECT_ROOT / "scripts" / "grafana" / "dashboard.json"
        assert dashboard_path.exists(), "Grafana dashboard JSON missing"
        with open(dashboard_path) as f:
            data = json.load(f)
        assert data["title"] == "RLFusion Orchestrator"
        assert len(data["panels"]) >= 6
        assert data["uid"] == "rlfusion-orchestrator-v1"

    def test_dashboard_panels_have_targets(self):
        dashboard_path = PROJECT_ROOT / "scripts" / "grafana" / "dashboard.json"
        with open(dashboard_path) as f:
            data = json.load(f)
        for panel in data["panels"]:
            assert "title" in panel
            assert "targets" in panel
            assert len(panel["targets"]) >= 1
            for target in panel["targets"]:
                assert "expr" in target

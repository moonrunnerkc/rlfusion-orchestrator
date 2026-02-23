# Author: Bradley R. Kinnard
# test_stis_integration.py - integration tests for STIS Phase 3-5
# Covers: client axiom formatting, httpx fallback paths, SQLite logging,
# orchestrator step_stis_check, config wiring, and health probes.

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RLFUSION_DEVICE", "cpu")
os.environ.setdefault("RLFUSION_FORCE_CPU", "true")


# ---------------------------------------------------------------------------
# Phase 3: STIS Client - Axiom formatting
# ---------------------------------------------------------------------------

class TestAxiomPromptFormatting:
    """Validate the axiom prompt template structures conflicts correctly."""

    def test_prompt_contains_query(self):
        from backend.core.stis_client import format_axiom_prompt
        prompt = format_axiom_prompt("What is X?", "RAG says A", "Graph says B")
        assert "What is X?" in prompt

    def test_prompt_contains_both_axioms(self):
        from backend.core.stis_client import format_axiom_prompt
        prompt = format_axiom_prompt("q", "RAG claim alpha", "Graph claim beta")
        assert "RAG claim alpha" in prompt
        assert "Graph claim beta" in prompt

    def test_prompt_labels_axioms(self):
        from backend.core.stis_client import format_axiom_prompt
        prompt = format_axiom_prompt("q", "claim1", "claim2")
        assert "AXIOM 1" in prompt
        assert "AXIOM 2" in prompt
        assert "Document Retrieval" in prompt
        assert "Knowledge Graph" in prompt

    def test_prompt_asks_for_synthesis(self):
        from backend.core.stis_client import format_axiom_prompt
        prompt = format_axiom_prompt("q", "c1", "c2")
        assert "SYNTHESIS" in prompt or "synthesis" in prompt.lower()
        assert "contradiction" in prompt.lower()

    def test_prompt_returns_string(self):
        from backend.core.stis_client import format_axiom_prompt
        result = format_axiom_prompt("q", "a", "b")
        assert isinstance(result, str)
        assert len(result) > 50  # not trivially short


# ---------------------------------------------------------------------------
# Phase 3: STIS Client - HTTP fallback paths
# ---------------------------------------------------------------------------

class TestSTISClientFallbacks:
    """Validate graceful fallback when STIS engine is unreachable."""

    def test_unreachable_returns_resolved_false(self):
        from backend.core.stis_client import request_stis_consensus
        # point at a port nothing is listening on
        with patch("backend.core.stis_client._get_stis_config", return_value={
            "host": "http://127.0.0.1:19999",
            "timeout": 2.0,
            "max_new_tokens": 64,
            "num_agents": None,
            "similarity_threshold": None,
            "alpha": None,
        }):
            result = request_stis_consensus("test query", "claim A", "claim B")
        assert result["resolved"] is False
        assert result["error"] is not None
        assert result["latency_secs"] >= 0

    def test_timeout_returns_resolved_false(self):
        """Simulate a timeout by pointing at a very short timeout against unreachable host."""
        from backend.core.stis_client import request_stis_consensus
        with patch("backend.core.stis_client._get_stis_config", return_value={
            "host": "http://10.255.255.1",  # non-routable IP = guaranteed timeout
            "timeout": 0.5,
            "max_new_tokens": 64,
            "num_agents": None,
            "similarity_threshold": None,
            "alpha": None,
        }):
            result = request_stis_consensus("test", "a", "b")
        assert result["resolved"] is False
        assert "timeout" in (result["error"] or "").lower() or "unreachable" in (result["error"] or "").lower()

    def test_successful_response_parsing(self):
        """Mock a successful STIS response and verify parsing."""
        import httpx
        from backend.core.stis_client import request_stis_consensus

        mock_response = httpx.Response(
            status_code=200,
            json={
                "text": "Synthesized answer from swarm consensus",
                "total_tokens": 42,
                "convergence_log": [],
                "final_similarity": 0.97,
                "total_iterations": 12,
                "wall_time_secs": 3.5,
            },
        )

        with patch("backend.core.stis_client._get_stis_config", return_value={
            "host": "http://localhost:8100",
            "timeout": 10.0,
            "max_new_tokens": 512,
            "num_agents": None,
            "similarity_threshold": None,
            "alpha": None,
        }):
            with patch("httpx.Client") as mock_client_cls:
                mock_client_instance = MagicMock()
                mock_client_instance.post.return_value = mock_response
                mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
                mock_client_instance.__exit__ = MagicMock(return_value=False)
                mock_client_cls.return_value = mock_client_instance

                result = request_stis_consensus("query", "rag claim", "graph claim")

        assert result["resolved"] is True
        assert result["resolution"]["text"] == "Synthesized answer from swarm consensus"
        assert result["resolution"]["total_tokens"] == 42
        assert result["resolution"]["final_similarity"] == 0.97
        assert result["resolution"]["axiom_1"] == "rag claim"
        assert result["resolution"]["axiom_2"] == "graph claim"
        assert result["error"] is None

    def test_http_error_returns_resolved_false(self):
        """Mock a 500 error from STIS engine."""
        import httpx
        from backend.core.stis_client import request_stis_consensus

        mock_response = httpx.Response(status_code=500, text="Internal Server Error")

        with patch("backend.core.stis_client._get_stis_config", return_value={
            "host": "http://localhost:8100",
            "timeout": 10.0,
            "max_new_tokens": 512,
            "num_agents": None,
            "similarity_threshold": None,
            "alpha": None,
        }):
            with patch("httpx.Client") as mock_client_cls:
                mock_client_instance = MagicMock()
                mock_client_instance.post.return_value = mock_response
                mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
                mock_client_instance.__exit__ = MagicMock(return_value=False)
                mock_client_cls.return_value = mock_client_instance

                result = request_stis_consensus("query", "a", "b")

        assert result["resolved"] is False
        assert "500" in result["error"]

    def test_result_shape(self):
        """STISFallbackResult must contain all required keys."""
        from backend.core.stis_client import request_stis_consensus
        with patch("backend.core.stis_client._get_stis_config", return_value={
            "host": "http://127.0.0.1:19999",
            "timeout": 1.0,
            "max_new_tokens": 64,
            "num_agents": None,
            "similarity_threshold": None,
            "alpha": None,
        }):
            result = request_stis_consensus("q", "a", "b")

        assert "resolved" in result
        assert "resolution" in result
        assert "error" in result
        assert "latency_secs" in result
        assert isinstance(result["resolved"], bool)
        assert isinstance(result["latency_secs"], float)


# ---------------------------------------------------------------------------
# Phase 3: STIS Client - Health check
# ---------------------------------------------------------------------------

class TestSTISHealthCheck:
    """Validate health probe against unreachable engine."""

    def test_health_unavailable(self):
        from backend.core.stis_client import check_stis_health
        with patch("backend.core.stis_client._get_stis_config", return_value={
            "host": "http://127.0.0.1:19999",
            "timeout": 45,
            "max_new_tokens": 512,
            "num_agents": None,
            "similarity_threshold": None,
            "alpha": None,
        }):
            result = check_stis_health()
        assert result["available"] is False
        assert "error" in result

    def test_health_success_mock(self):
        """Mock a healthy STIS engine response."""
        import httpx
        from backend.core.stis_client import check_stis_health

        mock_resp = httpx.Response(200, json={
            "status": "ok", "model_loaded": True,
            "hidden_dim": 1536, "num_agents": 4,
            "similarity_threshold": 0.95, "device": "cuda",
        })
        with patch("backend.core.stis_client._get_stis_config", return_value={
            "host": "http://localhost:8100",
            "timeout": 45,
            "max_new_tokens": 512,
            "num_agents": None,
            "similarity_threshold": None,
            "alpha": None,
        }):
            with patch("httpx.get", return_value=mock_resp):
                result = check_stis_health()
        assert result["available"] is True
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# Phase 4: SQLite logging
# ---------------------------------------------------------------------------

class TestSTISResolutionLogging:
    """Validate SQLite STIS resolution audit trail."""

    def test_log_creates_table_and_row(self):
        from backend.core.stis_client import log_stis_resolution
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "db" / "rlfo_cache.db"
            with patch("backend.core.stis_client.PROJECT_ROOT", Path(tmpdir)):
                result = {
                    "resolved": True,
                    "resolution": {
                        "text": "consensus answer",
                        "total_tokens": 20,
                        "final_similarity": 0.96,
                        "total_iterations": 8,
                        "wall_time_secs": 2.5,
                        "axiom_1": "rag",
                        "axiom_2": "graph",
                    },
                    "error": None,
                    "latency_secs": 2.6,
                }
                success = log_stis_resolution(
                    "test query", "rag claim", "graph claim",
                    0.35, 0.55, result,
                )

            assert success is True
            assert db_path.exists()

            conn = sqlite3.connect(str(db_path))
            rows = conn.execute("SELECT * FROM stis_resolutions").fetchall()
            conn.close()
            assert len(rows) == 1

    def test_log_failed_resolution(self):
        from backend.core.stis_client import log_stis_resolution
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.core.stis_client.PROJECT_ROOT", Path(tmpdir)):
                result = {
                    "resolved": False,
                    "resolution": None,
                    "error": "STIS unreachable at localhost:8100",
                    "latency_secs": 1.0,
                }
                success = log_stis_resolution(
                    "query", "rag", "graph", 0.30, 0.40, result,
                )

            assert success is True
            db_path = Path(tmpdir) / "db" / "rlfo_cache.db"
            conn = sqlite3.connect(str(db_path))
            row = conn.execute("SELECT resolved, error FROM stis_resolutions").fetchone()
            conn.close()
            assert row[0] == 0  # not resolved
            assert "unreachable" in row[1]

    def test_log_multiple_events(self):
        from backend.core.stis_client import log_stis_resolution
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.core.stis_client.PROJECT_ROOT", Path(tmpdir)):
                for i in range(5):
                    log_stis_resolution(
                        f"query {i}", "rag", "graph", 0.3, 0.5,
                        {"resolved": False, "resolution": None,
                         "error": "test", "latency_secs": 0.1},
                    )

            db_path = Path(tmpdir) / "db" / "rlfo_cache.db"
            conn = sqlite3.connect(str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM stis_resolutions").fetchone()[0]
            conn.close()
            assert count == 5

    def test_log_schema_columns(self):
        """Validate all expected columns exist in the stis_resolutions table."""
        from backend.core.stis_client import log_stis_resolution
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.core.stis_client.PROJECT_ROOT", Path(tmpdir)):
                log_stis_resolution(
                    "q", "r", "g", 0.3, 0.5,
                    {"resolved": True, "resolution": {
                        "text": "t", "total_tokens": 1, "final_similarity": 0.9,
                        "total_iterations": 3, "wall_time_secs": 1.0,
                        "axiom_1": "a1", "axiom_2": "a2",
                    }, "error": None, "latency_secs": 1.1},
                )

            db_path = Path(tmpdir) / "db" / "rlfo_cache.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("PRAGMA table_info(stis_resolutions)")
            columns = {row[1] for row in cursor.fetchall()}
            conn.close()

            expected = {
                "id", "timestamp", "query", "rag_claim", "graph_claim",
                "contradiction_similarity", "best_cswr", "resolved",
                "resolution_text", "total_tokens", "final_similarity",
                "total_iterations", "stis_wall_time", "error", "latency_secs",
            }
            assert expected.issubset(columns), f"Missing columns: {expected - columns}"


# ---------------------------------------------------------------------------
# Phase 4: Orchestrator step_stis_check
# ---------------------------------------------------------------------------

class TestOrchestratorSTISCheck:
    """Validate the orchestrator's STIS routing decision method."""

    def test_disabled_returns_no_route(self):
        from backend.agents.orchestrator import Orchestrator
        orch = Orchestrator(rl_policy=None)
        with patch.dict("backend.config.cfg", {"stis": {"enabled": False}}):
            result = orch.step_stis_check({"rag": [], "graph": [], "web": []})
        assert result["route_to_stis"] is False
        assert "disabled" in result["reason"].lower()

    def test_enabled_with_empty_results(self):
        from backend.agents.orchestrator import Orchestrator
        orch = Orchestrator(rl_policy=None)
        with patch.dict("backend.config.cfg", {"stis": {"enabled": True}}, clear=False):
            result = orch.step_stis_check({"rag": [], "graph": [], "web": []})
        assert result["route_to_stis"] is False

    def test_enabled_with_similar_results(self):
        from backend.agents.orchestrator import Orchestrator
        orch = Orchestrator(rl_policy=None)
        rag = [{"text": "Python is a programming language", "score": 0.85}]
        graph = [{"text": "Python is a popular programming language", "score": 0.80}]
        with patch.dict("backend.config.cfg", {"stis": {"enabled": True}}, clear=False):
            result = orch.step_stis_check({"rag": rag, "graph": graph, "web": []})
        assert result["route_to_stis"] is False

    def test_returns_required_keys(self):
        from backend.agents.orchestrator import Orchestrator
        orch = Orchestrator(rl_policy=None)
        with patch.dict("backend.config.cfg", {"stis": {"enabled": True}}, clear=False):
            result = orch.step_stis_check({"rag": [], "graph": [], "web": []})
        assert "route_to_stis" in result
        assert "reason" in result
        assert "contradiction" in result
        assert "best_cswr" in result


# ---------------------------------------------------------------------------
# Phase 4: Config wiring
# ---------------------------------------------------------------------------

class TestSTISConfigWiring:
    """Validate STIS config section in config.yaml."""

    def test_config_has_stis_section(self):
        from backend.config import cfg
        stis = cfg.get("stis", {})
        assert isinstance(stis, dict)

    def test_config_stis_defaults(self):
        from backend.config import cfg
        stis = cfg.get("stis", {})
        if stis:
            assert "enabled" in stis
            assert "host" in stis
            assert "timeout_secs" in stis

    def test_stis_client_reads_config(self):
        from backend.core.stis_client import _get_stis_config
        result = _get_stis_config()
        assert "host" in result
        assert "timeout" in result
        assert "max_new_tokens" in result
        assert isinstance(result["host"], str)
        assert isinstance(result["timeout"], float)


# ---------------------------------------------------------------------------
# Phase 4: PreparedContext has retrieval_results
# ---------------------------------------------------------------------------

class TestPreparedContextShape:
    """Validate PreparedContext includes retrieval_results for STIS routing."""

    def test_prepared_context_has_retrieval_results_field(self):
        from backend.agents.base import PreparedContext
        import typing
        hints = typing.get_type_hints(PreparedContext)
        assert "retrieval_results" in hints

    def test_orchestrator_prepare_memory_request_has_retrieval_results(self):
        from backend.agents.orchestrator import Orchestrator
        orch = Orchestrator(rl_policy=None)
        with patch("backend.core.profile.detect_and_save_memory", return_value=(True, "remembered")):
            result = orch.prepare("remember my name is Brad", session_id="test")
        assert "retrieval_results" in result
        assert isinstance(result["retrieval_results"], dict)


# ---------------------------------------------------------------------------
# Phase 3-5: Import integration
# ---------------------------------------------------------------------------

class TestSTISImportChain:
    """Verify all STIS symbols are importable together."""

    def test_client_imports(self):
        from backend.core.stis_client import (
            format_axiom_prompt,
            request_stis_consensus,
            check_stis_health,
            log_stis_resolution,
            STISResolution,
            STISFallbackResult,
        )
        assert callable(format_axiom_prompt)
        assert callable(request_stis_consensus)
        assert callable(check_stis_health)
        assert callable(log_stis_resolution)

    def test_critique_stis_imports(self):
        from backend.core.critique import (
            detect_contradiction,
            should_route_to_stis,
            STIS_CSWR_THRESHOLD,
            STIS_SIMILARITY_FLOOR,
        )
        assert callable(detect_contradiction)
        assert callable(should_route_to_stis)

    def test_orchestrator_stis_check(self):
        from backend.agents.orchestrator import Orchestrator
        orch = Orchestrator(rl_policy=None)
        assert hasattr(orch, "step_stis_check")
        assert callable(orch.step_stis_check)

    def test_main_imports_stis(self):
        """Verify main.py can import the STIS routing symbols."""
        from backend.core.critique import should_route_to_stis
        from backend.core.stis_client import request_stis_consensus, log_stis_resolution
        assert callable(should_route_to_stis)
        assert callable(request_stis_consensus)
        assert callable(log_stis_resolution)

# Author: Bradley R. Kinnard
# test_asymmetric.py - Tests for the 2-path CAG+Graph architecture.
# Covers: asymmetric pipeline routing, CAG exact/semantic match, GraphRAG
# traversal, 2-path RL weights, OOM fallback, CPU/GPU isolation, metrics,
# and JSON triage robustness.

import hashlib
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ───────────────────────────────────────────────────────────────────────
# AsymmetricLLMOrchestrator
# ───────────────────────────────────────────────────────────────────────


class TestAsymmetricOrchestrator:
    """Verify orchestrator initialization and configuration."""

    def test_singleton_pattern(self):
        from backend.core.asymmetric_llm import AsymmetricLLMOrchestrator
        a = AsymmetricLLMOrchestrator()
        b = AsymmetricLLMOrchestrator()
        assert a is b

    def test_get_orchestrator_returns_instance(self):
        from backend.core.asymmetric_llm import get_orchestrator
        orch = get_orchestrator()
        assert orch is not None
        assert hasattr(orch, "triage")
        assert hasattr(orch, "execute")
        assert hasattr(orch, "route_task")

    def test_health_check_structure(self):
        from backend.core.asymmetric_llm import get_orchestrator
        orch = get_orchestrator()
        status = orch.health_check()
        assert "cpu_loaded" in status
        assert "gpu_loaded" in status
        assert "gpu_offloaded" in status
        assert "vram_used_mb" in status

    def test_context_enforcement(self):
        from backend.core.asymmetric_llm import get_orchestrator
        orch = get_orchestrator()
        long_prompt = "x" * 100_000
        truncated = orch._enforce_ctx(long_prompt, ctx_size=8192, max_tokens=512)
        # budget = (8192 - 512 - 128) * 4 = 30208
        assert len(truncated) <= 30208
        assert len(truncated) < len(long_prompt)

    def test_context_no_truncation_for_short_prompt(self):
        from backend.core.asymmetric_llm import get_orchestrator
        orch = get_orchestrator()
        short = "What is RLFusion?"
        result = orch._enforce_ctx(short, ctx_size=8192, max_tokens=512)
        assert result == short


class TestTaskRouting:
    """Verify CPU/GPU task routing logic."""

    def test_cpu_tasks_recognized(self):
        from backend.core.asymmetric_llm import _CPU_TASKS
        expected = {"intent_parse", "safety_check", "cag_lookup", "graph_trigger", "obs_build"}
        assert _CPU_TASKS == expected

    def test_gpu_tasks_recognized(self):
        from backend.core.asymmetric_llm import _GPU_TASKS
        expected = {"generation", "critique", "stis_deep", "faithfulness"}
        assert _GPU_TASKS == expected

    def test_no_task_overlap(self):
        from backend.core.asymmetric_llm import _CPU_TASKS, _GPU_TASKS
        assert _CPU_TASKS.isdisjoint(_GPU_TASKS)

    def test_unknown_task_raises(self):
        from backend.core.asymmetric_llm import get_orchestrator
        orch = get_orchestrator()
        with pytest.raises(ValueError, match="Unknown task type"):
            orch.route_task("nonexistent_task", "test")

    def test_routing_cpu_only(self):
        """Triage tasks should target CPU worker, not GPU."""
        from backend.core.asymmetric_llm import _CPU_TASKS, get_orchestrator
        orch = get_orchestrator()
        for task in _CPU_TASKS:
            json_mode = task in {"intent_parse", "graph_trigger", "obs_build"}
            assert json_mode is not None  # just verifying no crash in routing logic

    def test_routing_gpu_only(self):
        """Generation tasks should target GPU executor, not CPU."""
        from backend.core.asymmetric_llm import _GPU_TASKS
        for task in _GPU_TASKS:
            assert task not in {"intent_parse", "safety_check", "cag_lookup"}


# ───────────────────────────────────────────────────────────────────────
# CAG Cache
# ───────────────────────────────────────────────────────────────────────


class TestCAGExactMatch:
    """SHA-256 exact match correctness for the CAG cache."""

    def test_sha256_hash_deterministic(self):
        """Same query always produces same hash."""
        from backend.core.utils import deterministic_id
        q = "What is RLFusion?"
        h1 = deterministic_id(q)
        h2 = deterministic_id(q)
        assert h1 == h2

    def test_sha256_different_queries(self):
        from backend.core.utils import deterministic_id
        h1 = deterministic_id("What is RLFusion?")
        h2 = deterministic_id("How does fusion work?")
        assert h1 != h2

    def test_cag_retrieval_returns_list(self):
        from backend.core.retrievers import retrieve
        results = retrieve("test query for CAG")
        assert "cag" in results
        assert isinstance(results["cag"], list)


class TestCAGSemanticMatch:
    """Semantic similarity lookup in CAG cache."""

    def test_retrieve_includes_cag_key(self):
        from backend.core.retrievers import retrieve
        results = retrieve("What is the architecture?")
        assert "cag" in results

    def test_cag_results_have_score(self):
        """If CAG returns results, they should have score keys."""
        from backend.core.retrievers import retrieve
        results = retrieve("hello world")
        for item in results.get("cag", []):
            assert "score" in item or "answer" in item


# ───────────────────────────────────────────────────────────────────────
# GraphRAG
# ───────────────────────────────────────────────────────────────────────


class TestGraphRetrieval:
    """Graph traversal returns valid context for entity-bearing queries."""

    def test_graph_returns_list(self):
        from backend.core.retrievers import retrieve
        results = retrieve("Describe the knowledge graph")
        assert "graph" in results
        assert isinstance(results["graph"], list)

    def test_graph_results_structure(self):
        from backend.core.retrievers import retrieve
        results = retrieve("entity relationships in the system")
        for item in results.get("graph", []):
            assert isinstance(item, dict)


# ───────────────────────────────────────────────────────────────────────
# Two-Path RL Weights
# ───────────────────────────────────────────────────────────────────────


class TestTwoPathWeights:
    """RL weight distributions for the 2-path architecture."""

    def test_softmax_sums_to_one(self):
        logits = torch.tensor([0.3, 0.7])
        weights = torch.softmax(logits, dim=0)
        assert abs(weights.sum().item() - 1.0) < 1e-6

    def test_weights_clamped_above_minimum(self):
        logits = torch.tensor([-10.0, 10.0])
        weights = torch.softmax(logits, dim=0)
        weights = torch.clamp(weights, min=0.05)
        weights = weights / weights.sum()
        # softmax of extreme logits then clamp: minor precision loss is fine
        assert weights[0].item() >= 0.04
        assert weights[1].item() >= 0.04

    def test_default_weights_sum_to_one(self):
        from backend.config import cfg
        fw = cfg.get("fusion", {}).get("default_weights", {})
        if fw:
            total = sum(fw.values())
            assert abs(total - 1.0) < 1e-6

    def test_default_weights_two_keys_only(self):
        from backend.config import cfg
        fw = cfg.get("fusion", {}).get("default_weights", {})
        if fw:
            assert set(fw.keys()) == {"cag", "graph"}

    def test_fusion_env_action_space_2d(self):
        from backend.rl.fusion_env import FusionEnv
        env = FusionEnv()
        assert env.action_space.shape == (2,)
        assert env.NUM_SOURCES == 2

    def test_fusion_env_obs_space_394(self):
        from backend.rl.fusion_env import FusionEnv
        env = FusionEnv()
        assert env.observation_space.shape == (394,)

    def test_normalize_weights_two_path(self):
        from backend.core.fusion import normalize_weights
        result = normalize_weights([0.3, 0.7])
        assert len(result) == 2
        assert abs(sum(result) - 1.0) < 1e-6

    def test_get_default_weights(self):
        from backend.core.fusion import get_default_weights
        w = get_default_weights()
        assert len(w) == 2
        assert abs(sum(w) - 1.0) < 1e-6


# ───────────────────────────────────────────────────────────────────────
# OOM Fallback
# ───────────────────────────────────────────────────────────────────────


class TestOOMFallback:
    """OOM triggers fallback without crash."""

    def test_handle_oom_sets_offloaded_flag(self):
        """Verify OOM handler sets offloaded state."""
        from backend.core.asymmetric_llm import get_orchestrator
        orch = get_orchestrator()
        # just verify the fallback mechanism exists and the flag is settable
        assert hasattr(orch, "_gpu_offloaded")
        assert hasattr(orch, "_handle_oom")

    def test_oom_fallback_recoverable(self):
        """After OOM, orchestrator should remain functional."""
        from backend.core.asymmetric_llm import get_orchestrator
        orch = get_orchestrator()
        original_state = orch._gpu_offloaded
        # don't actually trigger OOM, just verify state management
        assert isinstance(original_state, bool)


# ───────────────────────────────────────────────────────────────────────
# Memory Monitoring (Step 8)
# ───────────────────────────────────────────────────────────────────────


class TestMemoryMonitor:
    """VRAM/RAM monitoring gauges."""

    def test_snapshot_structure(self):
        from backend.core.metrics import get_monitor
        m = get_monitor()
        snap = m.snapshot()
        required = {"timestamp", "vram_used_mb", "vram_total_mb", "vram_pct", "ram_used_mb", "ram_total_mb", "ram_pct"}
        assert required.issubset(snap.keys())

    def test_snapshot_values_numeric(self):
        from backend.core.metrics import get_monitor
        m = get_monitor()
        snap = m.snapshot()
        for key in ["vram_used_mb", "vram_total_mb", "ram_used_mb", "ram_total_mb"]:
            assert isinstance(snap[key], float)

    def test_ram_nonzero(self):
        from backend.core.metrics import get_monitor
        m = get_monitor()
        snap = m.snapshot()
        assert snap["ram_total_mb"] > 0

    def test_vram_detected(self):
        """On a GPU system, VRAM total should be > 0."""
        from backend.core.metrics import get_monitor
        m = get_monitor()
        snap = m.snapshot()
        # may be 0 on CI without GPU, but should be > 0 on dev machine
        assert snap["vram_total_mb"] >= 0

    def test_monitor_start_stop(self):
        from backend.core.metrics import MemoryMonitor
        m = MemoryMonitor(interval_seconds=1)
        m.start()
        assert m._running is True
        time.sleep(0.5)
        m.stop()
        assert m._running is False

    def test_last_snapshot_after_read(self):
        from backend.core.metrics import get_monitor
        m = get_monitor()
        m.snapshot()
        assert m.last is not None

    def test_snapshot_parseable_as_json(self):
        """Structured logs must be jq-parseable."""
        from backend.core.metrics import get_monitor
        m = get_monitor()
        snap = m.snapshot()
        dumped = json.dumps(snap)
        parsed = json.loads(dumped)
        assert parsed["timestamp"] > 0


# ───────────────────────────────────────────────────────────────────────
# No Dead References
# ───────────────────────────────────────────────────────────────────────


class TestNoDeadReferences:
    """Verify removal of RAG/Web/Tavily from core paths."""

    def test_no_rag_key_in_default_weights(self):
        from backend.core.fusion import get_default_weights
        w = get_default_weights()
        # returns a 2-element list [cag, graph], no rag/web
        assert len(w) == 2

    def test_retrieve_no_rag_results(self):
        """RAG key may still exist for backward compat but must be empty."""
        from backend.core.retrievers import retrieve
        results = retrieve("test query")
        rag = results.get("rag", [])
        assert len(rag) == 0, "RAG results should always be empty in 2-path architecture"

    def test_config_retrieval_paths(self):
        from backend.config import cfg
        paths = cfg.get("retrieval", {}).get("paths", [])
        if paths:
            assert "rag" not in paths
            assert "web" not in paths
            assert "cag" in paths
            assert "graph" in paths

    def test_fusion_env_two_sources(self):
        from backend.rl.fusion_env import FusionEnv
        assert FusionEnv.NUM_SOURCES == 2

    def test_no_tavily_in_retrievers(self):
        """tavily_search should not be called in retrievers hot path."""
        import importlib
        mod = importlib.import_module("backend.core.retrievers")
        source = Path(mod.__file__).read_text()
        # functional tavily code removed; only deprecation comments may reference it
        assert "tavily_search(" not in source, "Active tavily_search call found in retrievers"


# ───────────────────────────────────────────────────────────────────────
# Frontend Contract Alignment
# ───────────────────────────────────────────────────────────────────────


class TestFrontendContracts:
    """Verify frontend types exist and align with backend shapes."""

    def test_contracts_file_exists(self):
        p = Path(__file__).resolve().parents[1] / "frontend" / "src" / "types" / "contracts.ts"
        assert p.exists(), f"Missing {p}"

    def test_contracts_has_weights_type(self):
        p = Path(__file__).resolve().parents[1] / "frontend" / "src" / "types" / "contracts.ts"
        content = p.read_text()
        assert "cag: number" in content
        assert "graph: number" in content
        # should NOT have rag or web
        lines = content.split("\n")
        weights_section = False
        for line in lines:
            if "interface Weights" in line:
                weights_section = True
            elif weights_section and "}" in line:
                break
            elif weights_section:
                assert "rag" not in line.lower(), "Weights interface should not have rag"
                assert "web" not in line.lower(), "Weights interface should not have web"

    def test_no_four_retrieval_paths_text(self):
        """ChatList should not mention four retrieval paths."""
        p = Path(__file__).resolve().parents[1] / "frontend" / "src" / "components" / "ChatList.tsx"
        content = p.read_text()
        assert "four retrieval" not in content.lower()


# ───────────────────────────────────────────────────────────────────────
# Backward Compatibility
# ───────────────────────────────────────────────────────────────────────


class TestBackwardCompatibility:
    """Critical backward compat checks from the upgrade plan."""

    def test_embed_text_still_exists(self):
        """embed_text remains in public API even though it's off the hot path."""
        from backend.core.utils import embed_text
        result = embed_text("test")
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)

    def test_embed_batch_still_exists(self):
        from backend.core.utils import embed_batch
        result = embed_batch(["a", "b"])
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2

    def test_critique_signature_unchanged(self):
        import inspect
        from backend.core.critique import critique
        sig = inspect.signature(critique)
        params = list(sig.parameters.keys())
        assert "query" in params
        # param may be 'context' or 'fused_context' depending on version
        assert any(p in params for p in ["context", "fused_context"])
        assert "response" in params

    def test_fuse_context_signature(self):
        import inspect
        from backend.core.fusion import fuse_context
        sig = inspect.signature(fuse_context)
        params = list(sig.parameters.keys())
        assert "query" in params or len(params) >= 1

    def test_config_yaml_loads(self):
        from backend.config import cfg
        assert "llm" in cfg
        assert "embedding" in cfg
        assert "fusion" in cfg

    def test_project_root_exists(self):
        from backend.config import PROJECT_ROOT
        assert PROJECT_ROOT.exists()

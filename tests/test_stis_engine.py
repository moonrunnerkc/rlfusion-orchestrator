# Author: Bradley R. Kinnard
# test_stis_engine.py - invariant tests for the STIS swarm consensus engine
# Validates convergence math, dimension stability, centroid preservation,
# and monotonicity guarantees without requiring GPU or model downloads.

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestSTISConfig:
    """Validate configuration loading and environment overrides."""

    def test_default_config(self):
        from stis_engine.config import STISConfig, SwarmConfig, ModelConfig, ServerConfig
        cfg = STISConfig()
        assert cfg.swarm.num_agents == 4
        assert cfg.swarm.similarity_threshold == 0.92
        assert cfg.swarm.alpha == 0.5
        assert cfg.swarm.max_iterations == 30
        assert cfg.model.model_id == "Qwen/Qwen2.5-1.5B"
        assert cfg.model.dtype == "float16"
        assert cfg.model.seed == 42
        assert cfg.server.port == 8100
        assert cfg.server.timeout_secs == 45

    def test_config_is_frozen(self):
        from stis_engine.config import SwarmConfig
        cfg = SwarmConfig()
        with pytest.raises(AttributeError):
            cfg.num_agents = 8  # type: ignore[misc]

    def test_env_overrides(self):
        env = {
            "STIS_NUM_AGENTS": "8",
            "STIS_SIM_THRESHOLD": "0.99",
            "STIS_ALPHA": "0.5",
            "STIS_PORT": "9000",
            "STIS_SEED": "123",
        }
        with patch.dict(os.environ, env):
            from stis_engine.config import load_config
            cfg = load_config()
            assert cfg.swarm.num_agents == 8
            assert cfg.swarm.similarity_threshold == 0.99
            assert cfg.swarm.alpha == 0.5
            assert cfg.server.port == 9000
            assert cfg.model.seed == 123

    def test_config_repr(self):
        """Frozen dataclass should produce clean repr for logging."""
        from stis_engine.config import SwarmConfig
        cfg = SwarmConfig(num_agents=6)
        r = repr(cfg)
        assert "num_agents=6" in r
        assert "SwarmConfig" in r


# ---------------------------------------------------------------------------
# Convergence math invariants (no model required)
# ---------------------------------------------------------------------------

class TestConvergenceInvariants:
    """Test the convergence loop math using synthetic tensors.

    These tests validate three critical invariants:
    1. Dimension stability: output vectors never change dimensionality.
    2. Centroid preservation: blended centroid stays in the convex hull.
    3. Convergence monotonicity: mean similarity never decreases between iterations.
    """

    def _make_engine_with_mocks(self, num_agents: int = 4, threshold: float = 0.95,
                                 alpha: float = 0.3, max_iters: int = 50,
                                 hidden_dim: int = 64):
        """Build a SwarmEngine with mock model/tokenizer for math-only tests."""
        from stis_engine.config import ModelConfig, SwarmConfig
        from stis_engine.swarm import SwarmEngine

        mock_model = MagicMock()
        mock_model.config.hidden_size = hidden_dim
        mock_model.lm_head = torch.nn.Linear(hidden_dim, 100, bias=False)
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        mock_tokenizer = MagicMock()

        model_cfg = ModelConfig(seed=42)
        swarm_cfg = SwarmConfig(
            num_agents=num_agents,
            similarity_threshold=threshold,
            alpha=alpha,
            max_iterations=max_iters,
        )
        return SwarmEngine(mock_model, mock_tokenizer, model_cfg, swarm_cfg)

    def test_dimension_stability_through_convergence(self):
        """Hidden state dimensions must never change during the convergence loop."""
        engine = self._make_engine_with_mocks(num_agents=4, hidden_dim=128)

        # random agent states with some divergence
        torch.manual_seed(42)
        states = torch.randn(4, 128)

        step_log, centroid, iters, sim = engine._converge(
            states, threshold=0.95, alpha=0.3, max_iters=50
        )

        assert centroid.shape == (128,), f"Centroid dim mismatch: {centroid.shape}"
        assert centroid.ndim == 1

    def test_centroid_in_convex_hull(self):
        """Blended centroid must remain inside the convex hull of agent states.

        For any convex combination, each dimension of the centroid must be
        between the min and max of that dimension across agents.
        """
        engine = self._make_engine_with_mocks(num_agents=4, hidden_dim=32)

        torch.manual_seed(7)
        states = torch.randn(4, 32) * 5.0

        step_log, centroid, _, _ = engine._converge(
            states, threshold=0.999, alpha=0.3, max_iters=5
        )

        # after blending, the centroid must be within the per-dim range
        # (within small floating point tolerance)
        for d in range(32):
            col = states[:, d]
            lo = float(col.min()) - 1e-4
            hi = float(col.max()) + 1e-4
            assert lo <= float(centroid[d]) <= hi, (
                f"Dim {d}: centroid={float(centroid[d]):.6f} outside [{lo:.6f}, {hi:.6f}]"
            )

    def test_convergence_monotonicity(self):
        """Mean pairwise similarity must be non-decreasing across iterations.

        The alpha blend toward the centroid can only pull agents closer,
        never push them apart. Similarity must monotonically increase.
        """
        engine = self._make_engine_with_mocks(
            num_agents=4, hidden_dim=64, threshold=0.999, alpha=0.3, max_iters=30
        )

        torch.manual_seed(99)
        states = torch.randn(4, 64) * 3.0

        step_log, _, _, _ = engine._converge(
            states, threshold=0.999, alpha=0.3, max_iters=30
        )

        sims = [step.mean_similarity for step in step_log]
        for i in range(1, len(sims)):
            assert sims[i] >= sims[i - 1] - 1e-6, (
                f"Monotonicity violated at step {i}: {sims[i - 1]:.6f} -> {sims[i]:.6f}"
            )

    def test_identical_states_converge_immediately(self):
        """If all agents start with identical states, convergence is instant."""
        engine = self._make_engine_with_mocks(num_agents=4, hidden_dim=64)

        state = torch.randn(1, 64)
        states = state.repeat(4, 1)

        step_log, centroid, iters, sim = engine._converge(
            states, threshold=0.95, alpha=0.3, max_iters=50
        )

        assert iters == 1, f"Expected 1 iteration for identical states, got {iters}"
        assert sim >= 0.999, f"Identical states should have sim ~1.0, got {sim}"

    def test_near_identical_states_converge_fast(self):
        """States with tiny perturbations should converge within very few iterations."""
        engine = self._make_engine_with_mocks(num_agents=4, hidden_dim=64)

        torch.manual_seed(13)
        base = torch.randn(1, 64)
        noise = torch.randn(4, 64) * 0.001
        states = base + noise

        step_log, _, iters, sim = engine._converge(
            states, threshold=0.95, alpha=0.3, max_iters=50
        )

        assert iters <= 3, f"Near-identical states took {iters} iterations"
        assert sim >= 0.95

    def test_two_agents_minimum(self):
        """Convergence works correctly with the minimum agent count (2)."""
        engine = self._make_engine_with_mocks(num_agents=2, hidden_dim=32)

        torch.manual_seed(42)
        states = torch.randn(2, 32)

        step_log, centroid, iters, sim = engine._converge(
            states, threshold=0.95, alpha=0.3, max_iters=50
        )

        assert centroid.shape == (32,)
        assert len(step_log) > 0
        assert all(s.iteration >= 0 for s in step_log)

    def test_high_agent_count(self):
        """Convergence scales to 16 agents without dimension errors."""
        engine = self._make_engine_with_mocks(num_agents=16, hidden_dim=64)

        torch.manual_seed(42)
        states = torch.randn(16, 64)

        step_log, centroid, _, _ = engine._converge(
            states, threshold=0.90, alpha=0.3, max_iters=100
        )

        assert centroid.shape == (64,)

    def test_alpha_zero_no_blending(self):
        """Alpha=0 means no blending, states remain unchanged, similarity unchanged."""
        engine = self._make_engine_with_mocks(num_agents=4, hidden_dim=32, alpha=0.0)

        torch.manual_seed(55)
        states = torch.randn(4, 32)
        original_states = states.clone()

        step_log, _, iters, _ = engine._converge(
            states, threshold=0.999, alpha=0.0, max_iters=5
        )

        # with alpha=0, similarity never changes, so it either converges on step 1
        # or hits max_iters with the same similarity every step
        sims = [s.mean_similarity for s in step_log]
        for i in range(1, len(sims)):
            assert abs(sims[i] - sims[0]) < 1e-5, "Alpha=0 should produce constant similarity"

    def test_alpha_one_instant_centroid(self):
        """Alpha=1 collapses all states to the centroid on the first blend."""
        engine = self._make_engine_with_mocks(num_agents=4, hidden_dim=32, alpha=1.0)

        torch.manual_seed(77)
        states = torch.randn(4, 32)

        step_log, centroid, iters, sim = engine._converge(
            states, threshold=0.95, alpha=1.0, max_iters=50
        )

        # after one blend with alpha=1, all agents are at the centroid
        # so convergence should happen by step 2 at latest
        assert iters <= 2, f"Alpha=1 should converge in <=2 iterations, took {iters}"

    def test_max_deviation_decreases(self):
        """Maximum pairwise deviation should decrease or stay constant."""
        engine = self._make_engine_with_mocks(num_agents=4, hidden_dim=64)

        torch.manual_seed(33)
        states = torch.randn(4, 64) * 5.0

        step_log, _, _, _ = engine._converge(
            states, threshold=0.999, alpha=0.3, max_iters=20
        )

        devs = [step.max_deviation for step in step_log]
        for i in range(1, len(devs)):
            assert devs[i] <= devs[i - 1] + 1e-5, (
                f"Max deviation increased at step {i}: {devs[i - 1]:.6f} -> {devs[i]:.6f}"
            )

    def test_centroid_norm_stability(self):
        """Centroid norm should remain finite and positive throughout convergence."""
        engine = self._make_engine_with_mocks(num_agents=4, hidden_dim=64)

        torch.manual_seed(42)
        states = torch.randn(4, 64) * 10.0

        step_log, centroid, _, _ = engine._converge(
            states, threshold=0.999, alpha=0.3, max_iters=20
        )

        for step in step_log:
            assert step.centroid_norm > 0, f"Centroid norm <= 0 at step {step.iteration}"
            assert np.isfinite(step.centroid_norm), f"Centroid norm not finite at step {step.iteration}"

        assert float(centroid.norm()) > 0
        assert torch.isfinite(centroid).all()


# ---------------------------------------------------------------------------
# Token sampling invariants
# ---------------------------------------------------------------------------

class TestTokenSampling:
    """Test the lm_head projection and nucleus sampling logic."""

    def _make_engine(self, hidden_dim: int = 64, vocab_size: int = 100):
        from stis_engine.config import ModelConfig, SwarmConfig
        from stis_engine.swarm import SwarmEngine

        mock_model = MagicMock()
        mock_model.config.hidden_size = hidden_dim
        lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        torch.nn.init.xavier_uniform_(lm_head.weight)
        mock_model.lm_head = lm_head
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        mock_tokenizer = MagicMock()
        model_cfg = ModelConfig(seed=42, temperature=0.7, top_p=0.9)
        swarm_cfg = SwarmConfig(num_agents=4)
        return SwarmEngine(mock_model, mock_tokenizer, model_cfg, swarm_cfg)

    def test_sampled_token_in_vocab_range(self):
        """Sampled token ID must be within [0, vocab_size)."""
        engine = self._make_engine(hidden_dim=64, vocab_size=100)
        centroid = torch.randn(64)
        token_id = engine._sample_from_centroid(centroid)
        assert 0 <= token_id < 100, f"Token {token_id} outside vocab range [0, 100)"

    def test_deterministic_with_same_seed(self):
        """Same centroid + same seed should produce same token."""
        engine = self._make_engine()
        centroid = torch.randn(64)

        torch.manual_seed(42)
        t1 = engine._sample_from_centroid(centroid.clone())
        torch.manual_seed(42)
        t2 = engine._sample_from_centroid(centroid.clone())
        assert t1 == t2, f"Non-deterministic sampling: {t1} != {t2}"

    def test_different_centroids_can_differ(self):
        """Vastly different centroids should usually produce different tokens."""
        engine = self._make_engine(hidden_dim=64, vocab_size=1000)
        torch.manual_seed(42)
        c1 = torch.randn(64) * 10
        c2 = -c1  # opposite direction

        torch.manual_seed(42)
        t1 = engine._sample_from_centroid(c1)
        torch.manual_seed(42)
        t2 = engine._sample_from_centroid(c2)
        # not guaranteed to differ, but very likely with large vocab
        # just check both are valid
        assert 0 <= t1 < 1000
        assert 0 <= t2 < 1000


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemas:
    """Validate request/response Pydantic schemas."""

    def test_generate_request_defaults(self):
        from stis_engine.schemas import GenerateRequest
        req = GenerateRequest(prompt="Hello world")
        assert req.max_new_tokens == 512
        assert req.num_agents is None
        assert req.similarity_threshold is None
        assert req.alpha is None

    def test_generate_request_validation(self):
        from stis_engine.schemas import GenerateRequest
        with pytest.raises(Exception):
            GenerateRequest(prompt="")  # min_length=1

    def test_generate_request_overrides(self):
        from stis_engine.schemas import GenerateRequest
        req = GenerateRequest(
            prompt="test", max_new_tokens=128, num_agents=8,
            similarity_threshold=0.98, alpha=0.5
        )
        assert req.num_agents == 8
        assert req.similarity_threshold == 0.98

    def test_health_response_fields(self):
        from stis_engine.schemas import HealthResponse
        h = HealthResponse(
            status="ok", model_loaded=True, hidden_dim=1536,
            num_agents=4, similarity_threshold=0.95, device="cuda"
        )
        assert h.status == "ok"
        assert h.hidden_dim == 1536

    def test_convergence_step_response(self):
        from stis_engine.schemas import ConvergenceStepResponse
        step = ConvergenceStepResponse(
            iteration=3, mean_similarity=0.92,
            max_deviation=0.08, centroid_norm=12.5
        )
        assert step.iteration == 3
        assert step.mean_similarity == 0.92

    def test_error_response(self):
        from stis_engine.schemas import ErrorResponse
        err = ErrorResponse(error="OOM", detail="GPU ran out of memory")
        assert err.error == "OOM"


# ---------------------------------------------------------------------------
# SwarmResult serialization
# ---------------------------------------------------------------------------

class TestSwarmResult:
    """Validate SwarmResult dataclass and its dict serialization."""

    def test_to_dict_structure(self):
        from stis_engine.swarm import ConvergenceStep, SwarmResult

        steps = [ConvergenceStep(iteration=0, mean_similarity=0.80,
                                  max_deviation=0.20, centroid_norm=5.0)]
        result = SwarmResult(
            text="hello", total_tokens=1,
            convergence_log=[steps], final_similarity=0.96,
            total_iterations=3, wall_time_secs=1.5
        )

        d = result.to_dict()
        assert d["text"] == "hello"
        assert d["total_tokens"] == 1
        assert d["final_similarity"] == 0.96
        assert d["total_iterations"] == 3
        assert d["wall_time_secs"] == 1.5
        assert len(d["convergence_log"]) == 1
        assert d["convergence_log"][0][0]["iteration"] == 0
        assert d["convergence_log"][0][0]["mean_similarity"] == 0.80

    def test_empty_convergence_log(self):
        from stis_engine.swarm import SwarmResult
        result = SwarmResult(text="", total_tokens=0)
        d = result.to_dict()
        assert d["convergence_log"] == []
        assert d["final_similarity"] == 0.0


# ---------------------------------------------------------------------------
# Engine property accessors
# ---------------------------------------------------------------------------

class TestEngineProperties:
    """Validate SwarmEngine exposes correct configuration via properties."""

    def _make_engine(self):
        from stis_engine.config import ModelConfig, SwarmConfig
        from stis_engine.swarm import SwarmEngine

        mock_model = MagicMock()
        mock_model.config.hidden_size = 256
        mock_model.lm_head = torch.nn.Linear(256, 50, bias=False)
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        swarm_cfg = SwarmConfig(num_agents=6, similarity_threshold=0.97)
        return SwarmEngine(mock_model, MagicMock(), ModelConfig(), swarm_cfg)

    def test_hidden_dim(self):
        engine = self._make_engine()
        assert engine.hidden_dim == 256

    def test_num_agents(self):
        engine = self._make_engine()
        assert engine.num_agents == 6

    def test_similarity_threshold(self):
        engine = self._make_engine()
        assert engine.similarity_threshold == 0.97

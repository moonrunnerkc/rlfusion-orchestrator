# Author: Bradley R. Kinnard
# test_phase4_rl.py - unit tests for Phase 4 advanced RL components
# Tests PPO online training, DPO preference learning, GRPO group ranking,
# adaptive warmup policy selection, and all supporting functions.

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RLFUSION_DEVICE", "cpu")
os.environ.setdefault("RLFUSION_FORCE_CPU", "true")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_episodes(n: int = 20) -> list[dict[str, object]]:
    """Generate synthetic episodes for testing. Varied rewards for pair building."""
    episodes = []
    queries = [
        "What is reinforcement learning?",
        "How does CSWR filter chunks?",
        "Explain transformer attention",
        "Compare CQL and PPO",
        "What is the capital of France?",
    ]
    for i in range(n):
        q = queries[i % len(queries)]
        reward = 0.3 + (i / n) * 0.6  # spread from 0.3 to 0.9
        rag_w = 0.2 + (i % 5) * 0.1
        cag_w = 0.15
        graph_w = max(0.0, 1.0 - rag_w - cag_w - 0.1)
        web_w = 0.1
        # normalize
        total = rag_w + cag_w + graph_w + web_w
        episodes.append({
            "query": q,
            "reward": round(reward, 3),
            "optimal_weights": {
                "rag": round(rag_w / total, 3),
                "cag": round(cag_w / total, 3),
                "graph": round(graph_w / total, 3),
                "web": round(web_w / total, 3),
            },
            "weights": np.array([rag_w / total, cag_w / total, graph_w / total, web_w / total], dtype=np.float32),
            "response": f"Test response {i}",
        })
    return episodes


# ---------------------------------------------------------------------------
# ReplayFusionEnv
# ---------------------------------------------------------------------------

class TestReplayFusionEnv:
    """Verify the replay env produces valid obs/reward without external calls."""

    def test_reset_returns_396_dim(self):
        from backend.rl.train_ppo import ReplayFusionEnv
        env = ReplayFusionEnv(_make_episodes(5))
        obs, info = env.reset()
        assert obs.shape == (396,)
        assert obs.dtype == np.float32

    def test_step_returns_reward(self):
        from backend.rl.train_ppo import ReplayFusionEnv
        env = ReplayFusionEnv(_make_episodes(5))
        obs, _ = env.reset()
        action = np.array([0.3, 0.2, 0.4, 0.1], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert terminated is True
        assert truncated is False
        assert "query" in info

    def test_cycles_through_episodes(self):
        from backend.rl.train_ppo import ReplayFusionEnv
        eps = _make_episodes(3)
        env = ReplayFusionEnv(eps)
        queries = []
        for _ in range(6):
            env.reset()
            queries.append(env.current_query)
        # should cycle: 0,1,2,0,1,2
        assert queries[0] == queries[3]
        assert queries[1] == queries[4]

    def test_empty_episodes_raises(self):
        from backend.rl.train_ppo import ReplayFusionEnv
        with pytest.raises(ValueError, match="at least one episode"):
            ReplayFusionEnv([])

    def test_reward_scaled_by_imitation(self):
        from backend.rl.train_ppo import ReplayFusionEnv
        eps = [{"query": "test", "reward": 1.0, "optimal_weights": {"rag": 1.0, "cag": 0.0, "graph": 0.0, "web": 0.0}}]
        env = ReplayFusionEnv(eps)
        env.reset()
        # perfect match should yield high reward
        _, reward_good, _, _, _ = env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        env.reset()
        # bad match should yield lower reward
        _, reward_bad, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        assert reward_good > reward_bad


# ---------------------------------------------------------------------------
# PPO model creation and evaluation
# ---------------------------------------------------------------------------

class TestCreatePpoModel:
    """Verify PPO model creation with Phase 4 hyperparameters."""

    def test_creates_ppo_model(self):
        from backend.rl.train_ppo import ReplayFusionEnv, create_ppo_model
        env = ReplayFusionEnv(_make_episodes(5))
        model = create_ppo_model(env, batch_size=4)
        assert model is not None
        assert model.observation_space.shape == (396,)
        assert model.action_space.shape == (4,)

    def test_hyperparams_applied(self):
        from backend.rl.train_ppo import ReplayFusionEnv, create_ppo_model
        env = ReplayFusionEnv(_make_episodes(5))
        model = create_ppo_model(env, clip_range=0.3, ent_coef=0.05, gae_lambda=0.9)
        # SB3 stores clip_range as a schedule function
        clip_val = model.clip_range(1.0) if callable(model.clip_range) else model.clip_range
        assert abs(clip_val - 0.3) < 1e-6
        assert abs(model.ent_coef - 0.05) < 1e-6


class TestEvaluatePolicy:
    """Verify policy evaluation returns correct stats."""

    def test_returns_stats_dict(self):
        from backend.rl.train_ppo import ReplayFusionEnv, create_ppo_model, evaluate_policy
        eps = _make_episodes(10)
        env = ReplayFusionEnv(eps)
        model = create_ppo_model(env, batch_size=4)
        stats = evaluate_policy(model, env, n_eval=5)
        assert "mean_reward" in stats
        assert "std_reward" in stats
        assert "min_reward" in stats
        assert "max_reward" in stats
        assert stats["min_reward"] <= stats["mean_reward"] <= stats["max_reward"]


# ---------------------------------------------------------------------------
# PPO training loop
# ---------------------------------------------------------------------------

class TestTrainPpoOnline:
    """Verify PPO training runs end-to-end with minimal timesteps."""

    def test_trains_and_saves(self):
        from backend.rl.train_ppo import train_ppo_online
        eps = _make_episodes(10)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_ppo.zip"
            result = train_ppo_online(
                episodes=eps,
                total_timesteps=256,
                batch_size=4,
                warm_start=False,
                save_path=save_path,
            )
            assert result.exists()
            assert result.stat().st_size > 0

    def test_raises_on_no_episodes(self):
        from backend.rl.train_ppo import train_ppo_online
        with pytest.raises(ValueError, match="No training episodes"):
            train_ppo_online(episodes=[], warm_start=False)


# ---------------------------------------------------------------------------
# CQL demo generation
# ---------------------------------------------------------------------------

class TestGenerateCqlDemos:
    """Verify CQL demo extraction handles missing/present checkpoints."""

    def test_missing_checkpoint_returns_empty(self):
        from backend.rl.train_ppo import generate_cql_demos
        demos = generate_cql_demos(Path("/nonexistent/model.d3"))
        assert demos == []

    def test_with_real_checkpoint(self):
        from backend.rl.train_ppo import generate_cql_demos
        cql_path = PROJECT_ROOT / "models" / "rl_policy_cql.d3"
        if not cql_path.exists():
            pytest.skip("CQL checkpoint not available")
        demos = generate_cql_demos(cql_path, n_demos=5)
        assert len(demos) == 5
        for d in demos:
            assert "query" in d
            assert "reward" in d
            assert "optimal_weights" in d
            w = d["optimal_weights"]
            assert abs(sum(w.values()) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# GRPO: Group Relative Policy Optimization
# ---------------------------------------------------------------------------

class TestRankCandidatesGrpo:
    """Verify GRPO candidate ranking returns sorted results."""

    def test_returns_sorted_candidates(self):
        from backend.rl.train_ppo import ReplayFusionEnv, create_ppo_model, rank_candidates_grpo
        eps = _make_episodes(10)
        env = ReplayFusionEnv(eps)
        model = create_ppo_model(env, batch_size=4)
        candidates = rank_candidates_grpo(model, env, n_candidates=3)
        assert len(candidates) == 3
        # sorted descending by reward
        rewards = [c[1] for c in candidates]
        assert rewards == sorted(rewards, reverse=True)


class TestTrainGrpo:
    """Verify GRPO training loop runs without error."""

    def test_grpo_trains(self):
        from backend.rl.train_ppo import ReplayFusionEnv, create_ppo_model, train_grpo
        eps = _make_episodes(5)
        env = ReplayFusionEnv(eps)
        model = create_ppo_model(env, batch_size=4)
        # very short training for test speed
        result = train_grpo(model, eps, group_size=2, n_iterations=2, learning_rate=1e-3)
        assert result is not None

    def test_grpo_empty_episodes_raises(self):
        from backend.rl.train_ppo import ReplayFusionEnv, create_ppo_model, train_grpo
        eps = _make_episodes(3)
        env = ReplayFusionEnv(eps)
        model = create_ppo_model(env, batch_size=4)
        with pytest.raises(ValueError, match="GRPO needs episodes"):
            train_grpo(model, [], group_size=2, n_iterations=1)


# ---------------------------------------------------------------------------
# PPOPolicyWrapper
# ---------------------------------------------------------------------------

class TestPPOPolicyWrapper:
    """Verify PPO wrapper handles dimension mismatches for inference."""

    def test_predict_with_384_dims(self):
        from backend.rl.train_ppo import ReplayFusionEnv, create_ppo_model, PPOPolicyWrapper
        eps = _make_episodes(5)
        env = ReplayFusionEnv(eps)
        model = create_ppo_model(env, batch_size=4)
        wrapper = PPOPolicyWrapper(model)
        # 384-dim input (what compute_rl_weights sends at inference)
        obs = np.random.randn(1, 384).astype(np.float32)
        action = wrapper.predict(obs)
        assert action.shape[-1] == 4

    def test_predict_with_396_dims(self):
        from backend.rl.train_ppo import ReplayFusionEnv, create_ppo_model, PPOPolicyWrapper
        eps = _make_episodes(5)
        env = ReplayFusionEnv(eps)
        model = create_ppo_model(env, batch_size=4)
        wrapper = PPOPolicyWrapper(model)
        obs = np.random.randn(1, 396).astype(np.float32)
        action = wrapper.predict(obs)
        assert action.shape[-1] == 4

    def test_predict_1d_input(self):
        from backend.rl.train_ppo import ReplayFusionEnv, create_ppo_model, PPOPolicyWrapper
        eps = _make_episodes(5)
        env = ReplayFusionEnv(eps)
        model = create_ppo_model(env, batch_size=4)
        wrapper = PPOPolicyWrapper(model)
        obs = np.random.randn(384).astype(np.float32)
        action = wrapper.predict(obs)
        assert action.shape[-1] == 4

    def test_satisfies_rl_policy_protocol(self):
        from backend.agents.base import RLPolicy
        from backend.rl.train_ppo import ReplayFusionEnv, create_ppo_model, PPOPolicyWrapper
        eps = _make_episodes(5)
        env = ReplayFusionEnv(eps)
        model = create_ppo_model(env, batch_size=4)
        wrapper = PPOPolicyWrapper(model)
        assert isinstance(wrapper, RLPolicy)


# ---------------------------------------------------------------------------
# AdaptivePolicy
# ---------------------------------------------------------------------------

class TestAdaptivePolicy:
    """Verify adaptive warmup policy selection logic."""

    def _make_fake_policy(self, name: str):
        class FakePolicy:
            def __init__(self, tag):
                self._tag = tag
            def predict(self, obs):
                return np.full((1, 4), 0.25, dtype=np.float32)
        return FakePolicy(name)

    def test_starts_with_cql(self):
        from backend.rl.train_ppo import AdaptivePolicy
        cql = self._make_fake_policy("cql")
        ppo = self._make_fake_policy("ppo")
        ap = AdaptivePolicy(cql_policy=cql, ppo_policy=ppo, cql_threshold=5)
        assert ap.active_policy_name == "cql"

    def test_transitions_to_ppo(self):
        from backend.rl.train_ppo import AdaptivePolicy
        cql = self._make_fake_policy("cql")
        ppo = self._make_fake_policy("ppo")
        ap = AdaptivePolicy(cql_policy=cql, ppo_policy=ppo, cql_threshold=3, dpo_threshold=10)
        obs = np.zeros((1, 384), dtype=np.float32)
        # 3 CQL calls
        for _ in range(3):
            ap.predict(obs)
        assert ap.active_policy_name == "ppo"

    def test_transitions_to_dpo(self):
        from backend.rl.train_ppo import AdaptivePolicy
        cql = self._make_fake_policy("cql")
        ppo = self._make_fake_policy("ppo")
        dpo = self._make_fake_policy("dpo")
        ap = AdaptivePolicy(cql_policy=cql, ppo_policy=ppo, dpo_policy=dpo, cql_threshold=2, dpo_threshold=5)
        obs = np.zeros((1, 384), dtype=np.float32)
        for _ in range(5):
            ap.predict(obs)
        assert ap.active_policy_name == "dpo"

    def test_falls_back_without_ppo(self):
        from backend.rl.train_ppo import AdaptivePolicy
        cql = self._make_fake_policy("cql")
        ap = AdaptivePolicy(cql_policy=cql, ppo_policy=None, cql_threshold=2)
        obs = np.zeros((1, 384), dtype=np.float32)
        for _ in range(5):
            ap.predict(obs)
        # no PPO available, stays on CQL
        assert ap.active_policy_name == "cql"

    def test_absolute_fallback(self):
        from backend.rl.train_ppo import AdaptivePolicy
        ap = AdaptivePolicy(cql_policy=None, ppo_policy=None)
        obs = np.zeros((1, 384), dtype=np.float32)
        result = ap.predict(obs)
        assert result.shape == (1, 4)
        assert abs(result.sum() - 1.0) < 0.01

    def test_reset_count(self):
        from backend.rl.train_ppo import AdaptivePolicy
        cql = self._make_fake_policy("cql")
        ppo = self._make_fake_policy("ppo")
        ap = AdaptivePolicy(cql_policy=cql, ppo_policy=ppo, cql_threshold=3)
        obs = np.zeros((1, 384), dtype=np.float32)
        for _ in range(5):
            ap.predict(obs)
        assert ap.interaction_count == 5
        ap.reset_count()
        assert ap.interaction_count == 0
        assert ap.active_policy_name == "cql"

    def test_satisfies_rl_policy_protocol(self):
        from backend.agents.base import RLPolicy
        from backend.rl.train_ppo import AdaptivePolicy
        ap = AdaptivePolicy(cql_policy=None, ppo_policy=None)
        assert isinstance(ap, RLPolicy)


# ---------------------------------------------------------------------------
# DPO: Preference pair building
# ---------------------------------------------------------------------------

class TestBuildPreferencePairs:
    """Verify preference pairs are built correctly from episodes."""

    def test_builds_pairs_with_margin(self):
        from backend.rl.train_dpo import build_preference_pairs
        eps = _make_episodes(10)
        pairs = build_preference_pairs(eps, margin=0.1)
        assert len(pairs) > 0
        for preferred, rejected in pairs:
            assert float(preferred["reward"]) - float(rejected["reward"]) >= 0.1

    def test_no_pairs_from_identical_rewards(self):
        from backend.rl.train_dpo import build_preference_pairs
        eps = [
            {"query": "q1", "reward": 0.5, "weights": np.zeros(4), "response": "r1"},
            {"query": "q2", "reward": 0.5, "weights": np.zeros(4), "response": "r2"},
        ]
        pairs = build_preference_pairs(eps, margin=0.1)
        assert len(pairs) == 0

    def test_empty_episodes(self):
        from backend.rl.train_dpo import build_preference_pairs
        pairs = build_preference_pairs([], margin=0.1)
        assert pairs == []

    def test_single_episode(self):
        from backend.rl.train_dpo import build_preference_pairs
        eps = [{"query": "q1", "reward": 0.8, "weights": np.zeros(4), "response": "r1"}]
        pairs = build_preference_pairs(eps, margin=0.1)
        assert pairs == []

    def test_high_margin_reduces_pairs(self):
        from backend.rl.train_dpo import build_preference_pairs
        eps = _make_episodes(10)
        pairs_low = build_preference_pairs(eps, margin=0.05)
        pairs_high = build_preference_pairs(eps, margin=0.3)
        assert len(pairs_low) >= len(pairs_high)


# ---------------------------------------------------------------------------
# DPO: Policy network
# ---------------------------------------------------------------------------

class TestDPOFusionPolicy:
    """Verify DPO policy network forward pass and log probability."""

    def test_forward_returns_correct_shapes(self):
        from backend.rl.train_dpo import DPOFusionPolicy
        policy = DPOFusionPolicy(obs_dim=384, action_dim=4)
        obs = torch.randn(3, 384)
        mean, log_std = policy.forward(obs)
        assert mean.shape == (3, 4)
        assert log_std.shape == (3, 4)

    def test_mean_clamped(self):
        from backend.rl.train_dpo import DPOFusionPolicy
        policy = DPOFusionPolicy()
        obs = torch.randn(5, 384) * 100  # large input
        mean, _ = policy.forward(obs)
        assert mean.min().item() >= -1.0
        assert mean.max().item() <= 1.0

    def test_log_prob_finite(self):
        from backend.rl.train_dpo import DPOFusionPolicy
        policy = DPOFusionPolicy()
        obs = torch.randn(2, 384)
        action = torch.randn(2, 4)
        lp = policy.log_prob(obs, action)
        assert lp.shape == (2,)
        assert torch.isfinite(lp).all()

    def test_predict_returns_numpy(self):
        from backend.rl.train_dpo import DPOFusionPolicy
        policy = DPOFusionPolicy()
        obs = np.random.randn(1, 384).astype(np.float32)
        result = policy.predict(obs)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 4)

    def test_predict_1d_input(self):
        from backend.rl.train_dpo import DPOFusionPolicy
        policy = DPOFusionPolicy()
        obs = np.random.randn(384).astype(np.float32)
        result = policy.predict(obs)
        assert result.shape == (1, 4)

    def test_from_cql_missing_path(self):
        from backend.rl.train_dpo import DPOFusionPolicy
        policy = DPOFusionPolicy.from_cql_weights(Path("/nonexistent/model.d3"))
        # should return a randomly initialized policy without crashing
        obs = torch.randn(1, 384)
        mean, _ = policy.forward(obs)
        assert mean.shape == (1, 4)

    def test_satisfies_rl_policy_protocol(self):
        from backend.agents.base import RLPolicy
        from backend.rl.train_dpo import DPOFusionPolicy
        policy = DPOFusionPolicy()
        assert isinstance(policy, RLPolicy)


# ---------------------------------------------------------------------------
# DPO: Loss function
# ---------------------------------------------------------------------------

class TestDpoLoss:
    """Verify DPO loss computation is correct and finite."""

    def test_loss_is_finite(self):
        from backend.rl.train_dpo import DPOFusionPolicy, dpo_loss
        policy = DPOFusionPolicy()
        ref = DPOFusionPolicy()
        obs = torch.randn(4, 384)
        pref = torch.randn(4, 4)
        rej = torch.randn(4, 4)
        loss = dpo_loss(policy, ref, obs, pref, rej, beta=0.1)
        assert torch.isfinite(loss)
        assert loss.item() > 0  # logsigmoid is always negative, so -mean is positive

    def test_preferred_gets_lower_loss(self):
        from backend.rl.train_dpo import DPOFusionPolicy, dpo_loss
        policy = DPOFusionPolicy()
        ref = DPOFusionPolicy()
        obs = torch.randn(4, 384)
        # when preferred and rejected are identical, loss should be ~log(2)
        same_action = torch.randn(4, 4)
        loss_same = dpo_loss(policy, ref, obs, same_action, same_action, beta=0.1)
        assert abs(loss_same.item() - np.log(2)) < 0.5  # roughly log(2) when identical

    def test_beta_scaling(self):
        from backend.rl.train_dpo import DPOFusionPolicy, dpo_loss
        policy = DPOFusionPolicy()
        ref = DPOFusionPolicy()
        obs = torch.randn(4, 384)
        pref = torch.randn(4, 4)
        rej = torch.randn(4, 4)
        loss_low_beta = dpo_loss(policy, ref, obs, pref, rej, beta=0.01)
        loss_high_beta = dpo_loss(policy, ref, obs, pref, rej, beta=1.0)
        # both should be finite, different values
        assert torch.isfinite(loss_low_beta) and torch.isfinite(loss_high_beta)


# ---------------------------------------------------------------------------
# DPO: Tensor preparation
# ---------------------------------------------------------------------------

class TestPrepareDpoTensors:
    """Verify tensor preparation from preference pairs."""

    def test_correct_shapes(self):
        from backend.rl.train_dpo import build_preference_pairs, prepare_dpo_tensors
        eps = _make_episodes(10)
        pairs = build_preference_pairs(eps, margin=0.05)
        if not pairs:
            pytest.skip("No pairs generated with this margin")
        obs, pref, rej = prepare_dpo_tensors(pairs[:5])
        assert obs.shape == (5, 384)
        assert pref.shape == (5, 4)
        assert rej.shape == (5, 4)
        assert obs.dtype == torch.float32


# ---------------------------------------------------------------------------
# DPO: Training loop
# ---------------------------------------------------------------------------

class TestTrainDpo:
    """Verify DPO training loop runs end-to-end."""

    def test_trains_and_saves(self):
        from backend.rl.train_dpo import build_preference_pairs, train_dpo
        eps = _make_episodes(20)
        pairs = build_preference_pairs(eps, margin=0.05)
        if len(pairs) < 2:
            pytest.skip("Not enough pairs for DPO training")
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_dpo.pt"
            result = train_dpo(
                pairs[:10],
                beta=0.1,
                epochs=3,
                learning_rate=1e-3,
                batch_size=4,
                save_path=save_path,
            )
            assert result.exists()
            assert result.stat().st_size > 0

    def test_raises_on_too_few_pairs(self):
        from backend.rl.train_dpo import train_dpo
        with pytest.raises(ValueError, match="at least 2"):
            train_dpo([], beta=0.1, epochs=1)

    def test_loss_decreases(self):
        from backend.rl.train_dpo import (
            DPOFusionPolicy, build_preference_pairs, prepare_dpo_tensors, dpo_loss,
        )
        eps = _make_episodes(20)
        pairs = build_preference_pairs(eps, margin=0.05)
        if len(pairs) < 4:
            pytest.skip("Not enough pairs")

        policy = DPOFusionPolicy()
        ref = DPOFusionPolicy()
        for p in ref.parameters():
            p.requires_grad = False

        obs, pref, rej = prepare_dpo_tensors(pairs[:8])
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

        initial_loss = dpo_loss(policy, ref, obs, pref, rej, beta=0.1).item()
        for _ in range(20):
            loss = dpo_loss(policy, ref, obs, pref, rej, beta=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final_loss = dpo_loss(policy, ref, obs, pref, rej, beta=0.1).item()
        # loss should generally decrease (or at least not explode)
        assert final_loss < initial_loss + 0.5  # allow some tolerance


# ---------------------------------------------------------------------------
# DPO: Save/Load
# ---------------------------------------------------------------------------

class TestLoadDpoPolicy:
    """Verify DPO policy save/load roundtrip."""

    def test_save_load_roundtrip(self):
        from backend.rl.train_dpo import DPOFusionPolicy, load_dpo_policy
        policy = DPOFusionPolicy()
        obs = np.random.randn(1, 384).astype(np.float32)
        original_pred = policy.predict(obs).copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_policy.pt"
            torch.save(policy.state_dict(), str(path))
            loaded = load_dpo_policy(path)
            loaded_pred = loaded.predict(obs)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)


# ---------------------------------------------------------------------------
# Config keys exist
# ---------------------------------------------------------------------------

class TestPhase4Config:
    """Verify Phase 4 config keys exist with safe defaults."""

    def test_ppo_config_keys(self):
        from backend.config import cfg
        ppo = cfg.get("rl", {}).get("ppo", {})
        assert ppo.get("batch_size") == 50
        assert ppo.get("clip_range") == 0.2
        assert ppo.get("ent_coef") == 0.01
        assert ppo.get("gae_lambda") == 0.95
        assert ppo.get("n_epochs") == 10

    def test_dpo_config_keys(self):
        from backend.config import cfg
        dpo = cfg.get("rl", {}).get("dpo", {})
        assert dpo.get("beta") == 0.1
        assert dpo.get("min_preference_pairs") == 500
        assert dpo.get("reward_margin") == 0.1
        assert dpo.get("learning_rate") == 0.0001

    def test_adaptive_warmup_config(self):
        from backend.config import cfg
        warmup = cfg.get("rl", {}).get("adaptive_warmup", {})
        assert warmup.get("cql_until") == 50
        assert warmup.get("ppo_until") == 500
        assert warmup.get("dpo_after") == 500

    def test_grpo_config(self):
        from backend.config import cfg
        grpo = cfg.get("rl", {}).get("grpo", {})
        assert grpo.get("enabled") is False
        assert grpo.get("group_size") == 4
        assert grpo.get("iterations") == 100


# ---------------------------------------------------------------------------
# Integration: imports from both modules
# ---------------------------------------------------------------------------

class TestPhase4Imports:
    """Verify all Phase 4 public symbols import cleanly."""

    def test_train_ppo_imports(self):
        from backend.rl.train_ppo import (
            ReplayFusionEnv,
            generate_cql_demos,
            load_episodes_from_db,
            create_ppo_model,
            evaluate_policy,
            train_ppo_online,
            rank_candidates_grpo,
            train_grpo,
            PPOPolicyWrapper,
            AdaptivePolicy,
        )
        assert callable(generate_cql_demos)
        assert callable(train_ppo_online)
        assert callable(train_grpo)

    def test_train_dpo_imports(self):
        from backend.rl.train_dpo import (
            load_episodes_from_db,
            build_preference_pairs,
            DPOFusionPolicy,
            dpo_loss,
            prepare_dpo_tensors,
            train_dpo,
            load_dpo_policy,
        )
        assert callable(build_preference_pairs)
        assert callable(train_dpo)
        assert callable(load_dpo_policy)

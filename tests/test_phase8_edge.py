# Author: Bradley R. Kinnard
# test_phase8_edge.py - tests for Phase 8 Edge AI & Distributed Inference
# Covers scheduler, quantization, hardware detection, federated learning.

import asyncio
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
# scheduler.py — Hardware detection
# ---------------------------------------------------------------------------

class TestDetectHardware:
    """Verify hardware probing returns a valid profile."""

    def test_returns_hardware_profile(self):
        from backend.core.scheduler import detect_hardware
        profile = detect_hardware()
        assert isinstance(profile, dict)
        assert "gpu_available" in profile
        assert "cpu_arch" in profile
        assert "ram_total_mb" in profile
        assert "is_arm" in profile
        assert "pointer_bits" in profile

    def test_cpu_count_positive(self):
        from backend.core.scheduler import detect_hardware
        profile = detect_hardware()
        assert profile["cpu_count"] >= 1

    def test_ram_detected(self):
        from backend.core.scheduler import detect_hardware
        profile = detect_hardware()
        # on any real machine, we should detect at least some RAM
        assert profile["ram_total_mb"] > 0

    def test_pointer_bits_valid(self):
        from backend.core.scheduler import detect_hardware
        profile = detect_hardware()
        assert profile["pointer_bits"] in (32, 64)


# ---------------------------------------------------------------------------
# scheduler.py — Quantization recommendation
# ---------------------------------------------------------------------------

class TestQuantRecommendation:
    """Verify quantization level selection logic."""

    def test_no_gpu_recommends_q4(self):
        from backend.core.scheduler import HardwareProfile, recommend_quantization
        profile = HardwareProfile(
            gpu_available=False, gpu_name="", vram_total_mb=0, vram_free_mb=0,
            ram_total_mb=8192, ram_free_mb=4096, cpu_arch="x86_64",
            cpu_count=4, is_arm=False, pointer_bits=64,
        )
        rec = recommend_quantization(profile)
        assert rec["level"] == "Q4_K_M"
        assert rec["fits_in_vram"] is False
        assert "No GPU" in rec["reason"]

    def test_high_vram_recommends_f16(self):
        from backend.core.scheduler import HardwareProfile, recommend_quantization
        profile = HardwareProfile(
            gpu_available=True, gpu_name="RTX 5070", vram_total_mb=24000,
            vram_free_mb=20000, ram_total_mb=32768, ram_free_mb=16384,
            cpu_arch="x86_64", cpu_count=16, is_arm=False, pointer_bits=64,
        )
        rec = recommend_quantization(profile)
        assert rec["level"] == "F16"
        assert rec["fits_in_vram"] is True

    def test_medium_vram_recommends_q8(self):
        from backend.core.scheduler import HardwareProfile, recommend_quantization
        profile = HardwareProfile(
            gpu_available=True, gpu_name="RTX 4060", vram_total_mb=8192,
            vram_free_mb=8000, ram_total_mb=16384, ram_free_mb=8000,
            cpu_arch="x86_64", cpu_count=8, is_arm=False, pointer_bits=64,
        )
        rec = recommend_quantization(profile)
        assert rec["level"] == "Q8_0"
        assert rec["fits_in_vram"] is True

    def test_low_vram_recommends_q5(self):
        from backend.core.scheduler import HardwareProfile, recommend_quantization
        profile = HardwareProfile(
            gpu_available=True, gpu_name="GTX 1660", vram_total_mb=6144,
            vram_free_mb=5600, ram_total_mb=16384, ram_free_mb=8000,
            cpu_arch="x86_64", cpu_count=8, is_arm=False, pointer_bits=64,
        )
        rec = recommend_quantization(profile)
        assert rec["level"] == "Q5_K_M"

    def test_very_low_vram_recommends_q4(self):
        from backend.core.scheduler import HardwareProfile, recommend_quantization
        profile = HardwareProfile(
            gpu_available=True, gpu_name="MX150", vram_total_mb=2048,
            vram_free_mb=1800, ram_total_mb=8192, ram_free_mb=4000,
            cpu_arch="x86_64", cpu_count=4, is_arm=False, pointer_bits=64,
        )
        rec = recommend_quantization(profile)
        assert rec["level"] == "Q4_K_M"
        assert rec["fits_in_vram"] is False


# ---------------------------------------------------------------------------
# scheduler.py — Policy quantization (weights)
# ---------------------------------------------------------------------------

class TestPolicyQuantization:
    """Verify RL policy weight quantization and dequantization."""

    def test_float16_quantization(self):
        from backend.core.scheduler import quantize_policy_weights
        weights = {"layer1": np.random.randn(10, 4).astype(np.float32)}
        q = quantize_policy_weights(weights, precision="float16")
        assert q["layer1"].dtype == np.float16
        # shape preserved
        assert q["layer1"].shape == (10, 4)

    def test_int8_quantization_and_dequant(self):
        from backend.core.scheduler import dequantize_int8, quantize_policy_weights
        original = np.array([0.1, 0.5, -0.3, 1.2, -1.0], dtype=np.float32)
        weights = {"w": original}
        q = quantize_policy_weights(weights, precision="int8")
        assert isinstance(q["w"], dict)
        assert "data" in q["w"]
        assert "scale" in q["w"]
        # dequantize and check closeness
        restored = dequantize_int8(q["w"])
        np.testing.assert_allclose(original, restored, atol=0.02)

    def test_float32_is_copy(self):
        from backend.core.scheduler import quantize_policy_weights
        original = np.ones((3, 3), dtype=np.float32)
        q = quantize_policy_weights({"w": original}, precision="float32")
        assert q["w"].dtype == np.float32
        np.testing.assert_array_equal(q["w"], original)
        # verify it's a copy, not the same object
        assert q["w"] is not original

    def test_invalid_precision_raises(self):
        from backend.core.scheduler import quantize_policy_weights
        with pytest.raises(ValueError, match="Unsupported precision"):
            quantize_policy_weights({"w": np.ones(3)}, precision="bfloat16")

    def test_non_float_keys_passed_through(self):
        from backend.core.scheduler import quantize_policy_weights
        weights = {"floats": np.ones(5, dtype=np.float32), "config": "some_string"}
        q = quantize_policy_weights(weights, precision="float16")
        assert q["config"] == "some_string"
        assert q["floats"].dtype == np.float16


class TestPolicyPrecisionRecommendation:
    """Verify recommend_policy_precision picks right precision for hardware."""

    def test_no_gpu_low_ram(self):
        from backend.core.scheduler import HardwareProfile, recommend_policy_precision
        profile = HardwareProfile(
            gpu_available=False, gpu_name="", vram_total_mb=0, vram_free_mb=0,
            ram_total_mb=4096, ram_free_mb=1500, cpu_arch="aarch64",
            cpu_count=4, is_arm=True, pointer_bits=64,
        )
        assert recommend_policy_precision(profile) == "int8"

    def test_no_gpu_high_ram(self):
        from backend.core.scheduler import HardwareProfile, recommend_policy_precision
        profile = HardwareProfile(
            gpu_available=False, gpu_name="", vram_total_mb=0, vram_free_mb=0,
            ram_total_mb=16384, ram_free_mb=8000, cpu_arch="x86_64",
            cpu_count=8, is_arm=False, pointer_bits=64,
        )
        assert recommend_policy_precision(profile) == "float16"

    def test_high_vram_gets_float32(self):
        from backend.core.scheduler import HardwareProfile, recommend_policy_precision
        profile = HardwareProfile(
            gpu_available=True, gpu_name="RTX 5070", vram_total_mb=24000,
            vram_free_mb=16000, cpu_arch="x86_64", cpu_count=16,
            ram_total_mb=32768, ram_free_mb=16000, is_arm=False, pointer_bits=64,
        )
        assert recommend_policy_precision(profile) == "float32"


# ---------------------------------------------------------------------------
# scheduler.py — Batch size recommendation
# ---------------------------------------------------------------------------

class TestBatchSizeRecommendation:
    """Confirm batch sizes scale with available resources."""

    def test_safety_always_one(self):
        from backend.core.scheduler import HardwareProfile, TaskLane, recommend_batch_size
        profile = HardwareProfile(
            gpu_available=True, gpu_name="A100", vram_total_mb=80000,
            vram_free_mb=60000, ram_total_mb=256000, ram_free_mb=200000,
            cpu_arch="x86_64", cpu_count=64, is_arm=False, pointer_bits=64,
        )
        assert recommend_batch_size(profile, TaskLane.SAFETY) == 1

    def test_no_memory_halves_base(self):
        from backend.core.scheduler import HardwareProfile, TaskLane, recommend_batch_size
        profile = HardwareProfile(
            gpu_available=False, gpu_name="", vram_total_mb=0, vram_free_mb=0,
            ram_total_mb=0, ram_free_mb=0, cpu_arch="x86_64",
            cpu_count=2, is_arm=False, pointer_bits=64,
        )
        batch = recommend_batch_size(profile, TaskLane.RETRIEVAL)
        assert batch >= 1

    def test_high_memory_scales_up(self):
        from backend.core.scheduler import HardwareProfile, TaskLane, recommend_batch_size
        profile = HardwareProfile(
            gpu_available=True, gpu_name="A100", vram_total_mb=80000,
            vram_free_mb=40000, ram_total_mb=128000, ram_free_mb=64000,
            cpu_arch="x86_64", cpu_count=32, is_arm=False, pointer_bits=64,
        )
        batch = recommend_batch_size(profile, TaskLane.RETRIEVAL)
        # base is 8, with 40GB VRAM it should scale up
        assert batch >= 8


# ---------------------------------------------------------------------------
# scheduler.py — TaskScheduler
# ---------------------------------------------------------------------------

class TestTaskScheduler:
    """Verify the async task scheduler with priority lanes."""

    def test_instantiation(self):
        from backend.core.scheduler import HardwareProfile, TaskScheduler
        profile = HardwareProfile(
            gpu_available=False, gpu_name="", vram_total_mb=0, vram_free_mb=0,
            ram_total_mb=8192, ram_free_mb=4096, cpu_arch="x86_64",
            cpu_count=4, is_arm=False, pointer_bits=64,
        )
        sched = TaskScheduler(profile=profile)
        assert sched.hardware_profile == profile

    def test_submit_safety_task(self):
        from backend.core.scheduler import HardwareProfile, TaskLane, TaskScheduler

        async def _run():
            profile = HardwareProfile(
                gpu_available=False, gpu_name="", vram_total_mb=0, vram_free_mb=0,
                ram_total_mb=8192, ram_free_mb=4096, cpu_arch="x86_64",
                cpu_count=4, is_arm=False, pointer_bits=64,
            )
            sched = TaskScheduler(profile=profile)

            async def safety_check():
                return {"safe": True}

            result = await sched.submit(TaskLane.SAFETY, safety_check())
            assert result == {"safe": True}

        asyncio.run(_run())

    def test_lane_status_after_tasks(self):
        from backend.core.scheduler import HardwareProfile, TaskLane, TaskScheduler

        async def _run():
            profile = HardwareProfile(
                gpu_available=False, gpu_name="", vram_total_mb=0, vram_free_mb=0,
                ram_total_mb=8192, ram_free_mb=4096, cpu_arch="x86_64",
                cpu_count=4, is_arm=False, pointer_bits=64,
            )
            sched = TaskScheduler(profile=profile)

            async def noop():
                return 42

            await sched.submit(TaskLane.RETRIEVAL, noop())
            await sched.submit(TaskLane.RETRIEVAL, noop())
            status = sched.get_lane_status()
            assert status["retrieval"]["completed"] == 2
            assert status["retrieval"]["pending"] == 0
            assert status["safety"]["completed"] == 0

        asyncio.run(_run())

    def test_recommend_batch_convenience(self):
        from backend.core.scheduler import HardwareProfile, TaskLane, TaskScheduler
        profile = HardwareProfile(
            gpu_available=False, gpu_name="", vram_total_mb=0, vram_free_mb=0,
            ram_total_mb=8192, ram_free_mb=4096, cpu_arch="x86_64",
            cpu_count=4, is_arm=False, pointer_bits=64,
        )
        sched = TaskScheduler(profile=profile)
        batch = sched.recommend_batch(TaskLane.SAFETY)
        assert batch == 1


# ---------------------------------------------------------------------------
# federated.py — Delta extraction
# ---------------------------------------------------------------------------

class TestDeltaExtraction:
    """Verify extract_delta computes correct weight diffs."""

    def test_basic_delta(self):
        from backend.rl.federated import extract_delta
        before = {"w1": np.ones((3, 3), dtype=np.float32)}
        after = {"w1": np.ones((3, 3), dtype=np.float32) * 2}
        delta = extract_delta(before, after, num_episodes=10, avg_reward=0.8)
        np.testing.assert_array_almost_equal(delta["layer_deltas"]["w1"], np.ones((3, 3)))
        assert delta["num_episodes"] == 10
        assert delta["avg_reward"] == 0.8
        assert len(delta["delta_id"]) == 16

    def test_key_mismatch_raises(self):
        from backend.rl.federated import extract_delta
        before = {"w1": np.ones(3, dtype=np.float32)}
        after = {"w2": np.ones(3, dtype=np.float32)}
        with pytest.raises(ValueError, match="Key mismatch"):
            extract_delta(before, after)

    def test_shape_mismatch_raises(self):
        from backend.rl.federated import extract_delta
        before = {"w": np.ones((3,), dtype=np.float32)}
        after = {"w": np.ones((4,), dtype=np.float32)}
        with pytest.raises(ValueError, match="Shape mismatch"):
            extract_delta(before, after)

    def test_zero_delta(self):
        from backend.rl.federated import extract_delta
        w = np.random.randn(5, 5).astype(np.float32)
        delta = extract_delta({"w": w.copy()}, {"w": w.copy()})
        np.testing.assert_array_almost_equal(delta["layer_deltas"]["w"], np.zeros((5, 5)))


# ---------------------------------------------------------------------------
# federated.py — Differential privacy
# ---------------------------------------------------------------------------

class TestClipAndNoise:
    """Verify DP sanitization clips norms and adds noise."""

    def test_clipping_large_delta(self):
        from backend.rl.federated import PrivacyConfig, clip_and_noise, extract_delta
        before = {"w": np.zeros((10,), dtype=np.float32)}
        after = {"w": np.ones((10,), dtype=np.float32) * 100}
        delta = extract_delta(before, after)
        privacy = PrivacyConfig(clip_norm=1.0, noise_scale=0.0, min_contributors=1)
        rng = np.random.default_rng(42)
        sanitized = clip_and_noise(delta, privacy, rng=rng)
        norm = float(np.linalg.norm(sanitized["layer_deltas"]["w"]))
        assert norm <= 1.0 + 1e-6

    def test_noise_added(self):
        from backend.rl.federated import PrivacyConfig, clip_and_noise, extract_delta
        before = {"w": np.zeros((50,), dtype=np.float32)}
        after = {"w": np.ones((50,), dtype=np.float32) * 0.01}
        delta = extract_delta(before, after)
        privacy = PrivacyConfig(clip_norm=10.0, noise_scale=1.0, min_contributors=1)
        rng = np.random.default_rng(99)
        sanitized = clip_and_noise(delta, privacy, rng=rng)
        # with noise_scale=1.0 on a tiny delta, the result should differ noticeably
        raw_norm = float(np.linalg.norm(delta["layer_deltas"]["w"]))
        noised_norm = float(np.linalg.norm(sanitized["layer_deltas"]["w"]))
        assert noised_norm != pytest.approx(raw_norm, abs=0.01)

    def test_zero_noise_preserves_direction(self):
        from backend.rl.federated import PrivacyConfig, clip_and_noise, extract_delta
        before = {"w": np.zeros((5,), dtype=np.float32)}
        after = {"w": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)}
        delta = extract_delta(before, after)
        privacy = PrivacyConfig(clip_norm=100.0, noise_scale=0.0, min_contributors=1)
        rng = np.random.default_rng(0)
        sanitized = clip_and_noise(delta, privacy, rng=rng)
        np.testing.assert_array_almost_equal(
            sanitized["layer_deltas"]["w"],
            delta["layer_deltas"]["w"],
        )


# ---------------------------------------------------------------------------
# federated.py — Aggregation
# ---------------------------------------------------------------------------

class TestAggregateDeltas:
    """Verify FedAvg-style aggregation of policy deltas."""

    def _make_delta(self, val: float, episodes: int = 10) -> "PolicyDelta":
        from backend.rl.federated import extract_delta
        before = {"w": np.zeros((4,), dtype=np.float32)}
        after = {"w": np.full((4,), val, dtype=np.float32)}
        return extract_delta(before, after, num_episodes=episodes, avg_reward=val)

    def test_two_equal_deltas(self):
        from backend.rl.federated import aggregate_deltas
        d1 = self._make_delta(1.0, episodes=10)
        d2 = self._make_delta(1.0, episodes=10)
        result = aggregate_deltas([d1, d2], min_contributors=1)
        np.testing.assert_array_almost_equal(
            result["merged_delta"]["w"],
            np.ones(4, dtype=np.float32),
        )
        assert result["num_contributors"] == 2

    def test_weighted_by_episodes(self):
        from backend.rl.federated import aggregate_deltas
        d1 = self._make_delta(1.0, episodes=90)
        d2 = self._make_delta(2.0, episodes=10)
        result = aggregate_deltas([d1, d2], min_contributors=1)
        # 90% weight on 1.0, 10% weight on 2.0 -> 1.1
        np.testing.assert_array_almost_equal(
            result["merged_delta"]["w"],
            np.full(4, 1.1, dtype=np.float32),
            decimal=5,
        )

    def test_too_few_contributors_raises(self):
        from backend.rl.federated import aggregate_deltas
        d = self._make_delta(1.0)
        with pytest.raises(ValueError, match="Need at least"):
            aggregate_deltas([d], min_contributors=3)

    def test_empty_list_raises(self):
        from backend.rl.federated import aggregate_deltas
        with pytest.raises(ValueError, match="Need at least"):
            aggregate_deltas([], min_contributors=1)

    def test_zero_episodes_uniform_weighting(self):
        from backend.rl.federated import aggregate_deltas
        d1 = self._make_delta(2.0, episodes=0)
        d2 = self._make_delta(4.0, episodes=0)
        result = aggregate_deltas([d1, d2], min_contributors=1)
        np.testing.assert_array_almost_equal(
            result["merged_delta"]["w"],
            np.full(4, 3.0, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# federated.py — Delta application
# ---------------------------------------------------------------------------

class TestApplyDelta:
    """Verify merging aggregated delta into base weights."""

    def test_full_learning_rate(self):
        from backend.rl.federated import aggregate_deltas, apply_delta, extract_delta
        base = {"w": np.ones((4,), dtype=np.float32)}
        before = {"w": np.zeros((4,), dtype=np.float32)}
        after = {"w": np.ones((4,), dtype=np.float32) * 0.5}
        d = extract_delta(before, after, num_episodes=5)
        update = aggregate_deltas([d, d], min_contributors=1)
        result = apply_delta(base, update, learning_rate=1.0)
        # base(1.0) + delta(0.5) = 1.5
        np.testing.assert_array_almost_equal(result["w"], np.full(4, 1.5))

    def test_half_learning_rate(self):
        from backend.rl.federated import aggregate_deltas, apply_delta, extract_delta
        base = {"w": np.ones((4,), dtype=np.float32)}
        before = {"w": np.zeros((4,), dtype=np.float32)}
        after = {"w": np.ones((4,), dtype=np.float32)}
        d = extract_delta(before, after, num_episodes=5)
        update = aggregate_deltas([d, d], min_contributors=1)
        result = apply_delta(base, update, learning_rate=0.5)
        # base(1.0) + 0.5 * delta(1.0) = 1.5
        np.testing.assert_array_almost_equal(result["w"], np.full(4, 1.5))

    def test_invalid_learning_rate_raises(self):
        from backend.rl.federated import AggregatedUpdate, apply_delta
        base = {"w": np.ones(3, dtype=np.float32)}
        update = AggregatedUpdate(
            merged_delta={"w": np.ones(3, dtype=np.float32)},
            num_contributors=1, total_episodes=5, avg_reward=0.5, applied_at=0.0,
        )
        with pytest.raises(ValueError, match="learning_rate"):
            apply_delta(base, update, learning_rate=0.0)
        with pytest.raises(ValueError, match="learning_rate"):
            apply_delta(base, update, learning_rate=1.5)


# ---------------------------------------------------------------------------
# federated.py — Serialization
# ---------------------------------------------------------------------------

class TestDeltaSerialization:
    """Verify save/load round-trip for policy deltas."""

    def test_save_and_load(self):
        from backend.rl.federated import extract_delta, load_delta, save_delta
        before = {"w1": np.zeros((3, 3), dtype=np.float32)}
        after = {"w1": np.eye(3, dtype=np.float32)}
        delta = extract_delta(before, after, num_episodes=7, avg_reward=0.65)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_delta(delta, tmpdir)
            loaded = load_delta(path)
            assert loaded["delta_id"] == delta["delta_id"]
            assert loaded["num_episodes"] == 7
            np.testing.assert_array_almost_equal(
                loaded["layer_deltas"]["w1"],
                delta["layer_deltas"]["w1"],
            )

    def test_load_missing_file_raises(self):
        from backend.rl.federated import load_delta
        with pytest.raises(FileNotFoundError):
            load_delta("/nonexistent/path.npz")


# ---------------------------------------------------------------------------
# federated.py — FederatedCoordinator
# ---------------------------------------------------------------------------

class TestFederatedCoordinator:
    """Integration tests for the full federated coordination flow."""

    def test_compute_round(self):
        from backend.rl.federated import FederatedCoordinator
        coord = FederatedCoordinator()
        before = {"w": np.zeros((4,), dtype=np.float32)}
        after = {"w": np.ones((4,), dtype=np.float32)}
        delta = coord.compute_round(before, after, num_episodes=10, avg_reward=0.9)
        # should be clipped+noised, so not exactly 1.0
        assert delta["num_episodes"] == 10
        assert "layer_deltas" in delta

    def test_contribute_and_merge(self):
        from backend.rl.federated import FederatedCoordinator, PrivacyConfig
        privacy = PrivacyConfig(clip_norm=10.0, noise_scale=0.0, min_contributors=2)
        coord = FederatedCoordinator(privacy=privacy)

        before = {"w": np.zeros((4,), dtype=np.float32)}
        after = {"w": np.ones((4,), dtype=np.float32)}

        d1 = coord.compute_round(before, after, num_episodes=5, avg_reward=0.7)
        d2 = coord.compute_round(before, after, num_episodes=5, avg_reward=0.7)
        coord.contribute(d1)
        coord.contribute(d2)
        assert coord.contribution_count == 2

        base = {"w": np.zeros((4,), dtype=np.float32)}
        result = coord.merge(base)
        assert result["w"].shape == (4,)
        # after merge, contributions should be cleared
        assert coord.contribution_count == 0

    def test_merge_with_too_few_raises(self):
        from backend.rl.federated import FederatedCoordinator, PrivacyConfig
        privacy = PrivacyConfig(clip_norm=1.0, noise_scale=0.01, min_contributors=5)
        coord = FederatedCoordinator(privacy=privacy)
        before = {"w": np.zeros((2,), dtype=np.float32)}
        after = {"w": np.ones((2,), dtype=np.float32)}
        d = coord.compute_round(before, after)
        coord.contribute(d)
        with pytest.raises(ValueError, match="Need at least"):
            coord.merge({"w": np.zeros((2,), dtype=np.float32)})

    def test_clear(self):
        from backend.rl.federated import FederatedCoordinator, PrivacyConfig
        privacy = PrivacyConfig(clip_norm=1.0, noise_scale=0.0, min_contributors=1)
        coord = FederatedCoordinator(privacy=privacy)
        before = {"w": np.zeros(3, dtype=np.float32)}
        after = {"w": np.ones(3, dtype=np.float32)}
        d = coord.compute_round(before, after)
        coord.contribute(d)
        assert coord.contribution_count == 1
        coord.clear()
        assert coord.contribution_count == 0

    def test_persist_round(self):
        from backend.rl.federated import FederatedCoordinator, PrivacyConfig
        privacy = PrivacyConfig(clip_norm=10.0, noise_scale=0.0, min_contributors=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            coord = FederatedCoordinator(privacy=privacy, storage_dir=tmpdir)
            before = {"w": np.zeros((3,), dtype=np.float32)}
            after = {"w": np.ones((3,), dtype=np.float32)}
            delta = coord.compute_round(before, after, persist=True)
            # check files were written
            files = list(Path(tmpdir).glob("delta_*.npz"))
            assert len(files) == 1


# ---------------------------------------------------------------------------
# scheduler.py — TaskLane enum
# ---------------------------------------------------------------------------

class TestTaskLane:
    """Verify lane priority ordering."""

    def test_safety_highest_priority(self):
        from backend.core.scheduler import TaskLane
        assert TaskLane.SAFETY < TaskLane.RETRIEVAL
        assert TaskLane.SAFETY < TaskLane.GENERATION

    def test_ordering(self):
        from backend.core.scheduler import TaskLane
        order = [TaskLane.SAFETY, TaskLane.RETRIEVAL, TaskLane.FUSION,
                 TaskLane.CRITIQUE, TaskLane.GENERATION]
        assert order == sorted(order)


# ---------------------------------------------------------------------------
# Import smoke test for Phase 8
# ---------------------------------------------------------------------------

class TestPhase8Imports:
    """Verify all Phase 8 symbols are importable."""

    def test_scheduler_imports(self):
        from backend.core.scheduler import (
            HardwareProfile,
            LaneStats,
            PolicyQuantInfo,
            QuantRecommendation,
            TaskLane,
            TaskScheduler,
            dequantize_int8,
            detect_hardware,
            quantize_policy_weights,
            recommend_batch_size,
            recommend_policy_precision,
            recommend_quantization,
        )

    def test_federated_imports(self):
        from backend.rl.federated import (
            AggregatedUpdate,
            FederatedCoordinator,
            PolicyDelta,
            PrivacyConfig,
            aggregate_deltas,
            apply_delta,
            clip_and_noise,
            extract_delta,
            load_delta,
            save_delta,
        )

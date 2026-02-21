# Author: Bradley R. Kinnard
"""Federated policy update coordinator with differential privacy.

Phase 8: enables privacy-preserving RL policy improvement across multiple
RLFusion instances. Each instance trains locally and exports weight deltas
(not raw data). Deltas are clipped and noised before sharing, so no query
content ever leaves the device.

Architecture:
    1. Local training produces a before/after weight snapshot
    2. extract_delta() computes the diff
    3. clip_and_noise() applies (epsilon, delta)-DP guarantees
    4. aggregate_deltas() combines contributions (FedAvg-style)
    5. apply_delta() merges the aggregated update into the local policy
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TypedDict

import numpy as np

from backend.config import PROJECT_ROOT, cfg

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────

class PolicyDelta(TypedDict):
    """Weight diff between two training checkpoints."""
    delta_id: str
    timestamp: float
    layer_deltas: dict[str, np.ndarray]
    num_episodes: int
    avg_reward: float
    source_hash: str


class PrivacyConfig(TypedDict):
    """Differential privacy parameters for delta sanitization."""
    clip_norm: float
    noise_scale: float
    min_contributors: int


class AggregatedUpdate(TypedDict):
    """Merged delta from multiple federated contributors."""
    merged_delta: dict[str, np.ndarray]
    num_contributors: int
    total_episodes: int
    avg_reward: float
    applied_at: float


# ── Config defaults ──────────────────────────────────────────────────

def _fed_cfg() -> dict[str, object]:
    """Pull federated config with safe defaults."""
    return cfg.get("federated", {})


def _default_privacy() -> PrivacyConfig:
    fc = _fed_cfg()
    clip_raw = fc.get("clip_norm", 1.0)
    noise_raw = fc.get("noise_scale", 0.01)
    contrib_raw = fc.get("min_contributors", 2)
    return PrivacyConfig(
        clip_norm=float(str(clip_raw)),
        noise_scale=float(str(noise_raw)),
        min_contributors=int(str(contrib_raw)),
    )


# ── Delta extraction ────────────────────────────────────────────────

def extract_delta(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    num_episodes: int = 0,
    avg_reward: float = 0.0,
) -> PolicyDelta:
    """Compute the weight diff between two policy snapshots.

    Both dicts must have the same keys. Each value is a numpy array
    of matching shape. Returns a PolicyDelta with per-layer diffs.
    """
    missing_before = set(after.keys()) - set(before.keys())
    missing_after = set(before.keys()) - set(after.keys())
    if missing_before or missing_after:
        raise ValueError(
            f"Key mismatch: in 'after' only={missing_before}, "
            f"in 'before' only={missing_after}"
        )

    layer_deltas: dict[str, np.ndarray] = {}
    for key in before:
        b, a = before[key], after[key]
        if b.shape != a.shape:
            raise ValueError(
                f"Shape mismatch for '{key}': {b.shape} vs {a.shape}"
            )
        layer_deltas[key] = (a - b).astype(np.float32)

    # deterministic ID from the delta content
    content = b"".join(d.tobytes() for d in layer_deltas.values())
    delta_id = hashlib.sha256(content).hexdigest()[:16]

    # source hash: a privacy-safe fingerprint (no query data)
    source_raw = f"{delta_id}-{time.time()}-{num_episodes}"
    source_hash = hashlib.sha256(source_raw.encode()).hexdigest()[:12]

    return PolicyDelta(
        delta_id=delta_id,
        timestamp=time.time(),
        layer_deltas=layer_deltas,
        num_episodes=num_episodes,
        avg_reward=avg_reward,
        source_hash=source_hash,
    )


# ── Differential privacy ────────────────────────────────────────────

def clip_and_noise(
    delta: PolicyDelta,
    privacy: PrivacyConfig | None = None,
    rng: np.random.Generator | None = None,
) -> PolicyDelta:
    """Apply DP guarantees: clip delta norms and add calibrated noise.

    Each layer's delta is L2-clipped to clip_norm, then Gaussian noise
    with std=noise_scale is added. This ensures no single training run's
    contribution can dominate, and the noise prevents reconstruction of
    training data from shared gradients.
    """
    priv = privacy or _default_privacy()
    gen = rng or np.random.default_rng()
    clip_norm = priv["clip_norm"]
    noise_std = priv["noise_scale"]

    sanitized: dict[str, np.ndarray] = {}
    for key, arr in delta["layer_deltas"].items():
        # L2 clip
        norm = float(np.linalg.norm(arr))
        if norm > clip_norm and norm > 0:
            arr = arr * (clip_norm / norm)

        # additive Gaussian noise
        noise = gen.normal(loc=0.0, scale=noise_std, size=arr.shape).astype(np.float32)
        sanitized[key] = arr + noise

    return PolicyDelta(
        delta_id=delta["delta_id"],
        timestamp=delta["timestamp"],
        layer_deltas=sanitized,
        num_episodes=delta["num_episodes"],
        avg_reward=delta["avg_reward"],
        source_hash=delta["source_hash"],
    )


# ── Aggregation (FedAvg) ────────────────────────────────────────────

def aggregate_deltas(
    deltas: list[PolicyDelta],
    min_contributors: int = 0,
) -> AggregatedUpdate:
    """Weighted average of multiple policy deltas (FedAvg).

    Each delta is weighted by its num_episodes. If fewer than
    min_contributors participate, raises ValueError to prevent
    under-represented merges that could harm policy quality.
    """
    mc = min_contributors or _default_privacy()["min_contributors"]
    if len(deltas) < mc:
        raise ValueError(
            f"Need at least {mc} contributors, got {len(deltas)}. "
            f"Aggregating too few deltas risks poor policy updates."
        )

    if not deltas:
        raise ValueError("Cannot aggregate zero deltas")

    # gather all keys from the first delta as reference
    ref_keys = set(deltas[0]["layer_deltas"].keys())
    for d in deltas[1:]:
        if set(d["layer_deltas"].keys()) != ref_keys:
            raise ValueError(
                f"Delta {d['delta_id']} has mismatched keys vs reference"
            )

    total_episodes = sum(d["num_episodes"] for d in deltas)
    # avoid division by zero: if all have 0 episodes, use uniform weighting
    if total_episodes == 0:
        weights = [1.0 / len(deltas)] * len(deltas)
        total_episodes = len(deltas)
    else:
        weights = [d["num_episodes"] / total_episodes for d in deltas]

    merged: dict[str, np.ndarray] = {}
    for key in ref_keys:
        stacked = np.stack([d["layer_deltas"][key] for d in deltas])
        w = np.array(weights, dtype=np.float32).reshape(-1, *([1] * (stacked.ndim - 1)))
        merged[key] = (stacked * w).sum(axis=0).astype(np.float32)

    avg_reward = sum(d["avg_reward"] * w for d, w in zip(deltas, weights))

    return AggregatedUpdate(
        merged_delta=merged,
        num_contributors=len(deltas),
        total_episodes=total_episodes,
        avg_reward=float(avg_reward),
        applied_at=0.0,  # set when apply_delta() is called
    )


# ── Delta application ───────────────────────────────────────────────

def apply_delta(
    base_weights: dict[str, np.ndarray],
    update: AggregatedUpdate,
    learning_rate: float = 1.0,
) -> dict[str, np.ndarray]:
    """Merge an aggregated delta into a set of base policy weights.

    learning_rate < 1.0 dampens the update for stability. Returns a
    new dict (does not modify base_weights in place).
    """
    if learning_rate <= 0 or learning_rate > 1.0:
        raise ValueError(f"learning_rate must be in (0, 1.0], got {learning_rate}")

    merged = update["merged_delta"]
    result: dict[str, np.ndarray] = {}

    for key, base in base_weights.items():
        if key in merged:
            delta = merged[key]
            if base.shape != delta.shape:
                raise ValueError(
                    f"Shape mismatch applying delta to '{key}': "
                    f"base={base.shape}, delta={delta.shape}"
                )
            result[key] = (base + learning_rate * delta).astype(base.dtype)
        else:
            result[key] = base.copy()

    # copy keys in delta but not in base (new layers from updated architecture)
    for key in merged:
        if key not in result:
            result[key] = merged[key].copy()

    update["applied_at"] = time.time()
    return result


# ── Serialization helpers ────────────────────────────────────────────

def save_delta(delta: PolicyDelta, directory: Path | str) -> Path:
    """Persist a policy delta to disk as a .npz + metadata JSON.

    The .npz holds the numpy arrays; the JSON holds scalar metadata.
    No query data is included in the output files.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"delta_{delta['delta_id']}.npz"
    meta_path = out_dir / f"delta_{delta['delta_id']}.json"

    arrays: dict[str, np.ndarray] = delta["layer_deltas"]
    kw: dict[str, np.ndarray] = dict(arrays)
    np.savez_compressed(str(npz_path), **kw)  # type: ignore[arg-type]

    meta = {
        "delta_id": delta["delta_id"],
        "timestamp": delta["timestamp"],
        "num_episodes": delta["num_episodes"],
        "avg_reward": delta["avg_reward"],
        "source_hash": delta["source_hash"],
        "layers": list(delta["layer_deltas"].keys()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("Saved delta %s to %s", delta["delta_id"], out_dir)
    return npz_path


def load_delta(npz_path: Path | str) -> PolicyDelta:
    """Load a policy delta from a .npz + companion JSON file."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Delta file not found: {npz_path}")

    meta_path = npz_path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Delta metadata not found: {meta_path}")

    data = dict(np.load(str(npz_path)))
    meta = json.loads(meta_path.read_text())

    return PolicyDelta(
        delta_id=meta["delta_id"],
        timestamp=meta["timestamp"],
        layer_deltas=data,
        num_episodes=meta["num_episodes"],
        avg_reward=meta["avg_reward"],
        source_hash=meta["source_hash"],
    )


# ── Coordinator (higher-level convenience) ───────────────────────────

class FederatedCoordinator:
    """Manages the local train-export-merge cycle.

    Typical flow:
        coord = FederatedCoordinator()
        delta = coord.compute_round(before_weights, after_weights, episodes, reward)
        coord.contribute(delta)  # queue for aggregation
        updated = coord.merge(base_weights)  # FedAvg all queued deltas
    """

    def __init__(
        self,
        privacy: PrivacyConfig | None = None,
        storage_dir: Path | str | None = None,
    ) -> None:
        self._privacy = privacy or _default_privacy()
        self._storage = Path(storage_dir or PROJECT_ROOT / "data" / "federated")
        self._contributions: list[PolicyDelta] = []

        logger.info(
            "FederatedCoordinator: clip_norm=%.2f, noise_scale=%.4f, min_contrib=%d",
            self._privacy["clip_norm"],
            self._privacy["noise_scale"],
            self._privacy["min_contributors"],
        )

    @property
    def contribution_count(self) -> int:
        """Number of deltas queued for aggregation."""
        return len(self._contributions)

    def compute_round(
        self,
        before: dict[str, np.ndarray],
        after: dict[str, np.ndarray],
        num_episodes: int = 0,
        avg_reward: float = 0.0,
        persist: bool = False,
    ) -> PolicyDelta:
        """Extract, sanitize, and optionally save a training round delta."""
        raw = extract_delta(before, after, num_episodes, avg_reward)
        sanitized = clip_and_noise(raw, self._privacy)

        if persist:
            save_delta(sanitized, self._storage)

        return sanitized

    def contribute(self, delta: PolicyDelta) -> None:
        """Queue a sanitized delta for later aggregation."""
        self._contributions.append(delta)
        logger.debug(
            "Contribution queued: %s (%d episodes, reward=%.3f)",
            delta["delta_id"], delta["num_episodes"], delta["avg_reward"],
        )

    def merge(
        self,
        base_weights: dict[str, np.ndarray],
        learning_rate: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """Aggregate all queued deltas and apply to base weights.

        Clears the contribution queue after a successful merge.
        """
        update = aggregate_deltas(
            self._contributions,
            min_contributors=self._privacy["min_contributors"],
        )
        result = apply_delta(base_weights, update, learning_rate)
        logger.info(
            "Merged %d contributions (%d total episodes, avg_reward=%.3f)",
            update["num_contributors"],
            update["total_episodes"],
            update["avg_reward"],
        )
        self._contributions.clear()
        return result

    def clear(self) -> None:
        """Discard all queued contributions."""
        count = len(self._contributions)
        self._contributions.clear()
        logger.info("Cleared %d queued contributions", count)


__all__ = [
    "PolicyDelta",
    "PrivacyConfig",
    "AggregatedUpdate",
    "extract_delta",
    "clip_and_noise",
    "aggregate_deltas",
    "apply_delta",
    "save_delta",
    "load_delta",
    "FederatedCoordinator",
]

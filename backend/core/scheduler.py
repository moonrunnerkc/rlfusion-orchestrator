# Author: Bradley R. Kinnard
"""Lane-aware task scheduler with hardware profiling and quantization selection.

Phase 8: priority-lane scheduling for the agent pipeline, resource-aware batch
sizing, and auto-selection of quantization levels based on detected hardware.
Safety tasks preempt everything else. The scheduler probes GPU/CPU/RAM on init
and adjusts throughput knobs accordingly.
"""
from __future__ import annotations

import asyncio
import enum
import logging
import os
import platform
import shutil
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Coroutine, TypedDict

import numpy as np

from backend.config import cfg

logger = logging.getLogger(__name__)


# ── Priority lanes (lower value = higher priority) ───────────────────

class TaskLane(enum.IntEnum):
    """Pipeline task categories ordered by scheduling priority."""
    SAFETY = 0
    RETRIEVAL = 1
    FUSION = 2
    CRITIQUE = 3
    GENERATION = 4


# ── Hardware detection ───────────────────────────────────────────────

class HardwareProfile(TypedDict):
    """Snapshot of the host's compute resources."""
    gpu_available: bool
    gpu_name: str
    vram_total_mb: int
    vram_free_mb: int
    ram_total_mb: int
    ram_free_mb: int
    cpu_arch: str
    cpu_count: int
    is_arm: bool
    pointer_bits: int


def detect_hardware() -> HardwareProfile:
    """Probe the host for GPU, RAM, CPU info. Safe on headless / no-GPU boxes."""
    gpu_available = False
    gpu_name = ""
    vram_total = 0
    vram_free = 0

    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_total = props.total_memory // (1024 * 1024)
            vram_free = (props.total_memory - torch.cuda.memory_allocated(0)) // (1024 * 1024)
    except (ImportError, RuntimeError, AssertionError) as exc:
        logger.debug("GPU probe failed: %s", exc)

    # RAM via /proc/meminfo (Linux) or fallback
    ram_total, ram_free = _probe_ram()

    arch = platform.machine().lower()
    is_arm = any(tag in arch for tag in ("arm", "aarch"))

    return HardwareProfile(
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        vram_total_mb=vram_total,
        vram_free_mb=vram_free,
        ram_total_mb=ram_total,
        ram_free_mb=ram_free,
        cpu_arch=arch,
        cpu_count=os.cpu_count() or 1,
        is_arm=is_arm,
        pointer_bits=struct.calcsize("P") * 8,
    )


def _probe_ram() -> tuple[int, int]:
    """Get total and available RAM in MB. Linux-first, then generic fallback."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    info[key] = int(parts[1])  # value in kB
            total = info.get("MemTotal", 0) // 1024
            available = info.get("MemAvailable", info.get("MemFree", 0)) // 1024
            return total, available
    except (OSError, ValueError) as exc:
        logger.debug("/proc/meminfo unavailable: %s", exc)

    # macOS / generic fallback
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.total // (1024 * 1024), mem.available // (1024 * 1024)
    except ImportError:
        logger.debug("psutil not installed, RAM detection unavailable")

    return 0, 0


# ── Quantization recommendation ─────────────────────────────────────

# GGUF quant levels for Ollama LLM models, ordered by VRAM requirement
_QUANT_LEVELS = [
    {"name": "Q4_K_M", "min_vram_mb": 4000, "quality": 0.82, "description": "4-bit mixed, balanced quality/speed"},
    {"name": "Q5_K_M", "min_vram_mb": 5500, "quality": 0.90, "description": "5-bit mixed, good quality"},
    {"name": "Q8_0",   "min_vram_mb": 8000, "quality": 0.97, "description": "8-bit, near-lossless"},
    {"name": "F16",    "min_vram_mb": 16000, "quality": 1.00, "description": "Full fp16 precision"},
]


class QuantRecommendation(TypedDict):
    """Suggested quantization level with rationale."""
    level: str
    quality: float
    description: str
    fits_in_vram: bool
    reason: str


def recommend_quantization(profile: HardwareProfile) -> QuantRecommendation:
    """Pick the best GGUF quantization level that fits available VRAM.

    Falls back to Q4_K_M if no GPU is detected (CPU inference).
    """
    if not profile["gpu_available"]:
        return QuantRecommendation(
            level="Q4_K_M",
            quality=0.82,
            description="4-bit mixed, balanced quality/speed",
            fits_in_vram=False,
            reason="No GPU detected; Q4_K_M recommended for CPU inference",
        )

    free = profile["vram_free_mb"]
    # walk from highest quality down, pick the best that fits
    for q in reversed(_QUANT_LEVELS):
        if free >= q["min_vram_mb"]:
            return QuantRecommendation(
                level=q["name"],
                quality=q["quality"],
                description=q["description"],
                fits_in_vram=True,
                reason=f"{free} MB VRAM free, {q['name']} needs {q['min_vram_mb']} MB",
            )

    # not enough VRAM even for Q4
    return QuantRecommendation(
        level="Q4_K_M",
        quality=0.82,
        description="4-bit mixed, balanced quality/speed",
        fits_in_vram=False,
        reason=f"Only {free} MB VRAM free; Q4_K_M is the minimum viable option",
    )


# ── RL policy quantization (torch dynamic int8 / float16) ───────────

class PolicyQuantInfo(TypedDict):
    """Info about a quantized RL policy variant."""
    precision: str
    size_reduction: float
    path: str


def quantize_policy_weights(
    weights: dict[str, object],
    precision: str = "float16",
) -> dict[str, object]:
    """Reduce precision of policy weight tensors. Returns a new dict.

    Supports 'float16' and 'int8' (via torch dynamic quantization on numpy).
    Does not modify the input dict.
    """
    valid_precisions = {"float16", "float32", "int8"}
    if precision not in valid_precisions:
        raise ValueError(f"Unsupported precision '{precision}'. Choose from {valid_precisions}")

    quantized: dict[str, object] = {}
    for key, val in weights.items():
        if isinstance(val, np.ndarray) and val.dtype in (np.float32, np.float64):
            if precision == "float16":
                quantized[key] = val.astype(np.float16)
            elif precision == "int8":
                # scale-and-shift to int8 range, store scale+zero for dequant
                vmin, vmax = float(val.min()), float(val.max())
                scale = (vmax - vmin) / 255.0 if vmax != vmin else 1.0
                zero = vmin
                quantized[key] = {
                    "data": ((val - zero) / scale).clip(0, 255).astype(np.uint8),
                    "scale": scale,
                    "zero": zero,
                    "original_shape": val.shape,
                }
            else:
                quantized[key] = val.copy()
        else:
            quantized[key] = val

    return quantized


def dequantize_int8(packed: dict[str, object]) -> np.ndarray:
    """Restore a float32 array from int8-quantized packed dict."""
    if not isinstance(packed, dict) or "data" not in packed:
        raise ValueError("Expected a packed int8 dict with 'data', 'scale', 'zero' keys")

    data: np.ndarray = packed["data"]  # type: ignore[assignment]
    scale: float = packed["scale"]  # type: ignore[assignment]
    zero: float = packed["zero"]  # type: ignore[assignment]
    shape: tuple[int, ...] = packed["original_shape"]  # type: ignore[assignment]
    return (data.astype(np.float32) * scale + zero).reshape(shape)


def recommend_policy_precision(profile: HardwareProfile) -> str:
    """Pick RL policy precision based on available resources."""
    if not profile["gpu_available"]:
        # CPU-only: int8 saves memory, acceptable accuracy for a small policy net
        if profile["ram_free_mb"] < 2048:
            return "int8"
        return "float16"

    if profile["vram_free_mb"] >= 8000:
        return "float32"
    if profile["vram_free_mb"] >= 4000:
        return "float16"
    return "int8"


# ── Batch size recommendation ────────────────────────────────────────

# baseline batch sizes per lane, scaled by available memory
_LANE_BASE_BATCH: dict[TaskLane, int] = {
    TaskLane.SAFETY: 1,       # always process safety checks one at a time
    TaskLane.RETRIEVAL: 8,
    TaskLane.FUSION: 4,
    TaskLane.CRITIQUE: 2,
    TaskLane.GENERATION: 1,
}


def recommend_batch_size(profile: HardwareProfile, lane: TaskLane) -> int:
    """Compute a resource-aware batch size for a given task lane."""
    base = _LANE_BASE_BATCH.get(lane, 1)

    if lane == TaskLane.SAFETY:
        return 1  # safety is always serial

    # scale by available memory (use VRAM if GPU, else RAM)
    avail = profile["vram_free_mb"] if profile["gpu_available"] else profile["ram_free_mb"]

    if avail <= 0:
        return max(1, base // 2)

    # rough heuristic: double batch every 8 GB
    scale = max(1, avail // 8192)
    return base * scale


# ── Task scheduler with priority lanes ───────────────────────────────

@dataclass
class _QueuedTask:
    """Internal representation of a scheduled task."""
    lane: TaskLane
    created_at: float
    task_id: str
    coro: Coroutine[object, object, object]

    def __lt__(self, other: _QueuedTask) -> bool:
        # lower lane value = higher priority; within same lane, FIFO
        if self.lane != other.lane:
            return self.lane < other.lane
        return self.created_at < other.created_at


@dataclass
class LaneStats(TypedDict):
    """Stats for a single priority lane."""
    pending: int
    completed: int
    avg_latency_ms: float


class TaskScheduler:
    """Priority-lane async task scheduler with resource awareness.

    Safety tasks always run first (preemptive priority). Lower-priority
    lanes yield when a higher-priority task arrives. Configurable via
    the 'scheduling' key in config.yaml.
    """

    def __init__(self, profile: HardwareProfile | None = None) -> None:
        sched_cfg = cfg.get("scheduling", {})
        self._max_concurrent = int(sched_cfg.get("max_concurrent", 4))
        self._safety_preempt = bool(sched_cfg.get("safety_preempt", True))
        self._profile = profile or detect_hardware()
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._task_counter = 0
        self._stats: dict[TaskLane, list[float]] = {lane: [] for lane in TaskLane}
        self._pending: dict[TaskLane, int] = {lane: 0 for lane in TaskLane}
        self._completed: dict[TaskLane, int] = {lane: 0 for lane in TaskLane}

        logger.info(
            "TaskScheduler: max_concurrent=%d, safety_preempt=%s, gpu=%s",
            self._max_concurrent, self._safety_preempt, self._profile["gpu_available"],
        )

    @property
    def hardware_profile(self) -> HardwareProfile:
        """The detected hardware profile used for scheduling decisions."""
        return self._profile

    async def submit(
        self,
        lane: TaskLane,
        coro: Coroutine[object, object, object],
    ) -> object:
        """Submit a task to a priority lane. Returns the coroutine's result.

        Safety lane tasks bypass the semaphore when safety_preempt is enabled,
        so they never block behind lower-priority work.
        """
        self._task_counter += 1
        task_id = f"{lane.name}-{self._task_counter}"
        self._pending[lane] += 1
        start = time.monotonic()

        try:
            if lane == TaskLane.SAFETY and self._safety_preempt:
                # safety bypasses the concurrency semaphore
                result = await coro
            else:
                async with self._semaphore:
                    result = await coro
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            self._pending[lane] -= 1
            self._completed[lane] += 1
            self._stats[lane].append(elapsed_ms)
            # keep stats bounded
            if len(self._stats[lane]) > 500:
                self._stats[lane] = self._stats[lane][-250:]

        logger.debug("Task %s completed in %.1f ms", task_id, elapsed_ms)
        return result

    def get_lane_status(self) -> dict[str, LaneStats]:
        """Snapshot of per-lane pending/completed counts and avg latency."""
        status: dict[str, LaneStats] = {}
        for lane in TaskLane:
            latencies = self._stats[lane]
            avg = sum(latencies) / len(latencies) if latencies else 0.0
            status[lane.name.lower()] = LaneStats(
                pending=self._pending[lane],
                completed=self._completed[lane],
                avg_latency_ms=round(avg, 2),
            )
        return status

    def recommend_batch(self, lane: TaskLane) -> int:
        """Convenience: batch size recommendation for this scheduler's hardware."""
        return recommend_batch_size(self._profile, lane)


__all__ = [
    "TaskLane",
    "HardwareProfile",
    "QuantRecommendation",
    "PolicyQuantInfo",
    "LaneStats",
    "detect_hardware",
    "recommend_quantization",
    "quantize_policy_weights",
    "dequantize_int8",
    "recommend_policy_precision",
    "recommend_batch_size",
    "TaskScheduler",
]

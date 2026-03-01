# Author: Bradley R. Kinnard
"""VRAM/RAM monitoring gauges and structured JSON logging.

Wraps pynvml for GPU memory and psutil for system RAM. Designed to run
as a background task logging every N seconds, parsed with jq in production.
"""
from __future__ import annotations

import json
import logging
import os
import time
import threading
from dataclasses import asdict, dataclass, field
from typing import TypedDict

logger = logging.getLogger(__name__)


class MemorySnapshot(TypedDict):
    """Single point-in-time memory reading."""
    timestamp: float
    vram_used_mb: float
    vram_total_mb: float
    vram_pct: float
    ram_used_mb: float
    ram_total_mb: float
    ram_pct: float


@dataclass
class MemoryMonitor:
    """Periodically logs VRAM and RAM usage as structured JSON.

    Safe to call even without a GPU or pynvml installed: gracefully
    falls back to ram-only metrics.
    """
    interval_seconds: int = 10
    _running: bool = field(default=False, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _last_snapshot: MemorySnapshot | None = field(default=None, init=False)

    def _read_vram(self) -> tuple[float, float]:
        """Read VRAM usage via pynvml. Returns (used_mb, total_mb)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = mem.used / (1024 * 1024)
            total = mem.total / (1024 * 1024)
            pynvml.nvmlShutdown()
            return used, total
        except Exception:
            return 0.0, 0.0

    def _read_ram(self) -> tuple[float, float]:
        """Read system RAM via /proc/meminfo (no psutil dependency)."""
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            mem = {}
            for line in lines[:5]:
                parts = line.split()
                mem[parts[0].rstrip(":")] = int(parts[1])
            total = mem.get("MemTotal", 0) / 1024
            available = mem.get("MemAvailable", mem.get("MemFree", 0)) / 1024
            used = total - available
            return used, total
        except Exception:
            return 0.0, 0.0

    def snapshot(self) -> MemorySnapshot:
        """Take a single memory reading."""
        vram_used, vram_total = self._read_vram()
        ram_used, ram_total = self._read_ram()
        snap: MemorySnapshot = {
            "timestamp": time.time(),
            "vram_used_mb": round(vram_used, 1),
            "vram_total_mb": round(vram_total, 1),
            "vram_pct": round(vram_used / max(vram_total, 1) * 100, 1),
            "ram_used_mb": round(ram_used, 1),
            "ram_total_mb": round(ram_total, 1),
            "ram_pct": round(ram_used / max(ram_total, 1) * 100, 1),
        }
        self._last_snapshot = snap
        return snap

    @property
    def last(self) -> MemorySnapshot | None:
        """Most recent snapshot without triggering a new read."""
        return self._last_snapshot

    def _monitor_loop(self) -> None:
        """Background loop: snapshot + structured log every interval."""
        while self._running:
            snap = self.snapshot()
            logger.info(
                "memory.gauge",
                extra={"memory": json.dumps(snap)},
            )
            time.sleep(self.interval_seconds)

    def start(self) -> None:
        """Start background monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="mem-monitor",
        )
        self._thread.start()
        logger.info("Memory monitor started (interval=%ds)", self.interval_seconds)

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.interval_seconds + 1)
            self._thread = None
        logger.info("Memory monitor stopped")


# Module-level singleton
_monitor: MemoryMonitor | None = None


def get_monitor(interval: int = 10) -> MemoryMonitor:
    """Get or create the singleton MemoryMonitor."""
    global _monitor
    if _monitor is None:
        _monitor = MemoryMonitor(interval_seconds=interval)
    return _monitor

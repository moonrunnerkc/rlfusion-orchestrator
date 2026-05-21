# Author: Bradley R. Kinnard
# sim_env.py - batch runner for FusionEnv stress testing

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message=".*Gym.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import subprocess
import psutil
import time
import numpy as np
import json
import argparse
import statistics
from typing import List, Dict, Any, Optional


def locate_fusion_env_import_path() -> str:
    candidates = ["backend.rl.fusion_env", "rl.fusion_env", "env.fusion_env", "backend.fusion_env"]
    for c in candidates:
        try:
            __import__(c, fromlist=["FusionEnv"])
            return c
        except Exception:
            continue
    raise ImportError(f"Could not locate FusionEnv - tried: {', '.join(candidates)}")


def get_gpu_stats() -> Dict[str, Any]:
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"
        ], stderr=subprocess.DEVNULL)
        line = out.decode("utf-8").strip().splitlines()[0]
        used, total, util = [int(x.strip()) for x in line.split(",")]
        return {"memory_used_mb": used, "memory_total_mb": total, "utilization": util}
    except Exception:
        return {}


def _worker_task(task: Dict[str, Any]) -> Dict[str, Any]:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    mod = __import__(task["module_path"], fromlist=["FusionEnv"])
    FusionEnv = getattr(mod, "FusionEnv")

    worker_id = task["worker_id"]
    mode = task["mode"]
    iterations = task.get("iterations", 1)
    duration = task.get("duration", 10)
    query = task.get("query", "What is RLFusion Orchestrator?")
    noise = task.get("noise")
    noise_level = float(task.get("noise_level", 0.1))

    proc = psutil.Process()
    latencies, responses, rewards = [], [], []
    errors = 0
    samples_gpu = []
    deadline = time.time() + duration if mode == "stress" else None

    def _inject_noise(q: str) -> str:
        if noise == "adversarial":
            return q + " " + "X" * max(1, int(10 * noise_level))
        if noise == "drift":
            return q.replace("a", "e")[:max(1, int(len(q) * (1 - 0.01 * noise_level)))]
        return q

    env = FusionEnv()
    loop_count = 0

    try:
        while True:
            if mode == "batch" and loop_count >= iterations:
                break
            if mode == "stress" and time.time() >= (deadline or 0):
                break

            q = _inject_noise(query)
            t0 = time.perf_counter()
            try:
                obs, _ = env.reset(options={"query": q})
                action = np.random.uniform(-1.0, 1.0, size=env.action_space.shape)
                _obs, reward, done, truncated, info = env.step(action)
                rewards.append(float(reward))
                if "response" in info:
                    responses.append(info["response"])
            except Exception:
                errors += 1
            latencies.append((time.perf_counter() - t0) * 1000.0)
            samples_gpu.append(get_gpu_stats())
            loop_count += 1
    finally:
        rss = proc.memory_info().rss // (1024 * 1024)
        vm = psutil.virtual_memory()
        return {
            "worker_id": worker_id, "requests": loop_count, "latencies_ms": latencies,
            "errors": errors, "rss_mb": rss, "system_ram_mb": vm.used // (1024 * 1024),
            "gpu_samples": samples_gpu, "responses": responses, "rewards": rewards,
        }


class SimEnvRunner:
    def __init__(self, concurrency: int = 4, module_path: Optional[str] = None):
        self.concurrency = max(1, concurrency)
        self.module_path = module_path or locate_fusion_env_import_path()

    def run_batch(self, total_requests: int = 100, iterations_per_worker: int = 10, **kwargs) -> Dict[str, Any]:
        workers = min(self.concurrency, total_requests)
        tasks = [{"module_path": self.module_path, "worker_id": i, "mode": "batch",
                  "iterations": max(1, iterations_per_worker), **kwargs} for i in range(workers)]
        with mp.Pool(processes=workers) as pool:
            results = pool.map(_worker_task, tasks)
        return self._aggregate(results)

    def run_stress(self, duration: int = 60, **kwargs) -> Dict[str, Any]:
        tasks = [{"module_path": self.module_path, "worker_id": i, "mode": "stress",
                  "duration": duration, **kwargs} for i in range(self.concurrency)]
        with mp.Pool(processes=self.concurrency) as pool:
            results = pool.map(_worker_task, tasks)
        return self._aggregate(results)

    def _aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_latencies, all_responses, all_rewards = [], [], []
        total_requests, total_errors, max_rss = 0, 0, 0
        system_ram, gpu_mem, gpu_util = [], [], []

        for r in results:
            total_requests += r.get("requests", 0)
            total_errors += r.get("errors", 0)
            max_rss = max(max_rss, r.get("rss_mb", 0))
            system_ram.append(r.get("system_ram_mb", 0))
            all_latencies.extend(r.get("latencies_ms", []))
            all_responses.extend(r.get("responses", []))
            all_rewards.extend(r.get("rewards", []))
            for g in r.get("gpu_samples", []):
                if g.get("memory_used_mb"):
                    gpu_mem.append(g["memory_used_mb"])
                if g.get("utilization"):
                    gpu_util.append(g["utilization"])

        return {
            "total_requests": total_requests, "total_errors": total_errors, "max_rss_mb": max_rss,
            "system_ram_used_mb_avg": statistics.mean(system_ram) if system_ram else None,
            "latency_ms_avg": statistics.mean(all_latencies) if all_latencies else None,
            "latency_ms_p50": np.percentile(all_latencies, 50).item() if all_latencies else None,
            "latency_ms_p95": np.percentile(all_latencies, 95).item() if all_latencies else None,
            "gpu_mem_mb_avg": statistics.mean(gpu_mem) if gpu_mem else None,
            "gpu_util_pct_avg": statistics.mean(gpu_util) if gpu_util else None,
            "responses": all_responses, "rewards": all_rewards,
            "avg_reward": statistics.mean(all_rewards) if all_rewards else None,
        }


class BenchmarkEnv:
    def __init__(self, module_path: Optional[str] = None):
        self.module_path = module_path or locate_fusion_env_import_path()
        mod = __import__(self.module_path, fromlist=["FusionEnv"])
        self.FusionEnv = getattr(mod, "FusionEnv")

    def batch_step(self, queries: List[str], noise_level: float = 0.0) -> Dict[str, Any]:
        if not queries:
            return {"total_requests": 0, "total_errors": 0}

        tasks = [{"module_path": self.module_path, "worker_id": i, "mode": "batch", "iterations": 1,
                  "query": q, "noise": "adversarial" if noise_level > 0 else None,
                  "noise_level": noise_level} for i, q in enumerate(queries)]

        with mp.Pool(processes=min(len(queries), mp.cpu_count())) as pool:
            results = pool.map(_worker_task, tasks)

        all_latencies, all_responses = [], []
        total_requests, total_errors, max_rss = 0, 0, 0
        system_ram, gpu_mem, gpu_util = [], [], []

        for r in results:
            total_requests += r.get("requests", 0)
            total_errors += r.get("errors", 0)
            max_rss = max(max_rss, r.get("rss_mb", 0))
            system_ram.append(r.get("system_ram_mb", 0))
            all_latencies.extend(r.get("latencies_ms", []))
            all_responses.extend(r.get("responses", []))
            for g in r.get("gpu_samples", []):
                if g.get("memory_used_mb"):
                    gpu_mem.append(g["memory_used_mb"])
                if g.get("utilization"):
                    gpu_util.append(g["utilization"])

        return {
            "total_requests": total_requests, "total_errors": total_errors, "max_rss_mb": max_rss,
            "system_ram_used_mb_avg": statistics.mean(system_ram) if system_ram else None,
            "latency_ms_avg": statistics.mean(all_latencies) if all_latencies else None,
            "latency_ms_p50": np.percentile(all_latencies, 50).item() if all_latencies else None,
            "latency_ms_p95": np.percentile(all_latencies, 95).item() if all_latencies else None,
            "latency_ms_p99": np.percentile(all_latencies, 99).item() if all_latencies else None,
            "gpu_mem_mb_avg": statistics.mean(gpu_mem) if gpu_mem else None,
            "gpu_util_pct_avg": statistics.mean(gpu_util) if gpu_util else None,
            "responses": all_responses,
        }


def run_benchmark(queries: List[str], noise: float = 0.0, parallelism: int = 8) -> dict:
    if not queries:
        return {"total_requests": 0, "total_errors": 0, "error": "No queries"}
    bench = BenchmarkEnv()
    metrics = bench.batch_step(queries, noise_level=noise)
    metrics["parallelism_used"] = min(parallelism, len(queries), mp.cpu_count())
    metrics["num_queries"] = len(queries)
    return metrics


def _cli():
    p = argparse.ArgumentParser(prog="sim_env")
    p.add_argument("--mode", choices=["batch", "stress"], default="batch")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--query", type=str, default="What is RLFusion Orchestrator?")
    p.add_argument("--noise", choices=["none", "adversarial", "drift"], default="none")
    p.add_argument("--noise-level", type=float, default=0.1)
    p.add_argument("--module-path", type=str, default=None)
    args = p.parse_args()

    runner = SimEnvRunner(concurrency=args.concurrency, module_path=args.module_path)
    noise_arg = None if args.noise == "none" else args.noise
    if args.mode == "batch":
        metrics = runner.run_batch(total_requests=args.concurrency * args.iterations,
                                   iterations_per_worker=args.iterations, query=args.query,
                                   noise=noise_arg, noise_level=args.noise_level)
    else:
        metrics = runner.run_stress(duration=args.duration, query=args.query,
                                    noise=noise_arg, noise_level=args.noise_level)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    _cli()

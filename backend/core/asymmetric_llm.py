# Author: Bradley R. Kinnard
"""Asymmetric dual-model LLM orchestrator.

Pins Qwen 2.5 1.5B (Q4_K_M) to CPU for triage tasks and Llama 3.1 8B (Q8_0)
to GPU for generation and critique. Both models loaded once at startup in a
single process, zero inter-process communication.

CPU worker handles: intent parsing, safety checks, CAG orchestration,
  GraphRAG entity extraction, observation building.
GPU executor handles: generation, critique, faithfulness verification.

Note: route_task() is currently not wired into the live request path.
The orchestrator routes work through fusion_agent/critique_agent/main.py
directly; this class is loaded but reserved for the Phase 5 routing wire-up.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Literal

from backend.config import PROJECT_ROOT, get_inference_config

logger = logging.getLogger(__name__)

# rough token-to-char ratio for context window enforcement
_CHARS_PER_TOKEN = 4

TaskType = Literal[
    "intent_parse",
    "safety_check",
    "cag_lookup",
    "graph_trigger",
    "obs_build",
    "generation",
    "critique",
    "faithfulness",
]

_CPU_TASKS: frozenset[str] = frozenset(
    {
        "intent_parse",
        "safety_check",
        "cag_lookup",
        "graph_trigger",
        "obs_build",
    }
)

_GPU_TASKS: frozenset[str] = frozenset(
    {
        "generation",
        "critique",
        "faithfulness",
    }
)


class AsymmetricLLMOrchestrator:
    """Singleton dual-model orchestrator. CPU triage + GPU executor.

    Thread-safe via per-model locks. Models loaded lazily on first call
    to avoid import-time side effects. Config-driven paths and context sizes.
    """

    _instance: AsymmetricLLMOrchestrator | None = None
    _init_lock = threading.Lock()

    def __new__(cls) -> AsymmetricLLMOrchestrator:
        # F1.10: the double-checked-lock pattern only stays safe when both
        # reads happen under the lock. The outer read is a fast path; the
        # inner check must re-verify because another thread could have
        # constructed in the meantime.
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            self._initialized = True

        inf = get_inference_config()
        self._cpu_model_path = str(PROJECT_ROOT / inf["cpu_model_path"])
        self._gpu_model_path = str(PROJECT_ROOT / inf["gpu_model_path"])
        self._cpu_ctx = int(inf.get("cpu_ctx_size", 8192))
        self._gpu_ctx = int(inf.get("gpu_ctx_size", 8192))
        self._seed = int(inf.get("seed", 42))

        self._cpu_worker = None
        self._gpu_executor = None
        self._cpu_lock = threading.Lock()
        self._gpu_lock = threading.Lock()
        self._gpu_offloaded = False
        # F1.8: after this many CPU-fallback turns we attempt a GPU reload.
        self._oom_recovery_counter = 0
        self._oom_recovery_threshold = 5

        logger.info(
            "AsymmetricLLMOrchestrator configured: cpu=%s, gpu=%s",
            Path(self._cpu_model_path).name,
            Path(self._gpu_model_path).name,
        )

    def _ensure_cpu(self) -> None:
        """Load CPU triage model on first use. Runs entirely in system RAM."""
        if self._cpu_worker is not None:
            return
        from llama_cpp import Llama

        logger.info("Loading CPU triage model: %s", Path(self._cpu_model_path).name)
        if not Path(self._cpu_model_path).exists():
            raise FileNotFoundError(
                f"CPU model not found: {self._cpu_model_path}. "
                "Run: huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF "
                "qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models/"
            )
        self._cpu_worker = Llama(
            model_path=self._cpu_model_path,
            n_gpu_layers=0,
            n_ctx=self._cpu_ctx,
            n_batch=512,
            seed=self._seed,
            chat_format="chatml",
            verbose=False,
        )
        logger.info("CPU triage model loaded (RAM only, 0 GPU layers)")

    def _ensure_gpu(self, force_reload: bool = False) -> None:
        """Load GPU executor model on first use. All layers pinned to VRAM.

        Pass force_reload=True from the OOM-recovery path to drop the existing
        Llama handle and rehydrate against the freshest VRAM availability.
        """
        if force_reload:
            self._gpu_executor = None
        if self._gpu_executor is not None:
            return
        from llama_cpp import Llama

        n_layers = -1 if not self._gpu_offloaded else 0
        logger.info(
            "Loading GPU executor model: %s (n_gpu_layers=%d)",
            Path(self._gpu_model_path).name,
            n_layers,
        )
        if not Path(self._gpu_model_path).exists():
            raise FileNotFoundError(
                f"GPU model not found: {self._gpu_model_path}. "
                "Run: huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF "
                "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf --local-dir models/"
            )
        self._gpu_executor = Llama(
            model_path=self._gpu_model_path,
            n_gpu_layers=n_layers,
            n_ctx=self._gpu_ctx,
            n_batch=512,
            seed=self._seed,
            chat_format="llama-3",
            verbose=False,
        )
        mode = "CPU fallback" if self._gpu_offloaded else "full GPU"
        logger.info("GPU executor model loaded (%s)", mode)

    def triage(
        self,
        prompt: str,
        *,
        system: str = "You are a precise assistant. Output only JSON when asked.",
        temperature: float = 0.1,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:
        """Run a triage task on the CPU worker. Fast structured output.

        Returns raw text. If json_mode=True, attempts JSON repair on output.
        Truncates prompt to fit within CPU context window.
        """
        prompt = self._enforce_ctx(prompt, self._cpu_ctx, max_tokens)
        with self._cpu_lock:
            self._ensure_cpu()
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            response = self._cpu_worker.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=self._seed,
            )
        text = response["choices"][0]["message"]["content"] or ""

        if json_mode:
            text = self._repair_json(text, prompt)
        return text

    def execute(
        self,
        prompt: str,
        *,
        system: str = "You are a helpful, accurate assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Run a generation task on the GPU executor.

        Truncates prompt to fit within GPU context window.
        """
        prompt = self._enforce_ctx(prompt, self._gpu_ctx, max_tokens)
        with self._gpu_lock:
            # F1.8: if we are in CPU-fallback mode, try to come back to GPU
            # after _oom_recovery_threshold successful CPU-fallback turns.
            if (
                self._gpu_offloaded
                and self._oom_recovery_counter >= self._oom_recovery_threshold
            ):
                logger.info("Attempting GPU recovery after CPU fallback streak")
                self._gpu_offloaded = False
                self._oom_recovery_counter = 0
                try:
                    self._ensure_gpu(force_reload=True)
                except Exception as exc:
                    logger.warning("GPU recovery failed, staying on CPU: %s", exc)
                    self._gpu_offloaded = True

            self._ensure_gpu()
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            try:
                response = self._gpu_executor.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=self._seed,
                )
            except Exception as exc:
                if "out of memory" in str(exc).lower() or "CUDA" in str(exc):
                    return self._handle_oom(messages, temperature, max_tokens)
                raise

            if self._gpu_offloaded:
                self._oom_recovery_counter += 1
        return response["choices"][0]["message"]["content"] or ""

    def stream_execute(
        self,
        prompt: str,
        *,
        system: str = "You are a helpful, accurate assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """Stream generation from the GPU executor. Yields token chunks."""
        with self._gpu_lock:
            self._ensure_gpu()
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            for chunk in self._gpu_executor.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=self._seed,
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {})
                if token := delta.get("content"):
                    yield token

    def route_task(self, task_type: str, prompt: str, **kwargs: object) -> str:
        """Route a task to the correct worker based on type.

        CPU tasks: intent_parse, safety_check, cag_lookup, graph_trigger, obs_build
        GPU tasks: generation, critique, faithfulness
        """
        if task_type in _CPU_TASKS:
            json_mode = task_type in {"intent_parse", "graph_trigger", "obs_build"}
            return self.triage(prompt, json_mode=json_mode, **kwargs)
        if task_type in _GPU_TASKS:
            return self.execute(prompt, **kwargs)
        raise ValueError(
            f"Unknown task type: {task_type!r}. "
            f"Valid: {sorted(_CPU_TASKS | _GPU_TASKS)}"
        )

    def _enforce_ctx(self, prompt: str, ctx_size: int, max_tokens: int) -> str:
        """Truncate prompt to fit within context window, reserving max_tokens for output."""
        budget = (
            ctx_size - max_tokens - 128
        ) * _CHARS_PER_TOKEN  # 128 tokens for system/template overhead
        if len(prompt) <= budget:
            return prompt
        logger.warning(
            "Prompt truncated: %d chars -> %d chars (ctx=%d, reserve=%d)",
            len(prompt),
            budget,
            ctx_size,
            max_tokens,
        )
        return prompt[:budget]

    def _repair_json(self, text: str, original_prompt: str) -> str:
        """Attempt JSON repair on CPU triage output. Retry once if needed."""
        from json_repair import repair_json

        try:
            repaired = repair_json(text)
            json.loads(repaired)
            return repaired
        except (json.JSONDecodeError, ValueError):
            pass

        # one retry with corrective prompt
        logger.warning("JSON repair failed, retrying with corrective prompt")
        retry_prompt = f"Fix this to valid JSON only:\n{text}"
        with self._cpu_lock:
            response = self._cpu_worker.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "Output only valid JSON. No explanation.",
                    },
                    {"role": "user", "content": retry_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
                seed=self._seed,
            )
        retry_text = response["choices"][0]["message"]["content"] or ""
        try:
            repaired = repair_json(retry_text)
            json.loads(repaired)
            return repaired
        except (json.JSONDecodeError, ValueError):
            logger.error(
                "JSON repair failed after retry. Returning raw: %s", text[:200]
            )
            return text

    def _handle_oom(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """OOM fallback: offload GPU model to CPU temporarily."""
        logger.error("GPU OOM detected. Falling back to CPU-only execution.")
        self._gpu_offloaded = True
        self._gpu_executor = None
        self._ensure_gpu()
        response = self._gpu_executor.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=self._seed,
        )
        return response["choices"][0]["message"]["content"] or ""

    def health_check(self) -> dict[str, bool | str]:
        """Report model load status and memory footprint."""
        status: dict[str, bool | str] = {
            "cpu_loaded": self._cpu_worker is not None,
            "gpu_loaded": self._gpu_executor is not None,
            "gpu_offloaded": self._gpu_offloaded,
        }
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            status["vram_used_mb"] = f"{mem.used / 1024 / 1024:.0f}"
            status["vram_total_mb"] = f"{mem.total / 1024 / 1024:.0f}"
            pynvml.nvmlShutdown()
        except Exception:
            status["vram_used_mb"] = "unavailable"
            status["vram_total_mb"] = "unavailable"
        return status


def get_orchestrator() -> AsymmetricLLMOrchestrator:
    """Module-level factory. Returns the singleton orchestrator."""
    return AsymmetricLLMOrchestrator()

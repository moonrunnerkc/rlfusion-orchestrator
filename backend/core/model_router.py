# Author: Bradley R. Kinnard
"""Multi-engine LLM router with MoE-style model selection.

Abstracts LLM inference behind a unified interface supporting Ollama (local dev),
vLLM, and TensorRT-LLM (production). The engine is selected via config, and all
call sites use generate() or stream() instead of touching SDK clients directly.

The ModelRouter class handles MoE task-to-model routing. The InferenceEngine
handles the actual LLM call, keyed by `inference.engine` in config.yaml.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Generator
from typing import Literal, TypedDict

from backend.config import cfg, get_inference_config

logger = logging.getLogger(__name__)

# Task types produced by decomposer + internal pipeline stages
TaskType = Literal[
    "explain", "compare", "troubleshoot", "list", "design", "summarize",
    "critique", "retrieval", "generation", "decomposition",
]

_ALL_TASK_TYPES: list[str] = [
    "explain", "compare", "troubleshoot", "list", "design", "summarize",
    "critique", "retrieval", "generation", "decomposition",
]


class ModelEntry(TypedDict):
    """Single model in the pool with task routing metadata."""
    name: str
    task_types: list[str]
    priority: int


# ---------------------------------------------------------------------------
# Inference engine abstraction
# ---------------------------------------------------------------------------

class InferenceEngine:
    """Unified LLM interface. Delegates to Ollama, vLLM, or TensorRT-LLM.

    All call sites interact through generate() (blocking) and stream()
    (yields token chunks). Engine selection is config-driven, not code-driven.
    """

    def __init__(self) -> None:
        inf = get_inference_config()
        self._engine: str = str(inf["engine"])
        self._base_url: str = str(inf["base_url"])
        self._model: str = str(inf["model"])
        self._timeout: int = int(inf["timeout_secs"])
        self._api_key: str = str(inf.get("openai_api_key", ""))
        logger.info(
            "InferenceEngine: engine=%s, base_url=%s, model=%s",
            self._engine, self._base_url, self._model,
        )

    @property
    def engine(self) -> str:
        return self._engine

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        num_ctx: int = 4096,
        num_predict: int | None = None,
        timeout: float | None = None,
        images: list[str] | None = None,
    ) -> str:
        """Blocking LLM call. Returns the full response text."""
        model = model or self._model
        timeout = timeout or self._timeout

        if self._engine == "ollama":
            return self._ollama_generate(
                messages, model=model, temperature=temperature,
                num_ctx=num_ctx, num_predict=num_predict,
                timeout=timeout, images=images,
            )
        # vLLM and TensorRT-LLM both expose OpenAI-compatible endpoints
        return self._openai_generate(
            messages, model=model, temperature=temperature,
            max_tokens=num_predict, timeout=timeout,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        num_ctx: int = 4096,
        num_predict: int | None = None,
    ) -> Generator[str, None, None]:
        """Streaming LLM call. Yields text chunks as they arrive."""
        model = model or self._model

        if self._engine == "ollama":
            yield from self._ollama_stream(
                messages, model=model, temperature=temperature,
                num_ctx=num_ctx, num_predict=num_predict,
            )
        else:
            yield from self._openai_stream(
                messages, model=model, temperature=temperature,
                max_tokens=num_predict,
            )

    # -- Ollama backend -------------------------------------------------------

    def _ollama_generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        num_ctx: int,
        num_predict: int | None,
        timeout: float,
        images: list[str] | None = None,
    ) -> str:
        from ollama import Client, Options

        client = Client(host=self._base_url, timeout=timeout)
        opts = Options(temperature=temperature, num_ctx=num_ctx)
        if num_predict is not None:
            opts["num_predict"] = num_predict

        # handle vision model calls with images
        payload_msgs: list[dict[str, object]] = list(messages)  # type: ignore[arg-type]
        if images:
            payload_msgs = [
                {**m, "images": images} if m.get("role") == "user" else m  # type: ignore[union-attr]
                for m in payload_msgs
            ]

        resp = client.chat(model=model, messages=payload_msgs, options=opts)  # type: ignore[arg-type]
        return resp["message"]["content"]

    def _ollama_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        num_ctx: int,
        num_predict: int | None,
    ) -> Generator[str, None, None]:
        from ollama import Client, Options

        client = Client(host=self._base_url)
        opts = Options(temperature=temperature, num_ctx=num_ctx)
        if num_predict is not None:
            opts["num_predict"] = num_predict

        for chunk in client.chat(model=model, messages=messages, options=opts, stream=True):  # type: ignore[arg-type]
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

    # -- OpenAI-compatible backend (vLLM, TensorRT-LLM) -----------------------

    def _openai_generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int | None,
        timeout: float,
    ) -> str:
        import httpx

        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/v1/chat/completions"
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _openai_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int | None,
    ) -> Generator[str, None, None]:
        import httpx

        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/v1/chat/completions"
        with httpx.stream("POST", url, json=payload, headers=headers, timeout=self._timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                import json
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content

    # -- Health checks --------------------------------------------------------

    def check_health(self) -> bool:
        """Verify the inference engine is reachable and the model is available."""
        try:
            if self._engine == "ollama":
                return self._check_ollama_health()
            return self._check_openai_health()
        except Exception as exc:
            logger.warning("Inference health check failed: %s", exc)
            return False

    def _check_ollama_health(self) -> bool:
        import httpx
        resp = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        base = self._model.split(":")[0]
        return any(base in m for m in models)

    def _check_openai_health(self) -> bool:
        import httpx
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        resp = httpx.get(f"{self._base_url}/v1/models", headers=headers, timeout=5.0)
        resp.raise_for_status()
        models = [m.get("id", "") for m in resp.json().get("data", [])]
        return any(self._model in m for m in models)


# Module-level singleton, created on first import
_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    """Return the module-level InferenceEngine singleton.

    When inference.engine == 'llama_cpp_dual', returns an adapter wrapping
    the AsymmetricLLMOrchestrator so existing call sites (generate/stream)
    work transparently.
    """
    global _engine
    if _engine is None:
        inf = get_inference_config()
        if str(inf["engine"]) == "llama_cpp_dual":
            _engine = _build_asymmetric_adapter()
        else:
            _engine = InferenceEngine()
    return _engine


def _build_asymmetric_adapter() -> InferenceEngine:
    """Wrap AlymmetricLLMOrchestrator in an InferenceEngine-compatible shell.

    This lets existing code call engine.generate() / engine.stream() without
    knowing about the dual-model split. Tasks default to GPU executor unless
    the caller hints via the model parameter.
    """
    from backend.core.asymmetric_llm import get_orchestrator

    orch = get_orchestrator()

    adapter = InferenceEngine.__new__(InferenceEngine)
    adapter._engine = "llama_cpp_dual"
    adapter._base_url = "local"
    adapter._model = "dual-model"
    adapter._timeout = 60
    adapter._api_key = ""

    def generate_adapter(
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        num_ctx: int = 4096,
        num_predict: int | None = None,
        timeout: float | None = None,
        images: list[str] | None = None,
    ) -> str:
        prompt = messages[-1]["content"] if messages else ""
        system = next(
            (m["content"] for m in messages if m["role"] == "system"),
            "You are a helpful assistant.",
        )
        return orch.execute(
            prompt, system=system, temperature=temperature,
            max_tokens=num_predict or 2048,
        )

    def stream_adapter(
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        num_ctx: int = 4096,
        num_predict: int | None = None,
        timeout: float | None = None,
    ) -> Generator:
        prompt = messages[-1]["content"] if messages else ""
        system = next(
            (m["content"] for m in messages if m["role"] == "system"),
            "You are a helpful assistant.",
        )
        return orch.stream_execute(
            prompt, system=system, temperature=temperature,
            max_tokens=num_predict or 2048,
        )

    adapter.generate = generate_adapter
    adapter.stream = stream_adapter
    logger.info("InferenceEngine: using asymmetric dual-model adapter")
    return adapter


# ---------------------------------------------------------------------------
# MoE model selection (preserved from original)
# ---------------------------------------------------------------------------


class ModelRouter:
    """Selects the optimal model for a given task type.

    MoE-inspired routing: maintains a pool of models where each model
    is tagged with the task types it specializes in. Lower priority value
    wins when multiple models match the same task.
    """

    def __init__(self) -> None:
        router_cfg = cfg.get("model_router", {})
        self._enabled = bool(router_cfg.get("enabled", True))
        self._general = str(router_cfg.get("general_model", cfg.get("llm", {}).get("model", "dolphin-llama3:8b")))
        self._auto_fallback = bool(router_cfg.get("auto_fallback", True))
        self._pool: list[ModelEntry] = []

        for entry in router_cfg.get("models", []):
            self._pool.append(ModelEntry(
                name=str(entry["name"]),
                task_types=[str(t) for t in entry.get("task_types", [])],
                priority=int(entry.get("priority", 100)),
            ))

        # if no models configured, register the general model as a catch-all
        if not self._pool:
            self._pool.append(ModelEntry(
                name=self._general,
                task_types=list(_ALL_TASK_TYPES),
                priority=100,
            ))

        logger.info(
            "ModelRouter: %d models, general=%s, enabled=%s",
            len(self._pool), self._general, self._enabled,
        )

    @property
    def general_model(self) -> str:
        """The fallback model name used when no specialist matches."""
        return self._general

    @property
    def enabled(self) -> bool:
        """Whether routing is active. When disabled, always returns general."""
        return self._enabled

    def select_model(self, task_type: str) -> str:
        """Pick the best model for a task type. Returns model name."""
        if not self._enabled:
            return self._general

        candidates = [
            m for m in self._pool
            if task_type in m["task_types"]
        ]

        if not candidates:
            if self._auto_fallback:
                logger.debug(
                    "No specialist for '%s', falling back to %s",
                    task_type, self._general,
                )
                return self._general
            raise ValueError(
                f"No model registered for task type '{task_type}' "
                f"and auto_fallback is disabled"
            )

        # lower priority number wins
        candidates.sort(key=lambda m: m["priority"])
        selected = candidates[0]["name"]
        logger.debug("Selected model '%s' for task type '%s'", selected, task_type)
        return selected

    def list_models(self) -> list[ModelEntry]:
        """Return a copy of all registered models."""
        return list(self._pool)

    def register_model(
        self,
        name: str,
        task_types: list[str],
        priority: int = 50,
    ) -> None:
        """Add a specialized model to the pool at runtime."""
        if not name or not name.strip():
            raise ValueError("Model name cannot be empty")
        if not task_types:
            raise ValueError(
                "Must specify at least one task type for model registration"
            )

        clean_name = name.strip()
        # deduplicate: remove existing entry with same name
        self._pool = [m for m in self._pool if m["name"] != clean_name]
        self._pool.append(ModelEntry(
            name=clean_name,
            task_types=[t.strip() for t in task_types],
            priority=priority,
        ))
        logger.info(
            "Registered model '%s' for tasks %s (priority=%d)",
            clean_name, task_types, priority,
        )

    def unregister_model(self, name: str) -> bool:
        """Remove a model from the pool. Returns True if found and removed."""
        before = len(self._pool)
        self._pool = [m for m in self._pool if m["name"] != name]
        removed = len(self._pool) < before
        if removed:
            logger.info("Unregistered model '%s'", name)
        return removed

    def check_availability(self, model_name: str) -> bool:
        """Verify the model is available on the configured inference engine."""
        try:
            engine = get_engine()
            return engine.check_health()
        except (ImportError, ConnectionError, OSError, KeyError, TypeError) as exc:
            logger.warning(
                "Availability check failed for '%s': %s",
                model_name, exc,
            )
            return False

    def select_with_fallback(self, task_type: str) -> str:
        """Select model with live availability check, falling back if needed."""
        preferred = self.select_model(task_type)
        if preferred == self._general:
            return preferred

        if self.check_availability(preferred):
            return preferred

        logger.warning(
            "Model '%s' unavailable for task '%s', falling back to '%s'",
            preferred, task_type, self._general,
        )
        return self._general

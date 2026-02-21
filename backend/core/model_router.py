# Author: Bradley R. Kinnard
"""MoE-style model router for Ollama multi-model serving.

Phase 6: routes tasks to specialized models based on query decomposition
output. The general model handles everything by default; specialized models
(code-tuned, critique-tuned, etc.) can be registered via config and take
precedence for matching task types. Compatible with Ollama multi-model serving.
"""
from __future__ import annotations

import logging
from typing import Literal, TypedDict

from backend.config import cfg

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


class ModelRouter:
    """Selects the optimal Ollama model for a given task type.

    MoE-inspired routing: maintains a pool of models where each model
    is tagged with the task types it specializes in. Lower priority value
    wins when multiple models match the same task.
    """

    def __init__(self) -> None:
        router_cfg = cfg.get("model_router", {})
        self._enabled = bool(router_cfg.get("enabled", True))
        self._general = str(router_cfg.get("general_model", cfg["llm"]["model"]))
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
        """Pick the best model for a task type. Returns Ollama model name."""
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
        """Ping Ollama to confirm the model is pulled and ready."""
        try:
            from ollama import Client
            client = Client(host=cfg["llm"]["host"])
            models_resp = client.list()
            available = [m["name"] for m in models_resp.get("models", [])]
            base = model_name.split(":")[0]
            return any(base in m for m in available)
        except (ImportError, ConnectionError, OSError, KeyError, TypeError) as exc:
            logger.warning(
                "Ollama availability check failed for '%s': %s",
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

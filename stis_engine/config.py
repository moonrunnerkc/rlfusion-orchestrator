# Author: Bradley R. Kinnard
"""STIS configuration: typed, centralized, safe defaults, env overridable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SwarmConfig:
    """Swarm consensus parameters for the convergence loop."""
    num_agents: int = 4
    similarity_threshold: float = 0.92
    alpha: float = 0.5
    max_iterations: int = 30


@dataclass(frozen=True)
class ModelConfig:
    """Qwen2.5-1.5B model loading settings."""
    model_id: str = "Qwen/Qwen2.5-1.5B"
    dtype: str = "float16"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42


@dataclass(frozen=True)
class ServerConfig:
    """FastAPI server settings."""
    host: str = "0.0.0.0"
    port: int = 8100
    timeout_secs: int = 45


@dataclass(frozen=True)
class STISConfig:
    """Top-level configuration container for the STIS engine."""
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config() -> STISConfig:
    """Build config from environment overrides on top of safe defaults."""
    swarm = SwarmConfig(
        num_agents=int(os.environ.get("STIS_NUM_AGENTS", "2")),
        similarity_threshold=float(os.environ.get("STIS_SIM_THRESHOLD", "0.92")),
        alpha=float(os.environ.get("STIS_ALPHA", "0.5")),
        max_iterations=int(os.environ.get("STIS_MAX_ITERATIONS", "20")),
    )
    model = ModelConfig(
        model_id=os.environ.get("STIS_MODEL_ID", "Qwen/Qwen2.5-1.5B"),
        dtype=os.environ.get("STIS_DTYPE", "float16"),
        max_new_tokens=int(os.environ.get("STIS_MAX_NEW_TOKENS", "512")),
        temperature=float(os.environ.get("STIS_TEMPERATURE", "0.7")),
        top_p=float(os.environ.get("STIS_TOP_P", "0.9")),
        seed=int(os.environ.get("STIS_SEED", "42")),
    )
    server = ServerConfig(
        host=os.environ.get("STIS_HOST", "0.0.0.0"),
        port=int(os.environ.get("STIS_PORT", "8100")),
        timeout_secs=int(os.environ.get("STIS_TIMEOUT", "45")),
    )
    return STISConfig(swarm=swarm, model=model, server=server)

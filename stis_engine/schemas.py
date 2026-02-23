# Author: Bradley R. Kinnard
"""Request/Response schemas for the STIS FastAPI server."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Inbound request for /generate."""
    prompt: str = Field(..., min_length=1, max_length=8000, description="Input prompt for swarm consensus generation")
    max_new_tokens: int = Field(default=512, ge=1, le=2048, description="Maximum tokens to generate")
    num_agents: int | None = Field(default=None, ge=2, le=16, description="Override swarm agent count")
    similarity_threshold: float | None = Field(default=None, ge=0.5, le=1.0, description="Override convergence threshold")
    alpha: float | None = Field(default=None, ge=0.01, le=1.0, description="Override blending rate")


class ConvergenceStepResponse(BaseModel):
    """A single convergence iteration snapshot."""
    iteration: int
    mean_similarity: float
    max_deviation: float
    centroid_norm: float


class GenerateResponse(BaseModel):
    """Outbound response from /generate."""
    text: str
    total_tokens: int
    convergence_log: list[list[ConvergenceStepResponse]]
    final_similarity: float
    total_iterations: int
    wall_time_secs: float


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: str
    model_loaded: bool
    model_id: str = ""
    hidden_dim: int
    num_agents: int
    similarity_threshold: float
    device: str


class ErrorResponse(BaseModel):
    """Structured error response."""
    error: str
    detail: str

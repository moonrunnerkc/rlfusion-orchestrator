# Author: Bradley R. Kinnard
"""Pydantic request/response models for the public HTTP surface.

All models use ConfigDict(extra="forbid") so unknown keys raise 422 rather
than being silently dropped. Bounded Field constraints replace the ad-hoc
length / magic-number checks that used to live inline in main.py.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


_MAX_QUERY_LEN = 4000


class ChatRequest(BaseModel):
    """Body for POST /chat and the first frame of /ws."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=_MAX_QUERY_LEN)
    mode: Literal["chat", "build"] = "chat"


class WsControlFrame(BaseModel):
    """Out-of-band control frame on /ws (auth handshake, clear-memory)."""

    model_config = ConfigDict(extra="forbid")

    auth: Optional[str] = Field(default=None, max_length=512)
    clear_memory: Optional[bool] = None
    new_chat: Optional[bool] = None


class ConfigPatch(BaseModel):
    """Body for PATCH /api/config. Only the web.enabled toggle is exposed."""

    model_config = ConfigDict(extra="forbid")

    class _WebPatch(BaseModel):
        model_config = ConfigDict(extra="forbid")
        enabled: bool

    web: _WebPatch


class FineTuneRequest(BaseModel):
    """Body for POST /api/fine-tune. All fields optional; defaults from config."""

    model_config = ConfigDict(extra="forbid")

    base_model: Optional[str] = Field(default=None, max_length=256)
    lora_rank: Optional[int] = Field(default=None, ge=1, le=256)
    lora_alpha: Optional[int] = Field(default=None, ge=1, le=512)
    lora_dropout: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    learning_rate: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    num_epochs: Optional[int] = Field(default=None, ge=1, le=100)
    batch_size: Optional[int] = Field(default=None, ge=1, le=128)
    max_seq_length: Optional[int] = Field(default=None, ge=128, le=32768)
    min_reward: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_episodes: Optional[int] = Field(default=None, ge=1, le=100_000)
    val_split: Optional[float] = Field(default=None, ge=0.0, le=0.5)
    output_dir: Optional[str] = Field(default=None, max_length=256)

    @field_validator("output_dir")
    @classmethod
    def _scope_output_dir(cls, value: Optional[str]) -> Optional[str]:
        """Reject absolute paths and path traversal in output_dir.

        The router rebases this under PROJECT_ROOT / "models" before passing
        to the trainer; the validator just keeps the input shape clean.
        """
        if value is None:
            return value
        if value.startswith("/") or value.startswith("\\"):
            raise ValueError("output_dir must be a relative path under models/")
        if ".." in value.split("/") or ".." in value.split("\\"):
            raise ValueError("output_dir cannot contain '..' segments")
        return value


class UploadResponse(BaseModel):
    """Response shape for POST /api/upload."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["uploaded", "empty"]
    saved: list[str]
    skipped: list[str]
    total_saved: int = Field(ge=0)
    total_skipped: int = Field(ge=0)


class HealthResponse(BaseModel):
    """Response shape for GET /ping."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["alive"]
    gpu: Optional[str]
    device: Literal["cuda", "cpu"]
    model: str
    inference_engine: str
    engine_resolution: str
    policy: str
    policy_exists: bool
    boot_id: str


# Magic-byte signatures for upload sniffing. Files that pass the extension
# check but fail the magic check are rejected.
_MAGIC_BYTES: dict[str, list[bytes]] = {
    ".pdf": [b"%PDF"],
    # txt and md are heterogeneous text formats; we only verify they decode
    # as UTF-8 / Latin-1 in the upload handler, not via magic bytes here.
}


def magic_bytes_for(ext: str) -> list[bytes]:
    return _MAGIC_BYTES.get(ext.lower(), [])


__all__ = [
    "ChatRequest",
    "WsControlFrame",
    "ConfigPatch",
    "FineTuneRequest",
    "UploadResponse",
    "HealthResponse",
    "magic_bytes_for",
]


def coerce_dict(model: BaseModel) -> dict[str, Any]:
    """Convenience helper used by routes to pull a plain dict from a model."""
    return model.model_dump(exclude_none=True)

# Author: Bradley R. Kinnard
"""Pydantic request/response models for the public HTTP surface.

The previous /chat and /api/fine-tune endpoints took raw `Dict[str, Any]`
bodies, which meant any junk field was silently accepted and downstream
type coercion was up to ad-hoc code in the handler. These models pin the
shape, bound the value ranges, and reject extra fields so a malformed
client request produces an HTTP 422 with a clear diagnostic rather than
arriving as a hidden None somewhere in the pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Match the runtime cap that backend.main enforces on query length.
MAX_QUERY_LEN = 4000

ChatMode = Literal["chat", "build", "test"]


class ChatRequest(BaseModel):
    """Body for POST /chat and the WS /ws first frame."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_LEN)
    mode: ChatMode = "chat"


class ConfigPatchWeb(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class ConfigPatch(BaseModel):
    """Body for PATCH /api/config; only the `web.enabled` toggle is supported."""

    model_config = ConfigDict(extra="forbid")

    web: ConfigPatchWeb


class FineTuneRequest(BaseModel):
    """Body for POST /api/fine-tune. Bounds keep one user from launching a
    process that would consume all of disk / VRAM by accident."""

    model_config = ConfigDict(extra="forbid")

    base_model: str | None = Field(default=None, max_length=200)
    lora_rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.9)
    learning_rate: float = Field(default=2e-4, ge=1e-6, le=1e-2)
    num_epochs: int = Field(default=3, ge=1, le=50)
    batch_size: int = Field(default=4, ge=1, le=64)
    max_seq_length: int = Field(default=2048, ge=128, le=32768)
    min_reward: float = Field(default=0.8, ge=0.0, le=1.0)
    max_episodes: int = Field(default=5000, ge=1, le=1_000_000)
    val_split: float = Field(default=0.1, ge=0.0, le=0.5)
    output_dir: str = Field(default="models/fine_tuned", max_length=512)

    @field_validator("output_dir")
    @classmethod
    def _scope_output_dir(cls, value: str) -> str:
        """Reject output_dir values that escape the project models/ tree.

        The actual project root resolution happens in the handler (we can't
        import config here without cycles), but we can reject anything with
        traversal markers, absolute paths, or shell-special characters at
        validation time.
        """
        p = Path(value)
        if p.is_absolute():
            raise ValueError("output_dir must be a relative path under the project root")
        if ".." in p.parts:
            raise ValueError("output_dir may not contain '..'")
        if any(ch in value for ch in ("\x00", "\n", "\r")):
            raise ValueError("output_dir contains forbidden whitespace/null bytes")
        return value


class UploadFileEntry(BaseModel):
    name: str
    bytes_written: int = Field(ge=0)


class UploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["uploaded", "empty", "error"]
    saved: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    total_saved: int = Field(ge=0)
    total_skipped: int = Field(ge=0)

# Author: Bradley R. Kinnard
# backend/core/utils.py
# Embedding + misc utilities for RLFO

from sentence_transformers import SentenceTransformer
from typing import List, Optional
import torch
import numpy as np
from pathlib import Path
import hashlib

# embedder cache - don't load this thing twice
_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    """Lazy-load sentence transformer. GPU only - project requirement."""
    global _embedder

    if _embedder is not None:
        return _embedder

    from backend.main import cfg  # late import to dodge circular nonsense

    model_name = cfg["embedding"]["model"]
    device_str = cfg["embedding"]["device"]

    # project mandate: CUDA everywhere. don't ask why.
    assert device_str == "cuda", f"Device must be 'cuda', got '{device_str}'"

    print(f"Loading embedding model '{model_name}' on {device_str}...")

    _embedder = SentenceTransformer(model_name)
    _embedder = _embedder.to(device_str)
    _embedder.eval()

    return _embedder


def embed_text(text: str) -> np.ndarray:
    """Single text -> embedding vector (384-d, normalized)"""
    embedder = get_embedder()

    with torch.no_grad():
        embedding = embedder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    return embedding.astype(np.float32)


def embed_batch(texts: List[str]) -> np.ndarray:
    """
    Batch version of embed_text - way faster for multiple inputs.
    Returns shape: (len(texts), 384)
    """
    embedder = get_embedder()

    with torch.no_grad():
        embeddings = embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    return embeddings.astype(np.float32)


def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    """
    Split text into roughly equal chunks. Tries to respect paragraph breaks.
    max_tokens is really max_chars but whatever, close enough for embeddings.
    """
    chunks = []
    paragraphs = text.split("\n\n")

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= max_tokens:
            chunks.append(para)
        else:
            # para too long, chop on spaces
            words = para.split()
            current = ""

            for w in words:
                test = current + " " + w if current else w

                if len(test) <= max_tokens:
                    current = test
                else:
                    if current:
                        chunks.append(current)
                    current = w

            if current:  # leftover
                chunks.append(current)

    return chunks


def softmax(weights: List[float], temperature: float = 1.0) -> List[float]:
    """
    Standard softmax with temp control.
    temperature=0 -> argmax (one-hot on winner)
    Uses the max-subtraction trick so it doesn't blow up with large weights.
    """
    w = np.array(weights, dtype=np.float64)

    if temperature == 0.0:
        # degenerate case: just pick the max
        result = np.zeros_like(w)
        result[np.argmax(w)] = 1.0
        return result.tolist()

    w = w / temperature
    w_max = np.max(w)
    w_shifted = w - w_max  # numerical stability hack

    exp_w = np.exp(w_shifted)
    sum_exp = np.sum(exp_w)
    probs = exp_w / sum_exp

    return probs.tolist()


def deterministic_id(text: str) -> str:
    """Hash text to stable 16-char ID. Same input = same output always."""
    h = hashlib.sha256(text.encode("utf-8"))
    return h.hexdigest()[:16]


def ensure_path(path_str: str) -> Path:
    """Convert str to Path and mkdir parent dirs if needed. Idempotent."""
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

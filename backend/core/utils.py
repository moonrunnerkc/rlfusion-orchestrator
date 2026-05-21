# Author: Bradley R. Kinnard
# utils.py - embedding, chunking, hashing, softmax, OOD detection
# Originally built for personal offline use, now open-sourced for public benefit.

import hashlib
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Resolve embedder device dynamically. Inputs in priority order:
#   1. RLFUSION_FORCE_CPU=true                     -> cpu
#   2. RLFUSION_DEVICE  (cpu|cuda|mps|auto)
#   3. embedding.device in config.yaml             (same set)
#   4. fallback                                    -> auto
# "auto" then resolves to cuda if available, else mps on Apple Silicon,
# else cpu. Anything torch doesn't recognize is treated as auto.
_VALID = {"cpu", "cuda", "mps"}


def _resolve_to_torch(requested: str) -> str:
    """Turn the textual preference into a device torch will accept."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        mps_ok = (
            bool(getattr(torch.backends, "mps", None))
            and torch.backends.mps.is_available()
        )
    except Exception:
        return "cpu"

    if requested == "cuda":
        if cuda_ok:
            return "cuda"
        logger.info(
            "embedding.device=cuda requested but CUDA unavailable; falling back"
        )
    if requested == "mps":
        if mps_ok:
            return "mps"
        logger.info("embedding.device=mps requested but MPS unavailable; falling back")
    # auto / unrecognized
    if cuda_ok:
        return "cuda"
    if mps_ok:
        return "mps"
    return "cpu"


def _get_device() -> str:
    if os.environ.get("RLFUSION_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return "cpu"
    env_device = (os.environ.get("RLFUSION_DEVICE") or "").strip().lower()
    if env_device in _VALID:
        return env_device
    if env_device and env_device != "auto":
        logger.warning("Unknown RLFUSION_DEVICE=%r; treating as auto", env_device)

    requested = env_device or "auto"
    if requested == "auto":
        try:
            cfg_path = Path(__file__).parent.parent / "config.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            requested = (
                str(cfg.get("embedding", {}).get("device", "auto")).strip().lower()
            )
        except Exception:
            requested = "auto"
    return _resolve_to_torch(requested)


_device = _get_device()
logger.info("Embedding device: %s", _device)
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=_device)


# LRU cache keyed on text content, eliminates ~13 redundant embed calls per query
_embed_cache: dict[str, np.ndarray] = {}
_EMBED_CACHE_MAX = 256


def embed_text(text: str) -> np.ndarray:
    """Cached single-text embedding. Same text returns same ndarray without recompute."""
    cache_key = hashlib.sha256(text.encode()).hexdigest()
    if cache_key in _embed_cache:
        return _embed_cache[cache_key]
    result = embedder.encode(
        text, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    # evict oldest if cache is full
    if len(_embed_cache) >= _EMBED_CACHE_MAX:
        oldest = next(iter(_embed_cache))
        del _embed_cache[oldest]
    _embed_cache[cache_key] = result
    return result


def clear_embed_cache() -> None:
    """Flush the embedding cache. Call after reindex or model swap."""
    _embed_cache.clear()


def embed_batch(texts: List[str]) -> np.ndarray:
    return embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    words = text.split()
    chunks, current, count = [], [], 0
    for word in words:
        count += 1
        if count > max_tokens:
            chunks.append(" ".join(current))
            current, count = [word], 1
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


def ensure_path(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def deterministic_id(text: str) -> str:
    return hashlib.shake_256(text.encode()).hexdigest(16)


def softmax(weights: List[float], temperature: float = 1.0) -> List[float]:
    w = np.array(weights, dtype=np.float64)
    if temperature == 0.0:
        result = np.zeros_like(w)
        result[np.argmax(w)] = 1.0
        return result.tolist()
    scaled = w / temperature
    shifted = scaled - np.max(scaled)
    exp_w = np.exp(shifted)
    return (exp_w / np.sum(exp_w)).tolist()


# OOD detection via Mahalanobis distance with Ledoit-Wolf shrinkage
_ood_cache = {"mean": None, "precision": None, "fitted": False}


def fit_ood_detector(embeddings: np.ndarray) -> None:
    from sklearn.covariance import LedoitWolf

    logger.info("Fitting OOD detector on %d samples", embeddings.shape[0])
    _ood_cache["mean"] = np.mean(embeddings, axis=0)
    lw = LedoitWolf()
    lw.fit(embeddings)
    _ood_cache["precision"] = lw.precision_
    _ood_cache["fitted"] = True
    logger.info("OOD detector fitted. Shrinkage: %.4f", lw.shrinkage_)


def mahalanobis_distance(embedding: np.ndarray) -> float:
    if not _ood_cache["fitted"]:
        return -1.0
    diff = embedding - _ood_cache["mean"]
    return float(np.sqrt(np.dot(np.dot(diff, _ood_cache["precision"]), diff)))


def is_ood(embedding: np.ndarray, threshold: float = 50.0) -> tuple:
    dist = mahalanobis_distance(embedding)
    if dist < 0:
        return (False, -1.0)
    ood = dist > threshold
    if ood:
        logger.warning("OOD flagged! Distance: %.2f > %.2f", dist, threshold)
    return (ood, dist)


def check_query_ood(query: str, threshold: float = 50.0) -> tuple:
    return is_ood(embed_text(query), threshold)

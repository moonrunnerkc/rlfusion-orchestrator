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

# Load device from config, fallback to env var, then to cuda if available
def _get_device() -> str:
    env_device = os.environ.get("RLFUSION_DEVICE")
    if env_device:
        return env_device
    try:
        cfg_path = Path(__file__).parent.parent / "config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("embedding", {}).get("device", "cuda")
    except Exception:
        return "cuda"

_device = _get_device()
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=_device)


def embed_text(text: str) -> np.ndarray:
    return embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True,
                          show_progress_bar=False).astype(np.float32)


def embed_batch(texts: List[str]) -> np.ndarray:
    return embedder.encode(texts, batch_size=32, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


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

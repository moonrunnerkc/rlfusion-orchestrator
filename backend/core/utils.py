"""
Utility functions for text processing, embedding, and numerical operations.

This module provides core functionality for text embedding using sentence transformers,
text chunking, hashing, and probability distributions used throughout the retrieval
and fusion pipeline.

Author: Bradley R. Kinnard
License: MIT
"""

import hashlib
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device="cpu"
)


def embed_text(text: str) -> np.ndarray:
    """
    Generate normalized embedding vector for input text.

    Uses BAAI/bge-small-en-v1.5 model on CPU to produce 384-dimensional
    embeddings suitable for semantic similarity search.

    Args:
        text: Input text to embed

    Returns:
        Normalized 384-d numpy array (float32)
    """
    return embedder.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

def embed_batch(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for multiple texts efficiently.

    Processes texts in batches for improved performance when embedding
    large document collections.

    Args:
        texts: List of text strings to embed

    Returns:
        Normalized embedding matrix of shape (n_texts, 384) as float32
    """
    return embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)


def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    """
    Split text into smaller chunks for processing.

    Performs word-level chunking with approximate token counting.
    Each chunk maintains word boundaries and does not exceed the
    specified token limit.

    Args:
        text: Input text to chunk
        max_tokens: Maximum number of tokens (words) per chunk

    Returns:
        List of text chunks, each <= max_tokens in length
    """
    words = text.split()
    chunks = []
    current_chunk = []
    token_count = 0

    for word in words:
        token_count += 1
        if token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            token_count = 1
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def ensure_path(path: str) -> None:
    """
    Create directory structure if it does not exist.

    Ensures all parent directories in the given path are created,
    similar to `mkdir -p`.

    Args:
        path: File or directory path to ensure
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def deterministic_id(text: str) -> str:
    """
    Generate deterministic hash identifier for text content.

    Uses SHAKE256 to produce a 16-character hexadecimal hash that
    uniquely identifies text content. Identical text always produces
    the same identifier.

    Args:
        text: Input text to hash

    Returns:
        16-character hexadecimal hash string
    """
    return hashlib.shake_256(text.encode()).hexdigest(16)


def softmax(weights: List[float], temperature: float = 1.0) -> List[float]:
    """
    Apply softmax transformation with temperature scaling.

    Converts a vector of weights into a probability distribution.
    Temperature parameter controls the sharpness of the distribution:
    - temperature > 1.0: Smoother, more uniform distribution
    - temperature = 1.0: Standard softmax
    - temperature < 1.0: Sharper, more peaked distribution
    - temperature = 0.0: Hard argmax (one-hot encoding)

    Uses numerical stabilization (max subtraction) to prevent overflow.

    Args:
        weights: List of numerical weights
        temperature: Scaling factor for distribution sharpness

    Returns:
        List of probabilities summing to 1.0
    """
    weight_array = np.array(weights, dtype=np.float64)

    if temperature == 0.0:
        result = np.zeros_like(weight_array)
        result[np.argmax(weight_array)] = 1.0
        return result.tolist()

    scaled_weights = weight_array / temperature
    max_weight = np.max(scaled_weights)
    shifted_weights = scaled_weights - max_weight

    exp_weights = np.exp(shifted_weights)
    weight_sum = np.sum(exp_weights)
    probabilities = exp_weights / weight_sum

    return probabilities.tolist()


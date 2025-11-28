import numpy as np
from typing import List
import hashlib
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Blackwell cuBLAS workaround - use CPU for embeddings
# GPU still used for: FAISS (if enabled), Ollama LLM, RL training
print("Using CPU for embeddings (Blackwell GPU compatibility)")
embedder = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device="cpu"
)

def embed_text(text: str) -> np.ndarray:
    """Turn text into a vector. Magic, basically."""
    return embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

def embed_batch(texts: List[str]) -> np.ndarray:
    """Like embed_text but for when you've got a whole pile of text to process."""
    return embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    """
    Chop up text into bite-sized pieces. We're calling them 'tokens' but really
    we're just counting words because life's too short for proper tokenization.
    """
    words = text.split()
    chunks = []
    current = []
    count = 0
    for word in words:
        count += 1
        if count > max_tokens:
            chunks.append(" ".join(current))
            current = [word]
            count = 1
        else:
            current.append(word)
    if current:  # don't forget the last chunk hanging around
        chunks.append(" ".join(current))
    return chunks

def ensure_path(path: str) -> None:
    """Make sure a path exists. If not, create it. Simple as that."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def deterministic_id(text: str) -> str:
    """Generate a stable hash for text. Same text in = same ID out. Always."""
    return hashlib.shake_256(text.encode()).hexdigest(16)

def softmax(weights: List[float], temperature: float = 1.0) -> List[float]:
    """
    Softmax with temperature control. Higher temp = more democratic,
    lower temp = winner takes all. Set to 0 and you get a hard argmax.
    We do the max-subtraction dance to avoid numeric explosions.
    """
    w = np.array(weights, dtype=np.float64)

    if temperature == 0.0:
        # Cold as ice - just pick the winner
        result = np.zeros_like(w)
        result[np.argmax(w)] = 1.0
        return result.tolist()

    w = w / temperature
    w_max = np.max(w)
    w_shifted = w - w_max  # keeps exp() from going to infinity and beyond

    exp_w = np.exp(w_shifted)
    sum_exp = np.sum(exp_w)
    probs = exp_w / sum_exp

    return probs.tolist()


# Author: Bradley R. Kinnard
# backend/core/retrievers.py
# Retrieval engines: RAG (vector), CAG (cache), Graph (semantic)

import faiss
import sqlite3
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Tuple
from backend.core.utils import embed_text, embed_batch, ensure_path, deterministic_id
from backend.main import cfg

# FAISS GPU resource - don't init this twice
_gpu_res: Any = None


def init_faiss_gpu() -> None:
    """Setup FAISS GPU resources once. Reentrant safe."""
    global _gpu_res

    if _gpu_res is not None:
        return

    _gpu_res = faiss.StandardGpuResources()
    print("FAISS GPU resources initialized")


# auto-init at import time
init_faiss_gpu()


def get_rag_index() -> faiss.IndexFlatL2:
    """Load or create FAISS index on GPU. 384-d for sentence-transformers."""
    index_path = Path(cfg["paths"]["index"])

    if index_path.exists():
        cpu_index = faiss.read_index(str(index_path))
        gpu_index = faiss.index_cpu_to_gpu(_gpu_res, 0, cpu_index)
        print("RAG index loaded from disk")
    else:
        # new index, 384 dims
        cpu_index = faiss.IndexFlatL2(384)
        gpu_index = faiss.index_cpu_to_gpu(_gpu_res, 0, cpu_index)
        print("Created new RAG index")

    return gpu_index


def retrieve_rag(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Vector search over RAG index. GPU-accelerated."""
    query_vec = embed_text(query)
    query_vec = query_vec.reshape(1, 384)  # faiss wants (1, dim)

    index = get_rag_index()
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        dist = float(distances[0][i])

        # TODO: load actual text from metadata store
        results.append({
            "text": "",
            "score": dist,
            "id": str(idx)
        })

    return results


def retrieve_cag(query: str, threshold: float = 0.85) -> List[Dict[str, Any]]:
    """Cache lookup. Returns cached result if score meets threshold."""
    key = deterministic_id(query)
    db_path = cfg["paths"]["db"]

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT value, score FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()

        if row and row["score"] >= threshold:
            return [{
                "text": row["value"],
                "score": row["score"],
                "id": key
            }]
        else:
            return []

    except sqlite3.OperationalError:
        # table not created yet, that's fine
        return []


def retrieve_graph(query: str, max_hops: int = 2) -> List[Dict[str, Any]]:
    """Graph-based retrieval. Find related nodes via BFS."""
    graph_path = Path(cfg["paths"]["graph"])

    # load graph or create empty one
    if graph_path.exists():
        try:
            import json
            with open(graph_path) as f:
                data = json.load(f)
            G = nx.node_link_graph(data)
        except:
            G = nx.DiGraph()
    else:
        G = nx.DiGraph()

    # TODO: implement graph traversal
    # - embed query
    # - find best node by cosine sim
    # - BFS to max_hops
    # - collect neighbors with relation labels

    return []


def retrieve(
    query: str,
    rag_weight: float = 1.0,
    cag_weight: float = 1.0,
    graph_weight: float = 1.0,
    top_k: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Main retrieval function. Runs all three modes and weights results.
    This is what the RL agent will call with learned weights.
    """
    # hit all three retrievers
    rag_results = retrieve_rag(query, top_k=top_k)
    cag_results = retrieve_cag(query)
    graph_results = retrieve_graph(query, max_hops=2)

    # apply weights and tag source
    for r in rag_results:
        r["score"] *= rag_weight
        r["source"] = "rag"

    for r in cag_results:
        r["score"] *= cag_weight
        r["source"] = "cag"

    for r in graph_results:
        r["score"] *= graph_weight
        r["source"] = "graph"

    # sort by score
    rag_results.sort(key=lambda x: x["score"], reverse=True)
    cag_results.sort(key=lambda x: x["score"], reverse=True)
    graph_results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "rag": rag_results[:top_k],
        "cag": cag_results[:top_k],
        "graph": graph_results[:top_k]
    }

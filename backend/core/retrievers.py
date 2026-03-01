# Author: Bradley R. Kinnard
# retrievers.py - CAG/Graph retrieval with CSWR stability filtering
# RAG (FAISS) and Web (Tavily) paths removed per upgrade plan Step 3.

import hashlib
import sqlite3
import json
import uuid
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import networkx as nx
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING, List, Dict, Any
from backend.core.utils import embed_text, embed_batch, ensure_path, deterministic_id

if TYPE_CHECKING:
    from backend.core.graph_engine import GraphEngine
from backend.core.decomposer import decompose_query
from backend.config import cfg, PROJECT_ROOT, get_web_api_key

logger = logging.getLogger("cswr")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

DOMAIN_KW = {
    "tech": ["api", "gpu", "cuda", "tensor", "model", "neural", "llm", "transformer",
             "embedding", "vector", "faiss", "pytorch", "rag", "retrieval"],
    "code": ["function", "class", "def ", "import", "return", "async", "await",
             "try:", "except", "lambda", "self.", "print(", "for ", "while "],
}


def detect_domain(query: str) -> str:
    q = query.lower()
    if sum(1 for kw in DOMAIN_KW["code"] if kw in q) >= 2:
        return "code"
    if sum(1 for kw in DOMAIN_KW["tech"] if kw in q) >= 1:
        return "tech"
    return "general"


def get_stability_thresh(domain: str) -> float:
    quants = cfg.get("cswr_quantiles", {})
    if domain in quants:
        return quants[domain].get("stability_threshold", 0.7)
    return cfg.get("cswr", {}).get("stability_threshold", 0.7)


def compute_domain_quantiles(episodes: list) -> dict:
    domain_scores = {"tech": [], "code": [], "general": []}
    for ep in episodes:
        scores = ep.get("stability_scores", [])
        if scores:
            domain_scores[detect_domain(ep.get("query", ""))].extend(scores)

    quantiles = {}
    for dom, scores in domain_scores.items():
        if len(scores) < 10:
            quantiles[dom] = {"stability_threshold": 0.7, "samples": len(scores)}
            continue
        arr = np.array(scores)
        q25, q50, q75 = [float(np.percentile(arr, p)) for p in [25, 50, 75]]
        quantiles[dom] = {"stability_threshold": q25, "q25": q25, "q50": q50,
                          "q75": q75, "samples": len(scores)}
    return quantiles


def save_quantiles(quantiles: dict):
    import yaml
    path = Path(__file__).parents[1] / "config.yaml"
    with open(path) as f:
        config = yaml.safe_load(f)
    config["cswr_quantiles"] = quantiles
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF file."""
    import PyPDF2
    text = ""
    with open(path, "rb") as f:
        for page in PyPDF2.PdfReader(f).pages:
            text += page.extract_text() + "\n"
    return text


def _get_docs_path() -> Path:
    """Return the docs directory path, creating if needed."""
    path = PROJECT_ROOT / "data" / "docs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_index_path() -> Path:
    """DEPRECATED: FAISS index path. Kept for backward compat, returns legacy path."""
    index_dir = PROJECT_ROOT / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir / "rag_index.faiss"


def _get_metadata_path() -> Path:
    """DEPRECATED: metadata JSON path. Kept for backward compat."""
    index_dir = PROJECT_ROOT / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir / "metadata.json"


def _compute_all_project_coherence(
    embeddings: np.ndarray, chunks: list[dict[str, str]], k: int = 5,
) -> list[float]:
    """Compute project coherence for every chunk at index time.

    For each chunk, finds its k nearest neighbors in embedding space and
    measures what fraction share the same project tag. A chunk whose neighbors
    are all from the same project scores 1.0; one surrounded by foreign-project
    chunks scores near 0.0. This is the novel CSWR axis.
    """
    n = len(chunks)
    if n <= 1:
        return [1.0] * n

    # self-similarity matrix via dot product (embeddings are L2-normalized)
    sims = embeddings @ embeddings.T
    coherence = []
    for i in range(n):
        # exclude self, get top-k neighbor indices
        row = sims[i].copy()
        row[i] = -1.0
        actual_k = min(k, n - 1)
        neighbor_idxs = np.argpartition(row, -actual_k)[-actual_k:]
        same_project = sum(
            1 for j in neighbor_idxs if chunks[j]["project"] == chunks[i]["project"]
        )
        coherence.append(same_project / actual_k)
    return coherence


def _compute_project_centroids(
    embeddings: np.ndarray, chunks: list[dict[str, str]],
) -> dict[str, np.ndarray]:
    """Compute mean embedding per project. Used for fast query routing."""
    project_vecs: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        project_vecs.setdefault(c["project"], []).append(i)
    centroids = {}
    for proj, idxs in project_vecs.items():
        proj_embs = embeddings[idxs]
        centroid = proj_embs.mean(axis=0)
        # L2 normalize
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[proj] = centroid
    return centroids


def _load_project_centroids() -> dict[str, np.ndarray]:
    """Load precomputed project centroids from disk."""
    path = PROJECT_ROOT / "indexes" / "project_centroids.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {proj: np.array(vec, dtype=np.float32) for proj, vec in data.items()}


def route_to_projects(query: str, gap_threshold: float = 0.15) -> list[str] | None:
    """Pre-filter which projects to search based on centroid similarity.

    Returns a list of project names to search, or None for 'search all'.
    If one project dominates by > gap_threshold, routes exclusively to it.
    """
    centroids = _load_project_centroids()
    if len(centroids) <= 1:
        return None  # single project, no routing needed

    q_emb = embed_text(query)
    scores = {proj: float(np.dot(q_emb, centroid)) for proj, centroid in centroids.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best_proj, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_score - second_score > gap_threshold:
        logger.info("Centroid routing: '%s' (%.3f) dominates by %.3f", best_proj, best_score, best_score - second_score)
        return [best_proj]

    # close scores, search all qualifying projects (within gap of best)
    qualifying = [proj for proj, sc in ranked if best_score - sc <= gap_threshold]
    logger.info("Centroid routing: multi-project %s (gap %.3f)", qualifying, best_score - second_score)
    return qualifying


def build_rag_index() -> None:
    """DEPRECATED: FAISS RAG index removed in Step 3 upgrade.

    Use retrieve_graph() for entity-based retrieval or populate CAG
    via the main pipeline's cache-on-success path.
    """
    import warnings
    warnings.warn(
        "build_rag_index() is deprecated. FAISS removed in CAG-RL-Fusion upgrade.",
        DeprecationWarning, stacklevel=2,
    )
    logger.warning("build_rag_index() called but FAISS has been removed.")

    # still build entity graph from docs if available
    if cfg.get("graph", {}).get("enabled", True):
        docs_path = _get_docs_path()
        files = (list(docs_path.rglob("*.txt")) + list(docs_path.rglob("*.md"))
                 + list(docs_path.rglob("*.pdf")))
        all_chunks = []
        for fpath in files:
            try:
                content = extract_pdf_text(fpath) if fpath.suffix.lower() == ".pdf" else fpath.read_text()
                rel = fpath.relative_to(docs_path)
                project = rel.parts[0] if len(rel.parts) > 1 else "default"
                from backend.core.utils import chunk_text
                for chunk in chunk_text(content, max_tokens=400):
                    all_chunks.append({"text": chunk, "source": str(rel), "project": project})
            except Exception as e:
                logger.warning("Failed to process %s: %s", fpath, e)
        if all_chunks:
            try:
                entity_count = build_entity_graph(all_chunks)
                logger.info("Entity graph built with %d entities", entity_count)
            except (ImportError, RuntimeError) as exc:
                logger.warning("Entity graph build skipped: %s", exc)


def get_rag_index() -> None:
    """DEPRECATED: FAISS RAG index removed in Step 3 upgrade."""
    import warnings
    warnings.warn(
        "get_rag_index() is deprecated. FAISS removed in CAG-RL-Fusion upgrade.",
        DeprecationWarning, stacklevel=2,
    )
    logger.warning("get_rag_index() called but FAISS has been removed.")


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def compute_stability(chunk: dict, all_chunks: list, embeddings: dict) -> float:
    idx = next((i for i, c in enumerate(all_chunks) if c["id"] == chunk["id"]), -1)
    if idx == -1 or len(all_chunks) < 2:
        return 0.5

    emb = embeddings[chunk["id"]]
    sims = []
    if idx > 0:
        sims.append(cosine_sim(emb, embeddings[all_chunks[idx-1]["id"]]))
    if idx < len(all_chunks) - 1:
        sims.append(cosine_sim(emb, embeddings[all_chunks[idx+1]["id"]]))

    if not sims:
        return 0.5
    avg = sum(sims) / len(sims)
    penalty = 0.15 if idx == 0 or idx == len(all_chunks) - 1 else 0.0
    return min(1.0, max(0.0, avg - penalty))


def compute_fit(chunk: dict, profile: dict, graph_context: dict[str, float] | None = None) -> float:
    text = chunk["text"].lower()
    score = 0.0

    entities = profile.get("key_entities", [])
    if entities:
        score += 0.4 * sum(1 for e in entities if e.lower() in text) / len(entities)
    else:
        score += 0.2

    facts = profile.get("required_facts", [])
    if facts:
        score += 0.3 * sum(1 for f in facts if f.lower() in text) / len(facts)
    else:
        score += 0.15

    intent_kw = {
        "explain": ["because", "how", "why", "reason", "process"],
        "troubleshoot": ["error", "fix", "issue", "solution"],
        "compare": ["versus", "vs", "difference", "unlike"],
        "list": ["include", "such as", "example", "following"],
        "design": ["architecture", "structure", "pattern", "build"]
    }
    kws = intent_kw.get(profile.get("primary_intent", "explain"), [])
    if kws:
        score += 0.2 * min(1.0, sum(1 for k in kws if k in text) / 3)

    shape_ind = {
        "definition": ["is", "means", "refers to"],
        "list": ["1.", "2.", "first", "-"],
        "code": ["def ", "class ", "import", "return"],
    }
    if any(i in text for i in shape_ind.get(profile.get("expected_shape", ""), [])):
        score += 0.1

    # Phase 2: graph-aware scoring bonuses/penalties
    if graph_context is not None:
        score += graph_context.get("co_occurrence_bonus", 0.0)
        score -= graph_context.get("coherence_penalty", 0.0)
        score += graph_context.get("path_distance_weight", 0.0)

    return min(1.0, score)


def compute_drift(chunk: dict, all_chunks: list, embeddings: dict) -> float:
    idx = next((i for i, c in enumerate(all_chunks) if c["id"] == chunk["id"]), -1)
    if idx == -1 or len(all_chunks) < 3:
        return 0.0

    emb = embeddings[chunk["id"]]
    sims = []
    if idx > 0:
        sims.append(cosine_sim(emb, embeddings[all_chunks[idx-1]["id"]]))
    if idx < len(all_chunks) - 1:
        sims.append(cosine_sim(emb, embeddings[all_chunks[idx+1]["id"]]))

    if not sims:
        return 0.0
    avg = sum(sims) / len(sims)
    if avg < 0.5:
        severity = (0.5 - avg) / 0.5
        penalty = -severity * (1.5 if len(sims) == 2 and all(s < 0.5 for s in sims) else 1.0)
        return max(-1.0, penalty)
    return 0.0


def score_chunks(chunks: list, profile: dict, cswr_cfg: dict) -> list:
    domain = detect_domain(profile.get("query_text", ""))
    thresh = get_stability_thresh(domain)

    vw = cswr_cfg.get("vector_weight", 0.35)
    sw = cswr_cfg.get("local_stability_weight", 0.25)
    fw = cswr_cfg.get("question_fit_weight", 0.20)
    dw = cswr_cfg.get("drift_penalty_weight", 0.10)
    cw = cswr_cfg.get("project_coherence_weight", 0.10)

    # batch-embed all chunk texts once (was O(n*k) individual calls)
    texts = [c["text"] for c in chunks]
    if texts:
        emb_matrix = embed_batch(texts)
        embeddings = {c["id"]: emb_matrix[i] for i, c in enumerate(chunks)}
    else:
        embeddings = {}

    # Phase 2: compute graph-aware scoring context per chunk
    graph_contexts = _compute_graph_contexts(chunks, profile)

    for c in chunks:
        c["local_stability"] = compute_stability(c, chunks, embeddings)
        gc = graph_contexts.get(c["id"])
        c["question_fit"] = compute_fit(c, profile, graph_context=gc)
        c["drift_penalty"] = compute_drift(c, chunks, embeddings)
        # project coherence: precomputed at index time, stored in metadata
        proj_coherence = c.get("project_coherence", 1.0)

        if c["local_stability"] < thresh:
            c["drift_penalty"] -= (thresh - c["local_stability"]) * 0.5

        c["csw_score"] = max(0.0, vw * c["score"] + sw * c["local_stability"] +
                            fw * c["question_fit"] + dw * c["drift_penalty"] +
                            cw * proj_coherence)

    return sorted(chunks, key=lambda x: x["csw_score"], reverse=True)


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except ImportError:
        return len(text) // 4


def build_pack(center: dict, all_chunks: list, budget: int = 1800) -> dict:
    idx = next((i for i, c in enumerate(all_chunks) if c["id"] == center["id"]), -1)
    if idx == -1:
        return {"pack_id": str(uuid.uuid4()), "main_text": center["text"],
                "supporting_text": None, "section_header": "",
                "source_chunks": [center["id"]], "pack_csw_score": center["csw_score"]}

    main, supporting, sources = [center], [], [center["id"]]
    tokens = count_tokens(center["text"])

    i = idx - 1
    while i >= 0 and tokens < budget:
        prev = all_chunks[i]
        t = count_tokens(prev["text"])
        if prev.get("local_stability", 0) >= 0.65 and tokens + t <= budget:
            main.insert(0, prev)
            sources.append(prev["id"])
            tokens += t
        elif prev.get("csw_score", 0) > 0.4 and tokens + t <= budget:
            supporting.insert(0, prev)
            sources.append(prev["id"])
            tokens += t
            break
        else:
            break
        i -= 1

    i = idx + 1
    while i < len(all_chunks) and tokens < budget:
        nxt = all_chunks[i]
        t = count_tokens(nxt["text"])
        if nxt.get("local_stability", 0) >= 0.65 and tokens + t <= budget:
            main.append(nxt)
            sources.append(nxt["id"])
            tokens += t
        elif nxt.get("csw_score", 0) > 0.4 and tokens + t <= budget:
            supporting.append(nxt)
            sources.append(nxt["id"])
            tokens += t
            break
        else:
            break
        i += 1

    src = center.get("source", "")
    header = src.split("/")[-1].replace(".pdf", "").replace(".txt", "").replace("_", " ").title() if src else ""
    main_text = "\n\n".join(c["text"] for c in main)
    if header:
        main_text = f"[Section: {header}]\n\n{main_text}"

    return {"pack_id": str(uuid.uuid4()), "main_text": main_text,
            "supporting_text": "\n\n".join(c["text"] for c in supporting) if supporting else None,
            "section_header": header, "source_chunks": sources, "pack_csw_score": center["csw_score"]}


def check_answerable(pack: dict, profile: dict) -> tuple:
    """CSWR-threshold answerability. No LLM call, ~0.1 ms.
    Scores already computed by score_chunks(), so this is just a gate."""
    csw = pack.get("pack_csw_score", 0.0)
    if csw >= 0.45:
        return (True, 0.7)
    if csw >= 0.35:
        return (True, 0.5)
    return (False, 0.3)


def format_pack(pack: dict, rank: int) -> str:
    conf = pack.get('answerability_confidence', 0.0)
    csw = pack.get('pack_csw_score', 0.0)
    out = f"### Context Source #{rank+1} | Conf: {conf:.2f} | CSW: {csw:.2f}\n"
    if pack.get('section_header'):
        out += f"Section: {pack['section_header']}\n\n"
    out += f"Main:\n{pack.get('main_text', '').strip()}\n\n"
    if pack.get('supporting_text'):
        out += f"Supporting:\n{pack['supporting_text'].strip()}\n\n"
    return out + "---\n"


def retrieve_rag(query: str, mode: str = "chat", top_k: int = None, project: str | None = None) -> str:
    cswr_cfg = cfg.get("cswr", {})
    top_k = top_k or cswr_cfg.get("top_k", 20)

    profile = decompose_query(query, mode)
    profile["query_text"] = query

    vec = embed_text(query).reshape(1, 384)
    index = get_rag_index()

    # empty index — nothing to search
    if index.ntotal == 0:
        return ""

    search_k = min(top_k, index.ntotal)
    dists, idxs = index.search(vec, search_k)

    meta_path = _get_metadata_path()
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else []

    # project routing
    target_projects = [project] if project else route_to_projects(query)

    chunks = []
    for i in range(len(idxs[0])):
        idx, dist = int(idxs[0][i]), float(dists[0][i])
        if idx < 0 or idx >= len(meta):
            continue
        entry = meta[idx]
        chunk_project = entry.get("project", "default")
        if target_projects and chunk_project not in target_projects:
            continue
        chunks.append({
            "text": entry["text"],
            "source": entry["source"],
            "project": chunk_project,
            "project_coherence": entry.get("coherence", 1.0),
            "score": 1.0 / (1.0 + dist), "id": str(idx),
            "local_stability": 0.0, "question_fit": 0.0, "drift_penalty": 0.0, "csw_score": 0.0
        })

    chunks = score_chunks(chunks, profile, cswr_cfg)

    packs, used = [], set()
    min_score = cswr_cfg.get("min_csw_score", 0.25)
    budget = cswr_cfg.get("pack_token_budget", 1800)

    for c in chunks:
        if c["id"] in used or c["csw_score"] < min_score:
            continue
        pack = build_pack(c, chunks, budget)
        used.update(pack["source_chunks"])
        packs.append(pack)
        if len(packs) >= 4:
            break

    thresh = cswr_cfg.get("answerability_threshold", 0.5)
    filtered = []
    for p in packs:
        ok, conf = check_answerable(p, profile)
        if ok and conf >= thresh:
            p["answerability_confidence"] = conf
            filtered.append(p)

    if not filtered and packs:
        best = max(packs, key=lambda x: x["pack_csw_score"])
        best["answerability_confidence"] = 0.0
        filtered = [best]

    result = "\n".join(format_pack(p, i) for i, p in enumerate(filtered))
    logger.info(f"CSWR SUCCESS | query={query[:40]}... | packs={len(filtered)}")
    return result


def retrieve_rag_structured(query: str, top_k: int = 5, project: str | None = None) -> list:
    """DEPRECATED: RAG structured retrieval removed. Returns empty list."""
    import warnings
    warnings.warn(
        "retrieve_rag_structured() is deprecated. Use retrieve() for CAG+Graph.",
        DeprecationWarning, stacklevel=2,
    )
    return []


def _get_db_path() -> Path:
    """Return the database file path, creating parent dirs if needed."""
    db_rel = cfg["paths"]["db"]
    db_path = PROJECT_ROOT / db_rel
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _cag_hash(query: str) -> str:
    """Normalize and SHA-256 hash a query for CAG exact-match lookup."""
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()


def retrieve_cag(query: str, threshold: float = 0.75) -> list:
    """Retrieve from the cached answer graph (CAG).

    Three-tier lookup: SHA-256 exact match -> case-insensitive -> semantic similarity.
    The hash tier is new (Step 3 upgrade) and provides sub-millisecond cache hits.
    """
    db_path = _get_db_path()
    logger.debug(f"[CAG] Searching ({len(query)} chars, threshold={threshold})")

    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error:
        logger.debug("[CAG] Database not available")
        return []

    cur = conn.cursor()

    try:
        q = query.strip()
        q_hash = _cag_hash(q)

        # tier 1: SHA-256 exact match (fastest, normalized)
        # gracefully skip if key_hash column doesn't exist yet
        try:
            cur.execute("SELECT value, score FROM cache WHERE key_hash = ?", (q_hash,))
            row = cur.fetchone()
            if row and row[1] >= threshold:
                logger.debug(f"[CAG] HASH HIT: score={row[1]:.2f}")
                return [{"text": row[0], "source": "cag", "score": row[1]}]
        except sqlite3.OperationalError:
            pass  # key_hash column not yet added to schema

        # tier 2: raw string exact match (legacy keys without hash)
        cur.execute("SELECT value, score FROM cache WHERE key = ?", (q,))
        row = cur.fetchone()
        if row and row[1] >= threshold:
            logger.debug(f"[CAG] EXACT HIT: score={row[1]:.2f}")
            return [{"text": row[0], "source": "cag", "score": row[1]}]

        # tier 3: case-insensitive match
        cur.execute("SELECT value, score FROM cache WHERE LOWER(key) = LOWER(?)", (q,))
        row = cur.fetchone()
        if row and row[1] >= threshold:
            logger.debug(f"[CAG] CASE HIT: score={row[1]:.2f}")
            return [{"text": row[0], "source": "cag", "score": row[1]}]

        # tier 4: semantic similarity via embeddings
        q_emb = embed_text(q)
        cur.execute("SELECT key, value, score FROM cache")
        rows = cur.fetchall()
        if rows:
            keys = [r[0] for r in rows]
            key_embs = embed_batch(keys)
            sims = key_embs @ q_emb
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= 0.85:
                _, v, s = rows[best_idx]
                logger.debug(f"[CAG] SEMANTIC HIT: sim={best_sim:.2f}")
                return [{"text": v, "source": "cag", "score": s}]

        logger.debug("[CAG] NO HIT")
        return []
    finally:
        conn.close()


def _get_ontology_path() -> Path:
    """Return the ontology file path."""
    ont_rel = cfg["paths"]["ontology"]
    return PROJECT_ROOT / ont_rel


# Graph + embedding cache: avoids reloading JSON and re-embedding on every query
_graph_cache: Dict[str, Any] = {"graph": None, "node_embeddings": {}, "mtime": 0.0}

# Phase 2: GraphEngine lazy singleton
_graph_engine_cache: dict[str, "GraphEngine | bool | None"] = {"engine": None, "attempted": False}


def _get_graph_engine() -> "GraphEngine | None":
    """Lazily initialize the GraphEngine singleton. Returns None if unavailable."""
    if _graph_engine_cache["attempted"]:
        return _graph_engine_cache["engine"]  # type: ignore[return-value]
    _graph_engine_cache["attempted"] = True
    try:
        from backend.core.graph_engine import GraphEngine as _GE
        engine = _GE()
        _graph_engine_cache["engine"] = engine
        return engine
    except ImportError:
        logger.debug("GraphEngine dependencies not available; graph retrieval degraded")
        return None


def _compute_graph_contexts(chunks: list[dict[str, object]], profile: dict[str, object]) -> dict[str, dict[str, float]]:
    """Compute graph-aware scoring context per chunk. Passes pre-computed query embedding."""
    engine = _get_graph_engine()
    if engine is None or engine.node_count == 0:
        return {}
    query_text = str(profile.get("query_text", ""))
    if not query_text:
        return {}
    # pre-compute query embedding once, reuse across all chunks
    q_emb = embed_text(query_text)
    contexts: dict[str, dict[str, float]] = {}
    for c in chunks:
        scores = engine.compute_chunk_graph_scores(
            str(c["text"]), query_text, query_embedding=q_emb,
        )
        contexts[str(c["id"])] = {
            "co_occurrence_bonus": float(scores.get("co_occurrence_bonus", 0.0)),
            "coherence_penalty": float(scores.get("coherence_penalty", 0.0)),
            "path_distance_weight": float(scores.get("path_distance_weight", 0.0)),
        }
    return contexts


def resolve_entities(entities: list[dict[str, str | list[str]]]) -> list[dict[str, str | list[str]]]:
    """Deduplicate entities by embedding similarity. Delegates to GraphEngine."""
    from backend.core.graph_engine import GraphEngine as _GE
    engine = _get_graph_engine()
    if engine is None:
        engine = _GE()
        _graph_engine_cache["engine"] = engine
        _graph_engine_cache["attempted"] = True
    return engine.resolve_entities(entities)  # type: ignore[arg-type]


def build_entity_graph(chunks: list[dict[str, str]] | None = None) -> int:
    """Construct knowledge graph from document chunks. Uses FAISS metadata as fallback.

    Resets the cached engine, builds from chunks, persists to disk.
    Returns the number of entities added.
    """
    from backend.core.graph_engine import GraphEngine as _GE

    # fresh engine for a clean rebuild
    engine = _GE()
    engine.clear()
    _graph_engine_cache["engine"] = engine
    _graph_engine_cache["attempted"] = True

    if chunks is None:
        meta_path = _get_metadata_path()
        if meta_path.exists():
            chunks = json.loads(meta_path.read_text())

    if not chunks:
        return 0

    count = engine.build_from_chunks(chunks)
    engine.save()
    return count


def community_summarize(query: str, top_k: int = 3) -> list[dict[str, str | float]]:
    """Retrieve community-level summaries most relevant to the query.

    Embeds the query, scores each community by avg member similarity,
    returns the top-k community summaries as retrieval results.
    """
    engine = _get_graph_engine()
    if engine is None or engine.node_count == 0:
        return []

    q_emb = embed_text(query)
    community_scores: list[tuple[int, float]] = []

    for comm_id, members in engine.communities.items():
        if not members:
            continue
        # avg similarity of community members to query
        sims: list[float] = []
        for nid in members[:10]:
            if nid in engine._entity_embeddings:
                sims.append(float(np.dot(q_emb, engine._entity_embeddings[nid])))
        if sims:
            community_scores.append((comm_id, sum(sims) / len(sims)))

    community_scores.sort(key=lambda x: x[1], reverse=True)

    results: list[dict[str, str | float]] = []
    for comm_id, score in community_scores[:top_k]:
        info = engine.get_community_summary(comm_id)
        results.append({
            "text": f"Community {comm_id}: {info['summary']}",
            "score": score,
            "source": "graph_community",
            "id": f"community_{comm_id}",
            "member_count": info["member_count"],
            "entities": ", ".join(info["representative_entities"]),
        })

    return results


def _load_graph() -> nx.DiGraph:
    """Load or return cached ontology graph. Re-reads only if file changed."""
    path = _get_ontology_path()
    if not path.exists():
        return nx.DiGraph()

    mtime = path.stat().st_mtime
    if _graph_cache["graph"] is not None and _graph_cache["mtime"] == mtime:
        return _graph_cache["graph"]

    try:
        data = json.loads(path.read_text())
        G = nx.DiGraph()
        for n in data.get("nodes", []):
            G.add_node(n["id"], **n)
        for e in data.get("edges", []):
            G.add_edge(e["from"], e["to"], **{k: v for k, v in e.items() if k not in ["from", "to"]})
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Failed to load ontology graph: %s", e)
        return nx.DiGraph()

    # pre-embed all node texts once
    node_embeddings = {}
    for node in G.nodes():
        nd = G.nodes[node]
        txt = nd.get("description", "") or nd.get("label", str(node))
        if txt:
            node_embeddings[node] = embed_text(txt)

    _graph_cache["graph"] = G
    _graph_cache["node_embeddings"] = node_embeddings
    _graph_cache["mtime"] = mtime
    logger.info("Graph loaded and %d node embeddings cached", len(node_embeddings))
    return G


def retrieve_graph(query: str, max_hops: int = 2) -> List[Dict[str, Any]]:
    """Retrieve from knowledge graph. Uses GraphEngine (Phase 2) with Qdrant hybrid
    search when available, falls back to legacy ontology traversal."""
    # Phase 2: use GraphEngine for richer retrieval
    engine = _get_graph_engine()
    if engine is not None and engine.node_count > 0:
        results = engine.hybrid_search(query, top_k=5)
        return [dict(r) for r in results]

    # legacy fallback: static ontology graph
    G = _load_graph()
    if not G.number_of_nodes():
        return []

    q_emb = embed_text(query)
    best_node, best_score = None, -1.0

    for node, node_emb in _graph_cache["node_embeddings"].items():
        sim = cosine_sim(q_emb, node_emb)
        if sim > best_score:
            best_score, best_node = sim, node

    if not best_node:
        return []

    visited, results, queue = set(), [], [(best_node, 0)]
    while queue:
        cur, hops = queue.pop(0)
        if cur in visited or hops > max_hops:
            continue
        visited.add(cur)
        nd = G.nodes[cur]

        parts = [f"Entity: {nd.get('label', cur)}"]
        if nd.get('description'):
            parts.append(f"Description: {nd['description']}")
        for neighbor in G.neighbors(cur):
            ed = G.edges[cur, neighbor]
            parts.append(f"-> {ed.get('label', 'relates_to')} -> {G.nodes[neighbor].get('label', neighbor)}")
            if neighbor not in visited:
                queue.append((neighbor, hops + 1))

        results.append({"text": "\n".join(parts), "score": best_score * (0.8 ** hops),
                       "source": "graph", "id": str(cur)})
    return results


def retrieve(query: str, cag_weight: float = 1.0,
             graph_weight: float = 1.0, top_k: int = 5,
             project: str | None = None,
             rag_weight: float = 0.0, web_weight: float = 0.0) -> Dict[str, Any]:
    """Two-path retrieval: CAG-first with GraphRAG fallback.

    CAG is checked first for a cache hit. On miss, GraphRAG runs entity traversal.
    The rag_weight and web_weight params are kept for backward compatibility
    but ignored (both paths removed in Step 3).
    """
    # CAG-first: check cache before anything else
    cag = retrieve_cag(query)
    if cag and cag[0].get("score", 0) >= 0.85:
        # strong CAG hit, bypass GraphRAG entirely
        for r in cag:
            r["score"] *= cag_weight
            r["retriever"] = "cag"
        logger.info("[retrieve] CAG HIT, skipping GraphRAG")
        return {
            "rag": [], "cag": cag[:top_k], "graph": [], "web": [],
            "images": [], "web_status": "disabled",
        }

    # GraphRAG: entity traversal via graph_engine
    graph = retrieve_graph(query, 2)

    # multimodal image retrieval (kept, not on critical path)
    images: list[dict[str, object]] = []
    if cfg.get("multimodal", {}).get("enabled", False):
        try:
            from backend.core.multimodal import retrieve_images
            images = retrieve_images(query, top_k=top_k)  # type: ignore[assignment]
        except (ImportError, RuntimeError) as exc:
            logger.debug("Image retrieval skipped: %s", exc)

    for r in cag:
        r["score"] *= cag_weight
        r["retriever"] = "cag"
    for r in graph:
        r["score"] *= graph_weight
        r["retriever"] = "graph"
    for r in images:
        r["retriever"] = "image"  # type: ignore[index]

    for lst in [cag, graph]:
        lst.sort(key=lambda x: x["score"], reverse=True)

    return {
        "rag": [], "cag": cag[:top_k], "graph": graph[:top_k], "web": [],
        "images": images[:top_k],
        "web_status": "disabled",
    }

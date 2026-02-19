# Author: Bradley R. Kinnard
# retrievers.py - RAG/CAG/Graph/Web retrieval with CSWR stability filtering
# Originally built for personal offline use, now open-sourced for public benefit.

import faiss
import sqlite3
import json
import uuid
import logging
import os
import sys
import numpy as np
import networkx as nx
import httpx
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from backend.core.utils import embed_text, embed_batch, ensure_path, deterministic_id
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


def tavily_search(query: str) -> tuple[str, str]:
    """
    Search the web using Tavily API. Requires TAVILY_API_KEY env var.
    Returns (result_text, status) where status is 'success', 'no_api_key', 'disabled', or 'error'.
    """
    if not cfg.get("web", {}).get("enabled", False):
        logger.debug("Web search disabled in config")
        return "", "disabled"
    api_key = get_web_api_key()
    if not api_key:
        logger.warning("⚠️ Web search enabled but TAVILY_API_KEY not set - skipping web retrieval")
        return "", "no_api_key"

    logger.info(f"🌐 Tavily web search: {query[:50]}...")
    try:
        resp = httpx.post("https://api.tavily.com/search", json={
            "api_key": api_key, "query": query, "search_depth": "basic",
            "include_answer": True, "include_images": False,
            "max_results": cfg.get("web", {}).get("max_results", 3)
        }, timeout=cfg.get("web", {}).get("search_timeout", 10))

        if resp.status_code != 200:
            logger.warning(f"Tavily API returned {resp.status_code}")
            return "", "error"

        data = resp.json()
        parts = []
        if data.get("answer"):
            parts.append(f"{data['answer']}\n")
        for i, r in enumerate(data.get("results", [])[:3], 1):
            title = r.get('title', 'Source')
            url = r.get('url', '')
            content = r.get('content', '')
            parts.append(f"**{title}**\n{url}\n{content}\n")
        return "\n".join(parts), "success"
    except Exception as e:
        logger.warning("Tavily search failed: %s", e)
        return "", "error"


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
    """Return the FAISS index file path."""
    index_dir = PROJECT_ROOT / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir / "rag_index.faiss"


def _get_metadata_path() -> Path:
    """Return the metadata JSON file path."""
    index_dir = PROJECT_ROOT / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir / "metadata.json"


def build_rag_index() -> faiss.IndexFlatL2:
    """Build the RAG FAISS index from documents in data/docs."""
    docs_path = _get_docs_path()
    index_path = _get_index_path()
    cpu_index = faiss.IndexFlatL2(384)

    if not docs_path.exists():
        ensure_path(str(index_path))
        faiss.write_index(cpu_index, str(index_path))
        return cpu_index

    files = list(docs_path.rglob("*.txt")) + list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.pdf"))
    all_chunks = []

    for fpath in files:
        try:
            content = extract_pdf_text(fpath) if fpath.suffix.lower() == ".pdf" else fpath.read_text()
            from backend.core.utils import chunk_text
            for chunk in chunk_text(content, max_tokens=400):
                all_chunks.append({"text": chunk, "source": str(fpath.relative_to(docs_path))})
        except Exception as e:
            logger.warning(f"Failed to process {fpath}: {e}")

    if not all_chunks:
        ensure_path(str(index_path))
        faiss.write_index(cpu_index, str(index_path))
        return cpu_index

    texts = [c["text"] for c in all_chunks]
    cpu_index.add(embed_batch(texts))
    ensure_path(str(index_path))
    faiss.write_index(cpu_index, str(index_path))

    _get_metadata_path().write_text(json.dumps(
        [{"text": c["text"], "source": c["source"]} for c in all_chunks], indent=2))
    logger.info(f"Built RAG index with {len(all_chunks)} chunks")
    return cpu_index


def get_rag_index() -> faiss.IndexFlatL2:
    """Load or build the RAG FAISS index."""
    path = _get_index_path()
    if path.exists():
        logger.debug("RAG index loaded from disk")
        return faiss.read_index(str(path))
    logger.info("Building RAG index...")
    return build_rag_index()


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


def compute_fit(chunk: dict, profile: dict) -> float:
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

    vw = cswr_cfg.get("vector_weight", 0.4)
    sw = cswr_cfg.get("local_stability_weight", 0.3)
    fw = cswr_cfg.get("question_fit_weight", 0.2)
    dw = cswr_cfg.get("drift_penalty_weight", 0.1)

    # batch-embed all chunk texts once (was O(n*k) individual calls)
    texts = [c["text"] for c in chunks]
    if texts:
        emb_matrix = embed_batch(texts)
        embeddings = {c["id"]: emb_matrix[i] for i, c in enumerate(chunks)}
    else:
        embeddings = {}

    for c in chunks:
        c["local_stability"] = compute_stability(c, chunks, embeddings)
        c["question_fit"] = compute_fit(c, profile)
        c["drift_penalty"] = compute_drift(c, chunks, embeddings)

        if c["local_stability"] < thresh:
            c["drift_penalty"] -= (thresh - c["local_stability"]) * 0.5

        c["csw_score"] = max(0.0, vw * c["score"] + sw * c["local_stability"] +
                            fw * c["question_fit"] + dw * c["drift_penalty"])

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
    try:
        from ollama import Client
        client = Client(host=cfg["llm"]["host"])
        resp = client.chat(model=cfg["llm"]["model"], messages=[
            {"role": "system", "content": "Respond YES or NO followed by confidence 0.0-1.0. Can this context answer the question?"},
            {"role": "user", "content": f"Q: {profile.get('original_query', '')}\n\nContext:\n{pack['main_text'][:1500]}"}
        ], options={"temperature": 0.0, "num_predict": 10}, stream=False)

        parts = resp["message"]["content"].strip().upper().split()
        verdict = parts[0] if parts else "NO"
        conf = float(parts[1]) if len(parts) > 1 else 0.5
        return (verdict.startswith("YES"), max(0.0, min(1.0, conf)))
    except Exception as e:
        logger.debug("Answerability check fell back to heuristic: %s", e)
        csw = pack.get("pack_csw_score", 0.0)
        if csw >= 0.45: return (True, 0.7)
        if csw >= 0.35: return (True, 0.5)
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


def retrieve_rag(query: str, mode: str = "chat", top_k: int = None) -> str:
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

    chunks = []
    for i in range(len(idxs[0])):
        idx, dist = int(idxs[0][i]), float(dists[0][i])
        # FAISS returns -1 for missing neighbors
        if idx < 0 or idx >= len(meta):
            continue
        chunks.append({
            "text": meta[idx]["text"],
            "source": meta[idx]["source"],
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


def retrieve_rag_structured(query: str, top_k: int = 5) -> list:
    cswr_cfg = cfg.get("cswr", {})
    top_k = top_k or cswr_cfg.get("top_k", 20)

    profile = decompose_query(query, "chat")
    profile["query_text"] = query

    vec = embed_text(query).reshape(1, 384)
    index = get_rag_index()

    # empty index — nothing to search
    if index.ntotal == 0:
        return []

    search_k = min(top_k, index.ntotal)
    dists, idxs = index.search(vec, search_k)

    meta_path = _get_metadata_path()
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else []

    chunks = []
    for i in range(len(idxs[0])):
        idx, dist = int(idxs[0][i]), float(dists[0][i])
        # FAISS returns -1 for missing neighbors
        if idx < 0 or idx >= len(meta):
            continue
        chunks.append({
            "text": meta[idx]["text"],
            "source": meta[idx]["source"],
            "score": 1.0 / (1.0 + dist), "id": str(idx),
            "local_stability": 0.0, "question_fit": 0.0, "drift_penalty": 0.0, "csw_score": 0.0
        })

    chunks = score_chunks(chunks, profile, cswr_cfg)
    logger.info(f"CSWR structured | query='{query[:50]}...' | {min(top_k, len(chunks))} chunks")
    return [{"text": c["text"], "score": c["csw_score"], "source": c["source"], "id": c["id"]}
            for c in chunks[:top_k]]


def _get_db_path() -> Path:
    """Return the database file path, creating parent dirs if needed."""
    db_rel = cfg["paths"]["db"]
    db_path = PROJECT_ROOT / db_rel
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def retrieve_cag(query: str, threshold: float = 0.75) -> list:
    """Retrieve from the cached answer graph (CAG). Uses embedding cache for semantic matching."""
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

        cur.execute("SELECT value, score FROM cache WHERE key = ?", (q,))
        row = cur.fetchone()
        if row and row[1] >= threshold:
            logger.debug(f"[CAG] EXACT HIT: score={row[1]:.2f}")
            return [{"text": row[0], "source": "cag", "score": row[1]}]

        cur.execute("SELECT value, score FROM cache WHERE LOWER(key) = LOWER(?)", (q,))
        row = cur.fetchone()
        if row and row[1] >= threshold:
            logger.debug(f"[CAG] CASE HIT: score={row[1]:.2f}")
            return [{"text": row[0], "source": "cag", "score": row[1]}]

        q_emb = embed_text(q)
        cur.execute("SELECT key, value, score FROM cache")
        rows = cur.fetchall()
        if rows:
            # batch-embed all keys at once instead of O(n) single calls
            keys = [r[0] for r in rows]
            key_embs = embed_batch(keys)
            sims = key_embs @ q_emb  # dot products (normalized vectors)
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
    """Retrieve related entities from the knowledge graph. Uses cached node embeddings."""
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


def retrieve_web(query: str, max_results: int = 3) -> tuple[List[Dict[str, Any]], str]:
    """
    Retrieve web results. Returns (results_list, status).
    Status is 'success', 'no_api_key', 'disabled', or 'error'.
    """
    if not cfg.get("web", {}).get("enabled", False):
        return [], "disabled"
    ctx, status = tavily_search(query)
    if not ctx:
        return [], status
    return [{"text": ctx, "url": "web", "score": 0.95, "source": "web",
             "title": f"Web: {query[:50]}"}], status


def retrieve(query: str, rag_weight: float = 1.0, cag_weight: float = 1.0,
             graph_weight: float = 1.0, web_weight: float = 1.0, top_k: int = 5) -> Dict[str, Any]:
    """Main retrieval function. Returns results and web_status."""
    rag = retrieve_rag_structured(query, top_k=top_k * 2)
    cag = retrieve_cag(query)
    graph = retrieve_graph(query, max_hops=2)

    web_status = "disabled"
    if cfg.get("web", {}).get("enabled", False):
        web, web_status = retrieve_web(query)
    else:
        web = []

    for r in rag: r["score"] *= rag_weight; r["retriever"] = "rag"
    for r in cag: r["score"] *= cag_weight; r["retriever"] = "cag"
    for r in graph: r["score"] *= graph_weight; r["retriever"] = "graph"
    for r in web: r["score"] *= web_weight; r["retriever"] = "web"

    for lst in [rag, cag, graph, web]:
        lst.sort(key=lambda x: x["score"], reverse=True)

    return {
        "rag": rag[:top_k], "cag": cag[:top_k], "graph": graph[:top_k], "web": web[:top_k],
        "web_status": web_status
    }

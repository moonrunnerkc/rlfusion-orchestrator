# Author: Bradley R. Kinnard
"""Knowledge graph engine: entity resolution, Leiden community detection,
multi-hop traversal, and vector-graph hybrid retrieval.

Built for Phase 2 of the RLFusion upgrade plan. Uses NetworkX for graph
structure, Qdrant (in-memory) for entity vector search, and leidenalg
for community detection. All optional components degrade gracefully.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import networkx as nx
import numpy as np

from backend.config import PROJECT_ROOT, cfg
from backend.core.utils import deterministic_id, embed_batch, embed_text

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# ── Optional: Qdrant for entity vector + payload search ─────────────
_HAS_QDRANT = False
try:
    from qdrant_client import QdrantClient as _QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    _HAS_QDRANT = True
except ImportError:
    logger.info("qdrant-client not installed; entity search uses numpy fallback")

# ── Optional: Leiden community detection ─────────────────────────────
_HAS_LEIDEN = False
try:
    import igraph as ig
    import leidenalg

    _HAS_LEIDEN = True
except ImportError:
    logger.info("leidenalg/igraph not installed; using connected-components fallback")

# Common capitalized words to skip during entity extraction
_SKIP_CAPS = frozenset({
    "THE", "AND", "FOR", "NOT", "BUT", "ARE", "WAS", "HAS", "ITS",
    "CAN", "MAY", "ALL", "USE", "THIS", "THAT", "WITH", "FROM",
    "WILL", "HAVE", "BEEN", "ALSO", "EACH", "WHEN", "THAN", "THEN",
    "INTO", "JUST", "VERY", "SOME", "ONLY", "MOST", "MANY", "SUCH",
    "NOTE", "MUST", "HERE", "DOES", "WHAT", "THEY", "THEM",
    "TODO", "NONE", "TRUE", "FALSE", "NULL", "ELSE", "THEN",
})


# ── TypedDicts ───────────────────────────────────────────────────────

class EntityNode(TypedDict, total=False):
    """Structured entity for the knowledge graph."""
    id: str
    label: str
    description: str
    entity_type: str
    source_chunks: list[str]
    community_id: int


class CommunityInfo(TypedDict):
    """Summary of a detected community cluster."""
    community_id: int
    member_count: int
    summary: str
    representative_entities: list[str]


class GraphSearchResult(TypedDict, total=False):
    """Single result from graph-based retrieval."""
    text: str
    score: float
    source: str
    id: str
    path: list[str]
    community_id: int


class ChunkGraphScores(TypedDict):
    """Graph-aware scoring components for CSWR integration."""
    co_occurrence_bonus: float
    coherence_penalty: float
    path_distance_weight: float


# ── Helpers ──────────────────────────────────────────────────────────

def _entity_point_id(entity_id: str) -> int:
    """Stable integer ID for Qdrant point storage."""
    return int(hashlib.md5(entity_id.encode()).hexdigest()[:15], 16)


# ── GraphEngine ──────────────────────────────────────────────────────

class GraphEngine:
    """Knowledge graph with entity resolution, community detection, hybrid search.

    Manages a directed graph of entities extracted from document chunks.
    Qdrant in-memory handles entity vector + payload search when available,
    numpy cosine similarity otherwise. Leiden detects topic communities,
    connected components is the fallback.
    """

    def __init__(
        self,
        graph_path: Path | None = None,
        ontology_path: Path | None = None,
    ) -> None:
        graph_cfg = cfg.get("graph", {})
        self._graph_path = graph_path or (
            PROJECT_ROOT / graph_cfg.get("entity_graph_path", "data/entity_graph.json")
        )
        self._ontology_path = ontology_path or (
            PROJECT_ROOT / cfg["paths"]["ontology"]
        )
        self._graph: nx.DiGraph = nx.DiGraph()
        self._entity_embeddings: dict[str, np.ndarray] = {}
        self._communities: dict[int, list[str]] = {}
        self._qdrant_id_map: dict[str, int] = {}

        # Qdrant in-memory for entity vector + payload search
        self._qdrant: QdrantClient | None = None
        if _HAS_QDRANT:
            self._qdrant = _QdrantClient(location=":memory:")
            self._qdrant.create_collection(
                collection_name="entities",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        self._load()

    def _load(self) -> None:
        """Load graph data from ontology.json and entity_graph.json (if they exist)."""
        for path in (self._ontology_path, self._graph_path):
            if not path or not path.exists():
                continue
            try:
                raw = path.read_text().strip()
                if not raw:
                    continue
                data = json.loads(raw)
                for n in data.get("nodes", []):
                    nid = n.get("id", "")
                    if not nid:
                        continue
                    attrs = {k: v for k, v in n.items() if k not in ("id", "embedding")}
                    self._graph.add_node(nid, **attrs)
                    if "embedding" in n:
                        emb = np.array(n["embedding"], dtype=np.float32)
                        self._entity_embeddings[nid] = emb
                        self._index_entity(nid, emb, n)

                for e in data.get("edges", []):
                    src, tgt = e.get("from", ""), e.get("to", "")
                    if src and tgt and src in self._graph and tgt in self._graph:
                        edge_attrs = {k: v for k, v in e.items() if k not in ("from", "to")}
                        self._graph.add_edge(src, tgt, **edge_attrs)

                # rebuild community assignment from saved node attributes
                for nid in self._graph.nodes():
                    cid = self._graph.nodes[nid].get("community_id", -1)
                    if cid >= 0:
                        self._communities.setdefault(cid, []).append(nid)

            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to load graph from %s: %s", path, exc)

        # embed any nodes loaded without embeddings
        if self._graph.number_of_nodes() > 0:
            unembedded = [
                nid for nid in self._graph.nodes()
                if nid not in self._entity_embeddings
            ]
            if unembedded:
                texts = [
                    self._graph.nodes[nid].get("description", "")
                    or self._graph.nodes[nid].get("label", nid)
                    for nid in unembedded
                ]
                emb_matrix = embed_batch(texts)
                for i, nid in enumerate(unembedded):
                    self._entity_embeddings[nid] = emb_matrix[i]
                    self._index_entity(nid, emb_matrix[i], dict(self._graph.nodes[nid]))

            logger.info(
                "GraphEngine loaded: %d nodes, %d edges, %d communities",
                self.node_count, self.edge_count, len(self._communities),
            )

    # ── Qdrant integration ───────────────────────────────────────────

    def _index_entity(self, nid: str, embedding: np.ndarray, attrs: dict[str, object]) -> None:
        """Upsert entity into the Qdrant in-memory index."""
        if self._qdrant is None:
            return
        point_id = _entity_point_id(nid)
        self._qdrant_id_map[nid] = point_id
        self._qdrant.upsert(
            collection_name="entities",
            points=[PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "node_id": nid,
                    "label": str(attrs.get("label", "")),
                    "entity_type": str(attrs.get("entity_type", "")),
                    "community_id": int(str(attrs.get("community_id", -1) or -1)),
                },
            )],
        )

    def _find_seed_entities(
        self,
        query_embedding: np.ndarray,
        max_seeds: int = 3,
    ) -> list[tuple[str, float]]:
        """Top entities by vector similarity. Uses Qdrant when available, numpy otherwise."""
        # Qdrant path
        if self._qdrant is not None and self._qdrant_id_map:
            try:
                response = self._qdrant.query_points(
                    collection_name="entities",
                    query=query_embedding.tolist(),
                    limit=max_seeds,
                )
                return [
                    (hit.payload["node_id"], float(hit.score))
                    for hit in response.points
                    if hit.payload
                ]
            except (RuntimeError, KeyError, AttributeError) as exc:
                logger.warning("Qdrant search failed, numpy fallback: %s", exc)

        # numpy fallback
        scored: list[tuple[str, float]] = []
        for nid, emb in self._entity_embeddings.items():
            sim = float(np.dot(query_embedding, emb))
            scored.append((nid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_seeds]

    # ── Core graph operations ────────────────────────────────────────

    def save(self) -> None:
        """Persist enriched entity graph to disk."""
        nodes: list[dict[str, object]] = []
        for nid in self._graph.nodes():
            nd: dict[str, object] = dict(self._graph.nodes[nid])
            nd["id"] = nid
            if nid in self._entity_embeddings:
                nd["embedding"] = self._entity_embeddings[nid].tolist()
            nodes.append(nd)

        edges: list[dict[str, object]] = []
        for u, v in self._graph.edges():
            ed: dict[str, object] = dict(self._graph.edges[u, v])
            ed["from"] = u
            ed["to"] = v
            edges.append(ed)

        communities_data: dict[str, list[str]] = {
            str(k): v for k, v in self._communities.items()
        }

        data = {"nodes": nodes, "edges": edges, "communities": communities_data}
        self._graph_path.parent.mkdir(parents=True, exist_ok=True)
        self._graph_path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("GraphEngine saved: %d nodes, %d edges to %s",
                     len(nodes), len(edges), self._graph_path)

    def add_entity(self, entity: EntityNode) -> str:
        """Add or update an entity node. Returns the node ID."""
        nid = entity.get("id") or deterministic_id(entity.get("label", ""))
        if not nid:
            raise ValueError("Entity must have 'id' or 'label'")

        attrs = {k: v for k, v in entity.items() if k != "id"}
        self._graph.add_node(nid, **attrs)

        text = entity.get("description", "") or entity.get("label", "")
        if text:
            emb = embed_text(text)
            self._entity_embeddings[nid] = emb
            self._index_entity(nid, emb, dict(entity))

        return nid

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        label: str = "relates_to",
        **attrs: object,
    ) -> None:
        """Add a directed edge. Increments weight if edge already exists."""
        if source_id not in self._graph:
            raise ValueError(f"Source entity '{source_id}' not in graph")
        if target_id not in self._graph:
            raise ValueError(f"Target entity '{target_id}' not in graph")

        if self._graph.has_edge(source_id, target_id):
            existing = self._graph.edges[source_id, target_id]
            existing["weight"] = existing.get("weight", 1.0) + 1.0
        else:
            self._graph.add_edge(source_id, target_id, label=label, weight=1.0, **attrs)

    # ── Entity resolution ────────────────────────────────────────────

    def resolve_entities(
        self,
        entities: list[EntityNode],
        threshold: float | None = None,
    ) -> list[EntityNode]:
        """Deduplicate entities whose embeddings exceed the similarity threshold.

        Groups near-duplicates, keeps the entity with the longest description
        as canonical, and merges source_chunks from all group members.
        """
        if not entities:
            return []

        graph_cfg = cfg.get("graph", {})
        thresh = threshold if threshold is not None else graph_cfg.get(
            "entity_similarity_threshold", 0.92
        )

        texts = [e.get("description", "") or e.get("label", "") for e in entities]
        embeddings = embed_batch(texts)

        merged: list[EntityNode] = []
        used: set[int] = set()

        for i, entity in enumerate(entities):
            if i in used:
                continue
            group = [entity]
            group_chunks: list[str] = list(entity.get("source_chunks", []))

            for j in range(i + 1, len(entities)):
                if j in used:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= thresh:
                    group.append(entities[j])
                    group_chunks.extend(entities[j].get("source_chunks", []))
                    used.add(j)

            # canonical = richest description in group
            canonical = max(group, key=lambda e: len(e.get("description", "")))
            canonical["source_chunks"] = list(set(group_chunks))
            canonical["id"] = canonical.get("id") or deterministic_id(
                canonical.get("label", "").lower()
            )
            merged.append(canonical)

        return merged

    # ── Community detection ──────────────────────────────────────────

    def detect_communities(self) -> dict[int, list[str]]:
        """Leiden algorithm for community clustering. Falls back to connected components."""
        if self._graph.number_of_nodes() < 2:
            self._communities = {}
            return {}

        undirected = self._graph.to_undirected()

        if _HAS_LEIDEN and undirected.number_of_edges() > 0:
            try:
                ig_graph = ig.Graph.from_networkx(undirected)
                partition = leidenalg.find_partition(
                    ig_graph, leidenalg.ModularityVertexPartition,
                )
                node_list = list(undirected.nodes())
                communities: dict[int, list[str]] = {}
                for comm_id, members in enumerate(partition):
                    member_ids = [node_list[m] for m in members]
                    communities[comm_id] = member_ids
                    for nid in member_ids:
                        self._graph.nodes[nid]["community_id"] = comm_id

                self._communities = communities
                logger.info("Leiden detected %d communities", len(communities))
                return communities
            except (RuntimeError, ValueError) as exc:
                logger.warning("Leiden failed, falling back to components: %s", exc)

        # fallback: connected components
        communities = {}
        for comm_id, component in enumerate(nx.connected_components(undirected)):
            member_ids = list(component)
            communities[comm_id] = member_ids
            for nid in member_ids:
                self._graph.nodes[nid]["community_id"] = comm_id

        self._communities = communities
        return communities

    def get_community_summary(self, community_id: int) -> CommunityInfo:
        """Summarize a community by its member entities."""
        members = self._communities.get(community_id, [])
        if not members:
            return CommunityInfo(
                community_id=community_id,
                member_count=0,
                summary="",
                representative_entities=[],
            )

        labels = [
            self._graph.nodes[nid].get("label", nid)
            for nid in members[:10]
        ]
        descriptions = [
            self._graph.nodes[nid].get("description", "")
            for nid in members
            if self._graph.nodes[nid].get("description")
        ]
        summary = "; ".join(descriptions[:3]) if descriptions else "; ".join(labels)

        return CommunityInfo(
            community_id=community_id,
            member_count=len(members),
            summary=summary,
            representative_entities=labels[:5],
        )

    # ── Traversal & search ───────────────────────────────────────────

    def multi_hop_traverse(
        self,
        start_node: str,
        max_hops: int | None = None,
    ) -> list[GraphSearchResult]:
        """BFS from a start node with evidence paths and hop-decay scoring."""
        if start_node not in self._graph:
            return []

        graph_cfg = cfg.get("graph", {})
        hops = max_hops if max_hops is not None else graph_cfg.get("max_hops", 2)
        decay = graph_cfg.get("path_distance_decay", 0.8)

        visited: set[str] = set()
        results: list[GraphSearchResult] = []
        queue: list[tuple[str, int, list[str]]] = [(start_node, 0, [start_node])]

        while queue:
            current, depth, path = queue.pop(0)
            if current in visited or depth > hops:
                continue
            visited.add(current)

            nd = self._graph.nodes[current]
            parts = [f"Entity: {nd.get('label', current)}"]
            if nd.get("description"):
                parts.append(f"Description: {nd['description']}")

            for neighbor in self._graph.neighbors(current):
                edge = self._graph.edges[current, neighbor]
                neighbor_label = self._graph.nodes[neighbor].get("label", neighbor)
                parts.append(f"-> {edge.get('label', 'relates_to')} -> {neighbor_label}")
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1, path + [neighbor]))

            results.append(GraphSearchResult(
                text="\n".join(parts),
                score=decay ** depth,
                source="graph",
                id=current,
                path=list(path),
                community_id=nd.get("community_id", -1),
            ))

        return results

    def hybrid_search(self, query: str, top_k: int = 5) -> list[GraphSearchResult]:
        """Vector similarity over entities + graph traversal.

        1. Find top seed entities by embedding similarity (Qdrant or numpy)
        2. Expand each seed via multi-hop BFS
        3. Merge and rank by combined score (seed similarity x hop decay)
        """
        if not self._entity_embeddings:
            return []

        q_emb = embed_text(query)
        seeds = self._find_seed_entities(q_emb, max_seeds=3)

        results: list[GraphSearchResult] = []
        seen: set[str] = set()

        for seed_id, seed_score in seeds:
            for r in self.multi_hop_traverse(seed_id):
                rid = r.get("id", "")
                if rid not in seen:
                    seen.add(rid)
                    r["score"] = float(r.get("score", 0.0)) * seed_score
                    results.append(r)

        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results[:top_k]

    # ── Build from chunks ────────────────────────────────────────────

    def build_from_chunks(self, chunks: list[dict[str, str]]) -> int:
        """Build graph from document chunks: extract entities, link co-occurrences.

        Returns the number of entities added.
        """
        if not chunks:
            return 0

        # extract entities per chunk
        chunk_entities: dict[int, list[EntityNode]] = {}
        all_entities: list[EntityNode] = []

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            source = chunk.get("source", "")
            extracted = extract_entities_heuristic(text, source)
            chunk_entities[i] = extracted
            all_entities.extend(extracted)

        if not all_entities:
            return 0

        # deduplicate across all chunks
        resolved = self.resolve_entities(all_entities)

        # add resolved entities to graph
        label_to_id: dict[str, str] = {}
        for entity in resolved:
            nid = self.add_entity(entity)
            label_to_id[entity.get("label", "").lower()] = nid

        # co-occurrence edges: entities from the same chunk get linked
        for _i, extracted in chunk_entities.items():
            ids_in_chunk: list[str] = []
            for ent in extracted:
                label_lower = ent.get("label", "").lower()
                if label_lower in label_to_id:
                    ids_in_chunk.append(label_to_id[label_lower])

            unique_ids = list(set(ids_in_chunk))
            for a_idx in range(len(unique_ids)):
                for b_idx in range(a_idx + 1, len(unique_ids)):
                    aid, bid = unique_ids[a_idx], unique_ids[b_idx]
                    if aid != bid:
                        self.add_relation(aid, bid, label="co_occurs")
                        self.add_relation(bid, aid, label="co_occurs")

        # cluster into communities
        if self._graph.number_of_nodes() >= 2:
            self.detect_communities()

        logger.info(
            "Built graph from %d chunks: %d entities, %d edges",
            len(chunks), self.node_count, self.edge_count,
        )
        return len(resolved)

    # ── CSWR graph-aware scoring ─────────────────────────────────────

    def compute_chunk_graph_scores(
        self,
        chunk_text: str,
        query: str,
        query_embedding: np.ndarray | None = None,
    ) -> ChunkGraphScores:
        """Compute graph-aware CSWR scoring for a single chunk.

        - co_occurrence_bonus: chunk entities overlap with query's graph neighborhood
        - coherence_penalty: chunk entities span too many unrelated communities
        - path_distance_weight: graph proximity between chunk and query entities
        """
        graph_cfg = cfg.get("graph", {})
        max_bonus = graph_cfg.get("co_occurrence_bonus", 0.15)
        max_penalty = graph_cfg.get("coherence_penalty", 0.10)

        if self.node_count == 0:
            return ChunkGraphScores(
                co_occurrence_bonus=0.0,
                coherence_penalty=0.0,
                path_distance_weight=0.0,
            )

        # entities mentioned in this chunk
        chunk_lower = chunk_text.lower()
        chunk_entity_ids: list[str] = []
        for nid in self._graph.nodes():
            label = self._graph.nodes[nid].get("label", "").lower()
            if label and len(label) > 2 and label in chunk_lower:
                chunk_entity_ids.append(nid)

        if not chunk_entity_ids:
            return ChunkGraphScores(
                co_occurrence_bonus=0.0,
                coherence_penalty=0.0,
                path_distance_weight=0.0,
            )

        # query-relevant seed entities (use pre-computed embedding when available)
        q_emb = query_embedding if query_embedding is not None else embed_text(query)
        query_seeds = self._find_seed_entities(q_emb, max_seeds=3)
        query_entity_ids = {sid for sid, _ in query_seeds}

        # co-occurrence: chunk entities that neighbor query entities in the graph
        co_hits = 0.0
        for cid in chunk_entity_ids:
            if cid in query_entity_ids:
                co_hits += 1.0
                continue
            chunk_neighbors = set(self._graph.neighbors(cid))
            if chunk_neighbors & query_entity_ids:
                co_hits += 1.0
        co_occurrence_bonus = max_bonus * min(1.0, co_hits / len(chunk_entity_ids))

        # community coherence: penalize chunks spanning many unrelated communities
        communities_hit: set[int] = set()
        for cid in chunk_entity_ids:
            comm = self._graph.nodes[cid].get("community_id", -1)
            if comm >= 0:
                communities_hit.add(comm)
        coherence_penalty = 0.0
        if len(communities_hit) > 2:
            coherence_penalty = max_penalty * min(1.0, (len(communities_hit) - 1) / 3)

        # path distance: shortest path from any chunk entity to any query entity
        path_weight = 0.0
        if query_entity_ids and chunk_entity_ids:
            undirected = self._graph.to_undirected()
            min_dist = float("inf")
            for c_eid in chunk_entity_ids[:3]:
                for q_eid in query_entity_ids:
                    try:
                        dist = nx.shortest_path_length(undirected, c_eid, q_eid)
                        min_dist = min(min_dist, dist)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
            if min_dist < float("inf"):
                decay = graph_cfg.get("path_distance_decay", 0.8)
                path_weight = 0.1 * (decay ** min_dist)

        return ChunkGraphScores(
            co_occurrence_bonus=co_occurrence_bonus,
            coherence_penalty=coherence_penalty,
            path_distance_weight=path_weight,
        )

    # ── Housekeeping ─────────────────────────────────────────────────

    def clear(self) -> None:
        """Reset graph to empty state."""
        self._graph.clear()
        self._entity_embeddings.clear()
        self._communities.clear()
        self._qdrant_id_map.clear()
        if self._qdrant is not None and _HAS_QDRANT:
            self._qdrant.delete_collection("entities")
            self._qdrant.create_collection(
                collection_name="entities",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def communities(self) -> dict[int, list[str]]:
        return dict(self._communities)

    @property
    def graph(self) -> nx.DiGraph:
        """Direct access to the underlying NetworkX graph (read-only intent)."""
        return self._graph


# ── Entity extraction ────────────────────────────────────────────────

def extract_entities_heuristic(text: str, source: str = "") -> list[EntityNode]:
    """Extract entity candidates using capitalized phrases, backtick terms, and acronyms.

    Not a full NER pipeline, but sufficient for document-level extraction against
    technical and general-purpose text.
    """
    entities: list[EntityNode] = []
    seen: set[str] = set()
    min_len = cfg.get("graph", {}).get("min_entity_length", 3)

    # capitalized multi-word phrases (2-4 words): "Machine Learning", "Chunk Stability"
    for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', text):
        label = match.group(1)
        normalized = label.lower()
        if normalized in seen or len(label) < min_len:
            continue
        seen.add(normalized)
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 80)
        entities.append(EntityNode(
            id=deterministic_id(normalized),
            label=label,
            description=text[start:end].strip().replace("\n", " "),
            entity_type="concept",
            source_chunks=[source] if source else [],
        ))

    # backtick-quoted terms: `FusionEnv`, `embed_text`
    for match in re.finditer(r'`([^`]{2,40})`', text):
        term = match.group(1).strip()
        normalized = term.lower()
        if normalized in seen or len(term) < min_len:
            continue
        seen.add(normalized)
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 80)
        entities.append(EntityNode(
            id=deterministic_id(normalized),
            label=term,
            description=text[start:end].strip().replace("\n", " "),
            entity_type="technical",
            source_chunks=[source] if source else [],
        ))

    # ALL-CAPS acronyms (3-10 letters), skip common English words
    for match in re.finditer(r'\b([A-Z]{3,10})\b', text):
        acronym = match.group(1)
        if acronym in _SKIP_CAPS or acronym in seen:
            continue
        seen.add(acronym)
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 80)
        entities.append(EntityNode(
            id=deterministic_id(acronym.lower()),
            label=acronym,
            description=text[start:end].strip().replace("\n", " "),
            entity_type="acronym",
            source_chunks=[source] if source else [],
        ))

    return entities

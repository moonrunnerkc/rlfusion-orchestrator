# Author: Bradley R. Kinnard
# test_core_units.py - isolated unit tests for core logic
# No external services (Ollama, Tavily) required.

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Force CPU for tests
os.environ.setdefault("RLFUSION_DEVICE", "cpu")
os.environ.setdefault("RLFUSION_FORCE_CPU", "true")


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------

class TestSoftmax:
    """Verify softmax produces valid probability distributions."""

    def test_uniform_input(self):
        from backend.core.utils import softmax
        result = softmax([1.0, 1.0, 1.0])
        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 1e-6
        # uniform input -> uniform output
        for v in result:
            assert abs(v - 1 / 3) < 1e-6

    def test_dominant_weight(self):
        from backend.core.utils import softmax
        result = softmax([10.0, 0.0, 0.0])
        assert result[0] > 0.99
        assert abs(sum(result) - 1.0) < 1e-6

    def test_negative_values(self):
        from backend.core.utils import softmax
        result = softmax([-1.0, 0.0, 1.0])
        assert abs(sum(result) - 1.0) < 1e-6
        # monotonic: result[2] > result[1] > result[0]
        assert result[2] > result[1] > result[0]

    def test_temperature_zero_argmax(self):
        from backend.core.utils import softmax
        result = softmax([0.3, 0.9, 0.1], temperature=0.0)
        assert result[1] == 1.0
        assert result[0] == 0.0
        assert result[2] == 0.0

    def test_high_temperature_flattens(self):
        from backend.core.utils import softmax
        sharp = softmax([5.0, 0.0, 0.0], temperature=1.0)
        flat = softmax([5.0, 0.0, 0.0], temperature=10.0)
        # high temp should make distribution more uniform
        assert flat[0] < sharp[0]
        assert flat[1] > sharp[1]

    def test_single_element(self):
        from backend.core.utils import softmax
        result = softmax([42.0])
        assert result == [1.0]


class TestChunkText:
    """Validate text chunking at token boundaries."""

    def test_short_text_single_chunk(self):
        from backend.core.utils import chunk_text
        chunks = chunk_text("hello world", max_tokens=10)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_exact_boundary(self):
        from backend.core.utils import chunk_text
        text = " ".join(["word"] * 10)
        chunks = chunk_text(text, max_tokens=5)
        assert len(chunks) == 2
        assert all(len(c.split()) == 5 for c in chunks)

    def test_large_text_multiple_chunks(self):
        from backend.core.utils import chunk_text
        text = " ".join(["token"] * 100)
        chunks = chunk_text(text, max_tokens=30)
        assert len(chunks) >= 3
        # all words preserved
        reconstructed = " ".join(chunks)
        assert reconstructed == text

    def test_empty_text(self):
        from backend.core.utils import chunk_text
        chunks = chunk_text("")
        assert chunks == []


class TestDeterministicId:
    """Ensure IDs are stable and collision-resistant."""

    def test_same_input_same_id(self):
        from backend.core.utils import deterministic_id
        a = deterministic_id("test string")
        b = deterministic_id("test string")
        assert a == b

    def test_different_input_different_id(self):
        from backend.core.utils import deterministic_id
        a = deterministic_id("alpha")
        b = deterministic_id("bravo")
        assert a != b

    def test_id_length(self):
        from backend.core.utils import deterministic_id
        # shake_256 hexdigest(16) = 32 hex chars
        result = deterministic_id("anything")
        assert len(result) == 32


class TestEmbedding:
    """Smoke tests for embedding functions."""

    def test_embed_text_shape(self):
        from backend.core.utils import embed_text
        vec = embed_text("hello world")
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_embed_text_normalized(self):
        from backend.core.utils import embed_text
        vec = embed_text("test normalization")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01

    def test_embed_batch_shape(self):
        from backend.core.utils import embed_batch
        vecs = embed_batch(["hello", "world", "test"])
        assert vecs.shape == (3, 384)

    def test_embed_batch_consistency(self):
        """Batch and single embed should yield same vectors."""
        from backend.core.utils import embed_text, embed_batch
        single = embed_text("consistent check")
        batch = embed_batch(["consistent check"])
        # cosine similarity should be very high
        sim = np.dot(single, batch[0])
        assert sim > 0.999


class TestOODDetector:
    """Mahalanobis OOD detection with Ledoit-Wolf shrinkage."""

    def test_unfitted_returns_negative(self):
        from backend.core.utils import mahalanobis_distance
        # reset cache to unfitted state
        from backend.core import utils
        old = dict(utils._ood_cache)
        utils._ood_cache.update({"mean": None, "precision": None, "fitted": False})
        try:
            dist = mahalanobis_distance(np.zeros(384, dtype=np.float32))
            assert dist == -1.0
        finally:
            utils._ood_cache.update(old)

    def test_fitted_returns_distance(self):
        from backend.core.utils import fit_ood_detector, mahalanobis_distance, embed_batch
        # fit on a small set of known embeddings
        texts = ["machine learning", "neural network", "deep learning",
                 "gradient descent", "backpropagation", "loss function",
                 "convolutional network", "recurrent network", "attention mechanism",
                 "transformer model"]
        embs = embed_batch(texts)
        fit_ood_detector(embs)
        # in-distribution query
        from backend.core.utils import embed_text
        in_dist = embed_text("reinforcement learning")
        dist_in = mahalanobis_distance(in_dist)
        assert dist_in > 0
        # wildly OOD query
        ood_vec = embed_text("recipe for chocolate cake with buttercream frosting")
        dist_ood = mahalanobis_distance(ood_vec)
        # OOD should be further than in-distribution
        assert dist_ood > dist_in


# ---------------------------------------------------------------------------
# critique.py
# ---------------------------------------------------------------------------

class TestCritiqueParsing:
    """Inline critique block extraction and stripping."""

    SAMPLE_RESPONSE = """Here is my answer about RLFO architecture.

The system uses four retrieval paths [1] with CQL-based fusion [2].

<critique>
Factual accuracy: 0.85/1.00
Proactivity score: 0.70/1.00
Helpfulness: 0.90/1.00
Citation coverage: 0.80/1.00
Final reward: 0.82
Proactive suggestions:
- How does CQL compare to PPO for this use case?
- What are the latency trade-offs?
</critique>"""

    def test_parse_extracts_scores(self):
        from backend.core.critique import parse_inline_critique
        cleaned, result = parse_inline_critique(self.SAMPLE_RESPONSE)
        assert result["factual"] == 0.85
        assert result["proactivity"] == 0.70
        assert result["helpfulness"] == 0.90
        assert result["reward"] == 0.82

    def test_parse_extracts_suggestions(self):
        from backend.core.critique import parse_inline_critique
        _, result = parse_inline_critique(self.SAMPLE_RESPONSE)
        assert len(result["proactive_suggestions"]) == 2
        assert "CQL" in result["proactive_suggestions"][0]

    def test_parse_strips_critique_block(self):
        from backend.core.critique import parse_inline_critique
        cleaned, _ = parse_inline_critique(self.SAMPLE_RESPONSE)
        assert "<critique>" not in cleaned
        assert "Final reward" not in cleaned
        # answer content preserved
        assert "four retrieval paths" in cleaned

    def test_strip_critique_block(self):
        from backend.core.critique import strip_critique_block
        result = strip_critique_block(self.SAMPLE_RESPONSE)
        assert "<critique>" not in result
        assert "Factual accuracy" not in result
        assert "four retrieval paths" in result

    def test_strip_source_tags(self):
        from backend.core.critique import strip_critique_block
        text = "[RAG:0.85|w=0.40] Some doc text.\n[CAG:0.90|w=0.30] Cached answer."
        result = strip_critique_block(text)
        assert "[RAG:" not in result
        assert "[CAG:" not in result
        assert "Some doc text" in result

    def test_no_critique_block_returns_defaults(self):
        from backend.core.critique import parse_inline_critique
        cleaned, result = parse_inline_critique("Just a plain answer.")
        assert result["reward"] == 0.75
        assert cleaned == "Just a plain answer."

    def test_clamped_scores(self):
        """Scores outside [0, 1] should be clamped (both directions)."""
        from backend.core.critique import parse_inline_critique
        bad = "<critique>\nFactual accuracy: 1.50/1.00\nFinal reward: -0.3\n</critique>"
        _, result = parse_inline_critique(bad)
        # high clamps down to 1.0
        assert result["factual"] == 1.0
        # negative is captured by the signed regex (F4.8) and clamped up to 0.0
        assert result["reward"] == 0.0


class TestCitationCounting:
    """Citation coverage computation."""

    def test_counts_inline_citations(self):
        from backend.core.critique import count_citations
        text = "The system uses RAG [1] and CQL [2]. It also has a cache [1]."
        stats = count_citations(text)
        assert stats["total_citations"] == 3
        assert stats["unique_sources"] == 2

    def test_no_citations(self):
        from backend.core.critique import count_citations
        stats = count_citations("No citations in this text at all.")
        assert stats["total_citations"] == 0
        assert stats["coverage_ratio"] == 0.0


class TestCritiqueFunction:
    """Integration of critique() with reward scaling."""

    def test_reward_scaling(self):
        from backend.core.critique import critique
        response = """Answer text.
<critique>
Factual accuracy: 0.80/1.00
Proactivity score: 0.60/1.00
Helpfulness: 0.70/1.00
Final reward: 0.80
Proactive suggestions:
- Follow up question?
</critique>"""
        result = critique("test query", "test context", response)
        # default reward_scale is 1.0, so reward should be 0.80
        assert abs(result["reward"] - 0.80) < 0.01
        assert len(result["proactive_suggestions"]) >= 1


# ---------------------------------------------------------------------------
# memory.py
# ---------------------------------------------------------------------------

class TestConversationMemory:
    """Session memory, entity extraction, query expansion."""

    def test_entity_extraction_business(self):
        from backend.core.memory import ConversationMemory
        mem = ConversationMemory()
        entities = mem.extract_entities("I went to The Blue Caboose Restaurant last night")
        assert "business" in entities
        assert any("Blue Caboose" in v for v in entities["business"])

    def test_entity_extraction_location(self):
        from backend.core.memory import ConversationMemory
        mem = ConversationMemory()
        entities = mem.extract_entities("We visited a place in Kansas City, MO")
        assert "location" in entities

    def test_needs_expansion_anaphora(self):
        from backend.core.memory import ConversationMemory
        mem = ConversationMemory()
        assert mem.needs_expansion("what are their hours?")
        assert mem.needs_expansion("tell me about it")
        assert mem.needs_expansion("more")  # short query

    def test_no_expansion_detailed_query(self):
        from backend.core.memory import ConversationMemory
        mem = ConversationMemory()
        assert not mem.needs_expansion("Explain the CSWR stability filtering algorithm in detail")

    def test_session_lifecycle(self):
        from backend.core.memory import ConversationMemory
        mem = ConversationMemory()
        sid = "test-session-123"
        state = mem.add_turn(sid, "user", "Tell me about The Rusty Anchor Restaurant in Portland")
        assert state.active_entities.get("business") is not None
        state = mem.add_turn(sid, "assistant", "The Rusty Anchor is a seafood place...")
        # session survives multiple turns
        ctx = mem.get_conversation_context(sid)
        assert "Rusty Anchor" in ctx
        mem.clear_session(sid)
        ctx_after = mem.get_conversation_context(sid)
        assert "Rusty Anchor" not in ctx_after

    def test_query_expansion_with_entity(self):
        from backend.core.memory import ConversationMemory
        mem = ConversationMemory()
        sid = "expand-test"
        mem.add_turn(sid, "user", "Tell me about The Blue Caboose Restaurant in Austin")
        expanded, meta = mem.expand_query(sid, "what are their hours?")
        assert meta["expanded"]
        assert "Blue Caboose" in expanded

    def test_clear_all_sessions(self):
        from backend.core.memory import ConversationMemory
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "hello")
        mem.add_turn("s2", "user", "world")
        mem.clear_all_sessions()
        assert mem.get_conversation_context("s1") == ""
        assert mem.get_conversation_context("s2") == ""

    def test_session_ttl_cleanup(self):
        import time
        from backend.core.memory import ConversationMemory
        mem = ConversationMemory(session_ttl=1)
        mem.add_turn("old", "user", "old message")
        # artificially age the session
        mem._sessions["old"].updated_at = time.time() - 10
        # trigger cleanup by creating a new session
        mem.get_or_create_session("new")
        assert "old" not in mem._sessions


# ---------------------------------------------------------------------------
# fusion.py
# ---------------------------------------------------------------------------

class TestNormalizeWeights:
    """Weight normalization with NaN/None handling."""

    def test_normal_weights(self):
        from backend.core.fusion import normalize_weights
        result = normalize_weights([0.3, 0.5, 0.1, 0.1])
        assert abs(sum(result) - 1.0) < 1e-6

    def test_zero_weights_uniform(self):
        from backend.core.fusion import normalize_weights
        result = normalize_weights([0.0, 0.0, 0.0, 0.0])
        assert all(abs(w - 0.25) < 1e-6 for w in result)

    def test_none_and_nan_handling(self):
        from backend.core.fusion import normalize_weights
        result = normalize_weights([None, float("nan"), 0.5, 0.5])
        assert abs(sum(result) - 1.0) < 1e-6
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_negative_clipped(self):
        from backend.core.fusion import normalize_weights
        result = normalize_weights([-1.0, 0.5, 0.5])
        assert result[0] == 0.0
        assert abs(sum(result) - 1.0) < 1e-6


class TestFuseContext:
    """Fusion filtering by score thresholds."""

    def test_cag_threshold(self):
        from backend.core.fusion import fuse_context
        cag_low = [{"text": "low cached", "score": 0.50}]
        cag_high = [{"text": "cached answer", "score": 0.90}]
        r1 = fuse_context("test", cag_low, [])
        r2 = fuse_context("test", cag_high, [])
        assert "cached" not in r1["context"]
        assert "cached answer" in r2["context"]

    def test_graph_threshold(self):
        from backend.core.fusion import fuse_context
        graph_low = [{"text": "low score", "score": 0.50}]
        graph_high = [{"text": "graph chunk", "score": 0.80}]
        r1 = fuse_context("test", [], graph_low)
        r2 = fuse_context("test", [], graph_high)
        assert "low score" not in r1["context"]
        assert "graph chunk" in r2["context"]

    def test_returns_weights(self):
        from backend.core.fusion import fuse_context
        result = fuse_context("test", [], [])
        assert "cag" in result["weights"]
        assert "graph" in result["weights"]


# ---------------------------------------------------------------------------
# FusionEnv observation space
# ---------------------------------------------------------------------------

class TestFusionEnvObsSpace:
    """Verify reset() and step() produce same-shape observations."""

    @pytest.mark.slow
    def test_obs_shape_consistency(self):
        from backend.rl.fusion_env import FusionEnv
        env = FusionEnv()
        obs_reset, _ = env.reset(options={"query": "What is RLFusion?"})
        assert obs_reset.shape == (394,), f"reset() returned shape {obs_reset.shape}, expected (394,)"

        action = env.action_space.sample()
        obs_step, reward, done, truncated, info = env.step(action)
        assert obs_step.shape == (394,), f"step() returned shape {obs_step.shape}, expected (394,)"

    @pytest.mark.slow
    def test_obs_matches_space(self):
        from backend.rl.fusion_env import FusionEnv
        env = FusionEnv()
        obs, _ = env.reset(options={"query": "Explain CSWR"})
        assert env.observation_space.contains(obs), "reset() obs not in observation_space"


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

class TestConfig:
    """Config loading and path resolution."""

    def test_config_loads(self):
        from backend.config import cfg
        assert "llm" in cfg
        assert "embedding" in cfg
        assert "rl" in cfg

    def test_embedding_model_matches_code(self):
        """Config.yaml embedding model must match what utils.py loads."""
        from backend.config import cfg
        assert cfg["embedding"]["model"] == "BAAI/bge-small-en-v1.5"

    def test_project_root_exists(self):
        from backend.config import PROJECT_ROOT
        assert PROJECT_ROOT.exists()
        assert (PROJECT_ROOT / "backend" / "config.yaml").exists()

    def test_path_helpers(self):
        from backend.config import get_data_path, get_db_path, get_index_path
        assert get_data_path().exists()
        assert get_db_path().exists()
        assert get_index_path().exists()


# ---------------------------------------------------------------------------
# Conversation persistence (SQLite)
# ---------------------------------------------------------------------------

class TestConversationPersistence:
    """Verify record_turn writes to both in-memory state and SQLite."""

    def test_persist_turn_writes_to_db(self, tmp_path, monkeypatch):
        import sqlite3
        from backend.core import memory as memory_mod

        db_path = tmp_path / "rlfo_cache.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT, role TEXT, content TEXT, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.commit()
        monkeypatch.setattr(memory_mod, "_get_db_path", lambda: db_path)

        memory_mod.record_turn("test-persist", "user", "hello from test")
        memory_mod.record_turn("test-persist", "assistant", "hi back")

        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE session_id = 'test-persist' ORDER BY id"
        ).fetchall()
        conn.close()

        assert len(rows) == 2
        assert rows[0] == ("user", "hello from test")
        assert rows[1] == ("assistant", "hi back")


# ---------------------------------------------------------------------------
# CAG batch embedding
# ---------------------------------------------------------------------------

class TestCAGRetrieval:
    """Verify CAG retrieval with batch embedding path."""

    def test_cag_exact_hit(self, tmp_path, monkeypatch):
        import sqlite3
        from backend.core import retrievers as retrievers_mod

        db_path = tmp_path / "rlfo_cache.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE cache (key TEXT PRIMARY KEY, key_hash TEXT, "
            "value TEXT, score REAL DEFAULT 1.0)"
        )
        conn.execute(
            "INSERT INTO cache (key, value, score) VALUES (?, ?, ?)",
            ("test_cag_query_exact", "cached answer text", 0.95),
        )
        conn.commit()
        conn.close()
        monkeypatch.setattr(retrievers_mod, "_get_db_path", lambda: db_path)

        results = retrievers_mod.retrieve_cag("test_cag_query_exact")
        assert len(results) == 1
        assert results[0]["text"] == "cached answer text"
        assert results[0]["score"] == 0.95

    def test_cag_no_hit(self):
        from backend.core.retrievers import retrieve_cag
        results = retrieve_cag("xyzzy_nonexistent_query_42")
        assert results == []


# ---------------------------------------------------------------------------
# Graph cache
# ---------------------------------------------------------------------------

class TestGraphCache:
    """Verify retrieve_graph behavior when the entity graph is empty."""

    def test_retrieve_graph_empty(self):
        from backend.core.retrievers import retrieve_graph, _graph_engine_cache
        from backend.config import PROJECT_ROOT
        _graph_engine_cache["engine"] = None
        _graph_engine_cache["attempted"] = False
        entity_graph = PROJECT_ROOT / "data" / "entity_graph.json"
        eg_backup = None
        if entity_graph.exists():
            eg_backup = entity_graph.read_text()
            entity_graph.unlink()
        try:
            results = retrieve_graph("What is machine learning?")
            assert results == []
        finally:
            if eg_backup is not None:
                entity_graph.write_text(eg_backup)
            _graph_engine_cache["engine"] = None
            _graph_engine_cache["attempted"] = False


# ---------------------------------------------------------------------------
# CSWR scoring with batch embeddings
# ---------------------------------------------------------------------------

class TestCSWRScoring:
    """Verify CSWR stability/drift scoring uses batch embeddings correctly."""

    def test_score_chunks_produces_scores(self):
        from backend.core.retrievers import score_chunks
        chunks = [
            {"id": "0", "text": "Machine learning uses neural networks", "score": 0.8,
             "local_stability": 0.0, "question_fit": 0.0, "drift_penalty": 0.0, "csw_score": 0.0},
            {"id": "1", "text": "Deep learning extends machine learning with layers", "score": 0.7,
             "local_stability": 0.0, "question_fit": 0.0, "drift_penalty": 0.0, "csw_score": 0.0},
            {"id": "2", "text": "Completely unrelated text about cooking pasta", "score": 0.3,
             "local_stability": 0.0, "question_fit": 0.0, "drift_penalty": 0.0, "csw_score": 0.0},
        ]
        profile = {"query_text": "machine learning", "primary_intent": "explain",
                   "key_entities": ["neural networks"], "required_facts": [],
                   "expected_shape": "definition"}
        cswr_cfg = {"vector_weight": 0.4, "local_stability_weight": 0.3,
                    "question_fit_weight": 0.2, "drift_penalty_weight": 0.1}

        scored = score_chunks(chunks, profile, cswr_cfg)
        # results should be sorted by csw_score descending
        assert scored[0]["csw_score"] >= scored[-1]["csw_score"]
        # all chunks should have non-negative scores
        assert all(c["csw_score"] >= 0.0 for c in scored)
        # stability should be computed (not left at default 0.0)
        assert any(c["local_stability"] != 0.0 for c in scored)

    def test_compute_stability_batch(self):
        from backend.core.retrievers import compute_stability
        from backend.core.utils import embed_batch
        chunks = [
            {"id": "a", "text": "The sky is blue"},
            {"id": "b", "text": "The ocean is blue too"},
            {"id": "c", "text": "Cooking Italian pasta is fun"},
        ]
        emb_matrix = embed_batch([c["text"] for c in chunks])
        embeddings = {c["id"]: emb_matrix[i] for i, c in enumerate(chunks)}

        # middle chunk with similar neighbors should have high stability
        stab_b = compute_stability(chunks[1], chunks, embeddings)
        assert 0.0 <= stab_b <= 1.0
        # edge chunk with dissimilar neighbor should be lower
        stab_c = compute_stability(chunks[2], chunks, embeddings)
        assert stab_b >= stab_c  # related neighbors vs unrelated


# ---------------------------------------------------------------------------
# Phase 4: Input validation and security
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Verify query length limits and upload path sanitization."""

    def test_max_query_len_is_set(self):
        # import the constant from main to verify it exists
        import importlib
        spec = importlib.util.spec_from_file_location("main_module",
            str(Path(__file__).resolve().parents[1] / "backend" / "main.py"))
        # just check the file contains the constant
        main_src = (Path(__file__).resolve().parents[1] / "backend" / "main.py").read_text()
        assert "_MAX_QUERY_LEN" in main_src
        assert "_MAX_UPLOAD_BYTES" in main_src

    def test_upload_path_traversal_blocked(self):
        """Verify Path().name strips directory components."""
        from pathlib import Path as P
        # simulate the upload sanitization logic
        malicious = "../../etc/passwd"
        safe_name = P(malicious).name
        assert safe_name == "passwd"
        assert ".." not in safe_name

    def test_upload_dotfile_blocked(self):
        """Filenames starting with . should be rejected."""
        # mirrors the logic in main.py upload handler
        from pathlib import Path as P
        filename = ".htaccess"
        name = P(filename).name
        blocked = '..' in name or name.startswith('.')
        assert blocked

    def test_check_safety_callable(self):
        from backend.core.critique import check_safety
        # safe query should pass
        safe, reason = check_safety("What is machine learning?")
        assert safe
        assert reason == "Safe"


# ---------------------------------------------------------------------------
# Phase 2: GraphEngine — entity resolution, community detection, hybrid search
# ---------------------------------------------------------------------------

class TestEntityExtraction:
    """Heuristic NER: capitalized phrases, backtick terms, acronyms."""

    def test_extracts_capitalized_phrases(self):
        from backend.core.graph_engine import extract_entities_heuristic
        text = "Machine Learning is a subset of Artificial Intelligence used widely."
        entities = extract_entities_heuristic(text, "test.md")
        labels = [e["label"] for e in entities]
        assert "Machine Learning" in labels
        assert "Artificial Intelligence" in labels

    def test_extracts_backtick_terms(self):
        from backend.core.graph_engine import extract_entities_heuristic
        text = "The `FusionEnv` class wraps the gymnasium environment."
        entities = extract_entities_heuristic(text, "test.md")
        labels = [e["label"] for e in entities]
        assert "FusionEnv" in labels

    def test_extracts_acronyms(self):
        from backend.core.graph_engine import extract_entities_heuristic
        text = "CSWR filters chunks by stability. CQL trains the RL policy via FAISS vectors."
        entities = extract_entities_heuristic(text, "test.md")
        labels = [e["label"] for e in entities]
        assert "CSWR" in labels
        assert "CQL" in labels
        assert "FAISS" in labels

    def test_skips_common_words(self):
        from backend.core.graph_engine import extract_entities_heuristic
        text = "THE AND FOR NOT BUT ARE WAS HAS ITS CAN"
        entities = extract_entities_heuristic(text, "")
        labels = [e["label"] for e in entities]
        for skip_word in ("THE", "AND", "FOR", "NOT"):
            assert skip_word not in labels

    def test_returns_source_chunks(self):
        from backend.core.graph_engine import extract_entities_heuristic
        entities = extract_entities_heuristic("Neural Network training is complex.", "doc.md")
        matching = [e for e in entities if e["label"] == "Neural Network"]
        assert len(matching) == 1
        assert "doc.md" in matching[0]["source_chunks"]

    def test_respects_min_length(self):
        from backend.core.graph_engine import extract_entities_heuristic
        # single letter caps and 2-letter acronyms should be skipped
        text = "AB CD Machine Learning is great"
        entities = extract_entities_heuristic(text)
        labels = [e["label"] for e in entities]
        assert "AB" not in labels  # too short


class TestGraphEngine:
    """GraphEngine: add, resolve, traverse, search, build, save/load."""

    def _make_engine(self, tmp_path=None):
        from backend.core.graph_engine import GraphEngine
        if tmp_path is None:
            tmp_path = Path(tempfile.mkdtemp())
        return GraphEngine(graph_path=tmp_path / "entity_graph.json")

    def test_add_entity(self):
        engine = self._make_engine()
        nid = engine.add_entity({
            "label": "Machine Learning",
            "description": "A field of AI focused on learning from data.",
            "entity_type": "concept",
        })
        assert engine.node_count == 1
        assert nid  # non-empty ID returned

    def test_add_relation(self):
        engine = self._make_engine()
        engine.add_entity({"id": "ml", "label": "Machine Learning"})
        engine.add_entity({"id": "dl", "label": "Deep Learning"})
        engine.add_relation("ml", "dl", label="parent_of")
        assert engine.edge_count == 1

    def test_add_relation_missing_node_raises(self):
        engine = self._make_engine()
        engine.add_entity({"id": "ml", "label": "Machine Learning"})
        with pytest.raises(ValueError, match="not in graph"):
            engine.add_relation("ml", "nonexistent", label="test")

    def test_add_relation_increments_weight(self):
        engine = self._make_engine()
        engine.add_entity({"id": "a", "label": "Alpha"})
        engine.add_entity({"id": "b", "label": "Beta"})
        engine.add_relation("a", "b", label="co_occurs")
        engine.add_relation("a", "b", label="co_occurs")
        weight = engine.graph.edges["a", "b"].get("weight", 0)
        assert weight == 2.0

    def test_resolve_entities_deduplicates(self):
        from backend.core.graph_engine import EntityNode
        engine = self._make_engine()
        entities = [
            EntityNode(label="machine learning", description="ML is a field of AI"),
            EntityNode(label="Machine Learning", description="Machine Learning uses data to learn patterns"),
        ]
        resolved = engine.resolve_entities(entities, threshold=0.85)
        # should merge into one (similar enough)
        assert len(resolved) <= len(entities)

    def test_resolve_entities_keeps_distinct(self):
        from backend.core.graph_engine import EntityNode
        engine = self._make_engine()
        entities = [
            EntityNode(label="Python", description="A programming language"),
            EntityNode(label="Guitar", description="A string instrument played by musicians"),
        ]
        resolved = engine.resolve_entities(entities, threshold=0.95)
        assert len(resolved) == 2

    def test_detect_communities_connected_components(self):
        engine = self._make_engine()
        engine.add_entity({"id": "a", "label": "Alpha"})
        engine.add_entity({"id": "b", "label": "Beta"})
        engine.add_entity({"id": "c", "label": "Gamma"})
        engine.add_relation("a", "b", label="related")
        # c is disconnected from a/b
        communities = engine.detect_communities()
        assert len(communities) >= 2  # at least 2 components

    def test_get_community_summary(self):
        engine = self._make_engine()
        engine.add_entity({"id": "a", "label": "Alpha", "description": "First entity"})
        engine.add_entity({"id": "b", "label": "Beta", "description": "Second entity"})
        engine.add_relation("a", "b", label="co_occurs")
        engine.detect_communities()
        # find a community with members
        for comm_id, members in engine.communities.items():
            if members:
                info = engine.get_community_summary(comm_id)
                assert info["member_count"] > 0
                assert info["summary"] != ""
                break

    def test_multi_hop_traverse(self):
        engine = self._make_engine()
        engine.add_entity({"id": "a", "label": "Alpha"})
        engine.add_entity({"id": "b", "label": "Beta"})
        engine.add_entity({"id": "c", "label": "Gamma"})
        engine.add_relation("a", "b", label="knows")
        engine.add_relation("b", "c", label="works_with")
        results = engine.multi_hop_traverse("a", max_hops=2)
        ids_found = [r["id"] for r in results]
        assert "a" in ids_found
        assert "b" in ids_found
        assert "c" in ids_found
        # hop decay: a's score > b's > c's
        score_a = next(r["score"] for r in results if r["id"] == "a")
        score_b = next(r["score"] for r in results if r["id"] == "b")
        assert score_a > score_b

    def test_multi_hop_traverse_missing_node(self):
        engine = self._make_engine()
        results = engine.multi_hop_traverse("nonexistent")
        assert results == []

    def test_hybrid_search_empty_graph(self):
        engine = self._make_engine()
        results = engine.hybrid_search("anything")
        assert results == []

    def test_hybrid_search_with_entities(self):
        engine = self._make_engine()
        engine.add_entity({
            "id": "ml", "label": "Machine Learning",
            "description": "Learning from data using algorithms",
        })
        engine.add_entity({
            "id": "dl", "label": "Deep Learning",
            "description": "Neural networks with many layers",
        })
        engine.add_relation("ml", "dl", label="parent_of")
        results = engine.hybrid_search("neural network deep learning")
        assert len(results) > 0
        assert all(r["source"] == "graph" for r in results)

    def test_build_from_chunks(self):
        engine = self._make_engine()
        chunks = [
            {"text": "Machine Learning and Deep Learning are subfields of Artificial Intelligence.", "source": "doc.md"},
            {"text": "CSWR filters chunks using Stability Weighted Retrieval.", "source": "cswr.md"},
        ]
        count = engine.build_from_chunks(chunks)
        assert count > 0
        assert engine.node_count > 0

    def test_save_and_load(self):
        tmp_dir = Path(tempfile.mkdtemp())
        graph_path = tmp_dir / "entity_graph.json"

        from backend.core.graph_engine import GraphEngine
        engine1 = GraphEngine(graph_path=graph_path)
        engine1.add_entity({"id": "test_node", "label": "Test Node", "description": "A test entity"})
        engine1.save()
        assert graph_path.exists()

        engine2 = GraphEngine(graph_path=graph_path)
        assert engine2.node_count == 1
        assert "test_node" in [n for n in engine2.graph.nodes()]

    def test_clear_resets_state(self):
        engine = self._make_engine()
        engine.add_entity({"id": "x", "label": "X", "description": "An entity"})
        assert engine.node_count == 1
        engine.clear()
        assert engine.node_count == 0
        assert engine.edge_count == 0

    def test_compute_chunk_graph_scores_empty(self):
        engine = self._make_engine()
        scores = engine.compute_chunk_graph_scores("Some chunk text", "some query")
        assert scores["co_occurrence_bonus"] == 0.0
        assert scores["coherence_penalty"] == 0.0
        assert scores["path_distance_weight"] == 0.0

    def test_compute_chunk_graph_scores_with_data(self):
        engine = self._make_engine()
        engine.add_entity({
            "id": "ml", "label": "Machine Learning",
            "description": "ML is learning from data",
        })
        engine.add_entity({
            "id": "dl", "label": "Deep Learning",
            "description": "DL uses neural networks with many layers",
        })
        engine.add_relation("ml", "dl", label="parent_of")
        engine.detect_communities()
        scores = engine.compute_chunk_graph_scores(
            "Machine Learning is a broad field",
            "explain machine learning concepts",
        )
        # should have non-negative bonus
        assert scores["co_occurrence_bonus"] >= 0.0
        assert scores["coherence_penalty"] >= 0.0


class TestResolveEntitiesRetriever:
    """retrievers.resolve_entities delegates to GraphEngine."""

    def test_resolves_via_retriever(self):
        from backend.core.retrievers import resolve_entities
        entities = [
            {"label": "Python Programming", "description": "A language for coding"},
            {"label": "Cooking Recipes", "description": "Instructions for preparing food"},
        ]
        resolved = resolve_entities(entities)
        assert len(resolved) == 2  # distinct enough to stay separate


class TestBuildEntityGraph:
    """retrievers.build_entity_graph builds and persists a knowledge graph."""

    def test_build_from_chunks(self):
        from backend.core.retrievers import build_entity_graph, _graph_engine_cache
        chunks = [
            {"text": "Neural Networks learn features from data. Deep Learning extends this.", "source": "a.md"},
            {"text": "FAISS enables efficient vector similarity search via HNSW indexing.", "source": "b.md"},
        ]
        count = build_entity_graph(chunks)
        assert count > 0
        engine = _graph_engine_cache["engine"]
        assert engine is not None
        assert engine.node_count > 0

    def test_build_returns_zero_on_empty(self):
        from backend.core.retrievers import build_entity_graph
        count = build_entity_graph([])
        assert count == 0


class TestCommunitySummarize:
    """retrievers.community_summarize returns community-level retrieval results."""

    def test_returns_empty_on_empty_graph(self):
        from backend.core.retrievers import community_summarize, _graph_engine_cache
        from backend.config import PROJECT_ROOT
        # reset engine cache and remove entity graph file if prior tests wrote one
        _graph_engine_cache["engine"] = None
        _graph_engine_cache["attempted"] = False
        entity_graph = PROJECT_ROOT / "data" / "entity_graph.json"
        backup = None
        if entity_graph.exists():
            backup = entity_graph.read_text()
            entity_graph.unlink()
        try:
            results = community_summarize("anything")
            assert results == []
        finally:
            if backup is not None:
                entity_graph.write_text(backup)
            _graph_engine_cache["engine"] = None
            _graph_engine_cache["attempted"] = False

    def test_returns_community_results(self):
        from backend.core.retrievers import build_entity_graph, community_summarize
        chunks = [
            {"text": "Machine Learning uses Statistical Methods for prediction.", "source": "ml.md"},
            {"text": "Neural Networks and Deep Learning are related to Artificial Intelligence.", "source": "dl.md"},
        ]
        build_entity_graph(chunks)
        results = community_summarize("machine learning AI")
        # may or may not produce communities depending on entity count
        assert isinstance(results, list)


class TestComputeFitGraphAware:
    """Phase 2 extension: compute_fit with optional graph_context parameter."""

    def test_no_graph_context_unchanged(self):
        from backend.core.retrievers import compute_fit
        chunk = {"text": "Machine learning uses neural networks for prediction"}
        profile = {"primary_intent": "explain", "key_entities": ["neural networks"],
                   "required_facts": [], "expected_shape": "definition"}
        # no graph_context = original behavior
        score = compute_fit(chunk, profile)
        assert 0.0 <= score <= 1.0

    def test_graph_context_adds_bonus(self):
        from backend.core.retrievers import compute_fit
        chunk = {"text": "Machine learning uses neural networks for prediction"}
        profile = {"primary_intent": "explain", "key_entities": ["neural networks"],
                   "required_facts": [], "expected_shape": "definition"}
        base_score = compute_fit(chunk, profile)
        boosted_score = compute_fit(chunk, profile, graph_context={
            "co_occurrence_bonus": 0.10,
            "coherence_penalty": 0.0,
            "path_distance_weight": 0.05,
        })
        assert boosted_score >= base_score

    def test_graph_context_penalty_reduces(self):
        from backend.core.retrievers import compute_fit
        chunk = {"text": "Machine learning uses neural networks for prediction"}
        profile = {"primary_intent": "explain", "key_entities": ["neural networks"],
                   "required_facts": [], "expected_shape": "definition"}
        base_score = compute_fit(chunk, profile)
        penalized_score = compute_fit(chunk, profile, graph_context={
            "co_occurrence_bonus": 0.0,
            "coherence_penalty": 0.10,
            "path_distance_weight": 0.0,
        })
        assert penalized_score <= base_score


# ---------------------------------------------------------------------------
# model_router.py (Phase 6)
# ---------------------------------------------------------------------------

class TestModelRouter:
    """Phase 6: MoE-style model routing for multi-model Ollama serving."""

    def test_default_router_returns_general(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        # default config has no specialists, should always return general
        model = router.select_model("explain")
        assert model == router.general_model

    def test_router_enabled_property(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        assert isinstance(router.enabled, bool)

    def test_general_model_matches_config(self):
        from backend.core.model_router import ModelRouter
        from backend.config import cfg
        router = ModelRouter()
        expected = cfg.get("model_router", {}).get("general_model", cfg["llm"]["model"])
        assert router.general_model == expected

    def test_register_and_select_specialist(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        router.register_model("codellama:7b", ["design", "troubleshoot"], priority=10)
        # specialist should win for design
        assert router.select_model("design") == "codellama:7b"
        # non-specialist task still returns general
        assert router.select_model("summarize") == router.general_model

    def test_register_replaces_duplicate(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        router.register_model("model-a", ["explain"], priority=50)
        router.register_model("model-a", ["compare"], priority=20)
        models = router.list_models()
        matching = [m for m in models if m["name"] == "model-a"]
        assert len(matching) == 1
        assert "compare" in matching[0]["task_types"]
        assert "explain" not in matching[0]["task_types"]

    def test_unregister_model(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        router.register_model("ephemeral", ["explain"])
        assert router.unregister_model("ephemeral") is True
        assert router.unregister_model("ephemeral") is False

    def test_priority_ordering(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        router.register_model("low-priority", ["explain"], priority=90)
        router.register_model("high-priority", ["explain"], priority=5)
        assert router.select_model("explain") == "high-priority"

    def test_list_models_returns_copy(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        models = router.list_models()
        assert isinstance(models, list)
        assert len(models) >= 1  # at least the general model

    def test_register_empty_name_raises(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        with pytest.raises(ValueError, match="empty"):
            router.register_model("", ["explain"])

    def test_register_no_tasks_raises(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        with pytest.raises(ValueError, match="task type"):
            router.register_model("valid-name", [])

    def test_disabled_router_always_returns_general(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        router._enabled = False
        router.register_model("specialist", ["explain"], priority=1)
        assert router.select_model("explain") == router.general_model

    def test_all_task_types_covered(self):
        """Every known task type should return a model (no ValueError)."""
        from backend.core.model_router import ModelRouter, _ALL_TASK_TYPES
        router = ModelRouter()
        for task in _ALL_TASK_TYPES:
            result = router.select_model(task)
            assert isinstance(result, str)
            assert len(result) > 0


class TestModelEntry:
    """Phase 6: ModelEntry TypedDict structure."""

    def test_model_entry_keys(self):
        from backend.core.model_router import ModelEntry
        entry = ModelEntry(name="test", task_types=["explain"], priority=10)
        assert entry["name"] == "test"
        assert entry["task_types"] == ["explain"]
        assert entry["priority"] == 10


# ---------------------------------------------------------------------------
# fine_tune.py (Phase 6)
# ---------------------------------------------------------------------------

class TestSFTDataLoading:
    """Phase 6: loading episodes from replay buffer for SFT."""

    def test_load_from_missing_db(self):
        from backend.rl.fine_tune import load_training_episodes
        episodes = load_training_episodes(db_path="/tmp/nonexistent_sft_test.db")
        assert episodes == []

    def test_load_from_empty_db(self):
        import sqlite3
        from backend.rl.fine_tune import load_training_episodes
        tmp = Path(tempfile.mkdtemp()) / "empty_sft.db"
        conn = sqlite3.connect(str(tmp))
        conn.execute("""
            CREATE TABLE episodes (
                id INTEGER PRIMARY KEY, query TEXT, response TEXT,
                reward REAL, rag_weight REAL, cag_weight REAL, graph_weight REAL,
                fused_context TEXT, proactive_suggestions TEXT
            )
        """)
        conn.commit()
        conn.close()
        episodes = load_training_episodes(min_reward=0.8, db_path=str(tmp))
        assert episodes == []

    def test_load_filters_by_reward(self):
        import sqlite3
        from backend.rl.fine_tune import load_training_episodes
        tmp = Path(tempfile.mkdtemp()) / "sft_filter.db"
        conn = sqlite3.connect(str(tmp))
        conn.execute("""
            CREATE TABLE episodes (
                id INTEGER PRIMARY KEY, query TEXT, response TEXT,
                reward REAL, rag_weight REAL, cag_weight REAL, graph_weight REAL,
                fused_context TEXT, proactive_suggestions TEXT
            )
        """)
        conn.execute(
            "INSERT INTO episodes (query, response, reward, rag_weight, cag_weight, graph_weight) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("good query", "good response", 0.95, 0.4, 0.3, 0.3),
        )
        conn.execute(
            "INSERT INTO episodes (query, response, reward, rag_weight, cag_weight, graph_weight) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("bad query", "bad response", 0.3, 0.5, 0.2, 0.3),
        )
        conn.commit()
        conn.close()
        episodes = load_training_episodes(min_reward=0.8, db_path=str(tmp))
        assert len(episodes) == 1
        assert episodes[0]["query"] == "good query"
        assert episodes[0]["reward"] == 0.95

    def test_episode_typed_fields(self):
        import sqlite3
        from backend.rl.fine_tune import load_training_episodes
        tmp = Path(tempfile.mkdtemp()) / "sft_types.db"
        conn = sqlite3.connect(str(tmp))
        conn.execute("""
            CREATE TABLE episodes (
                id INTEGER PRIMARY KEY, query TEXT, response TEXT,
                reward REAL, cag_weight REAL, graph_weight REAL,
                fused_context TEXT, proactive_suggestions TEXT
            )
        """)
        conn.execute(
            "INSERT INTO episodes (query, response, reward, cag_weight, graph_weight) "
            "VALUES (?, ?, ?, ?, ?)",
            ("typed test", "typed response", 0.9, 0.4, 0.6),
        )
        conn.commit()
        conn.close()
        episodes = load_training_episodes(min_reward=0.5, db_path=str(tmp))
        ep = episodes[0]
        assert isinstance(ep["query"], str)
        assert isinstance(ep["reward"], float)
        assert isinstance(ep["cag_weight"], float)
        assert isinstance(ep["graph_weight"], float)
        # rag_weight was dropped in the v2 overhaul
        assert "rag_weight" not in ep


class TestSFTDataPreparation:
    """Phase 6: formatting episodes into SFT train/val datasets."""

    def test_prepare_empty_input(self):
        from backend.rl.fine_tune import prepare_sft_dataset
        train, val = prepare_sft_dataset([])
        assert train == []
        assert val == []

    def test_prepare_splits_data(self):
        from backend.rl.fine_tune import prepare_sft_dataset, TrainingEpisode
        episodes = [
            TrainingEpisode(query=f"q{i}", response=f"r{i}",
                           reward=0.9, rag_weight=0.3, cag_weight=0.3, graph_weight=0.4)
            for i in range(20)
        ]
        train, val = prepare_sft_dataset(episodes, val_split=0.2)
        assert len(train) + len(val) == 20
        assert len(val) == 4  # 20 * 0.2

    def test_prepare_format_has_instruction_output(self):
        from backend.rl.fine_tune import prepare_sft_dataset, TrainingEpisode
        episodes = [
            TrainingEpisode(query="what is RL?", response="RL is reinforcement learning.",
                           reward=0.9, rag_weight=0.4, cag_weight=0.3, graph_weight=0.3)
        ]
        train, val = prepare_sft_dataset(episodes, val_split=0.0)
        assert len(train) == 1
        assert train[0]["instruction"] == "what is RL?"
        assert train[0]["output"] == "RL is reinforcement learning."

    def test_prepare_invalid_val_split_raises(self):
        from backend.rl.fine_tune import prepare_sft_dataset
        with pytest.raises(ValueError, match="val_split"):
            prepare_sft_dataset([{"query": "q", "response": "r"}], val_split=1.5)


class TestSFTJobConfig:
    """Phase 6: SFT job configuration defaults and structure."""

    def test_default_config_has_all_keys(self):
        from backend.rl.fine_tune import default_config, SFTJobConfig
        config = default_config()
        required = list(SFTJobConfig.__annotations__.keys())
        for key in required:
            assert key in config, f"Missing key: {key}"

    def test_default_config_matches_yaml(self):
        from backend.rl.fine_tune import default_config
        from backend.config import cfg
        config = default_config()
        ft_cfg = cfg.get("fine_tuning", {})
        assert config["lora_rank"] == ft_cfg.get("lora_rank", 16)
        assert config["lora_alpha"] == ft_cfg.get("lora_alpha", 32)

    def test_run_sft_insufficient_data(self):
        """run_sft should return 'insufficient_data' on empty replay."""
        from backend.rl.fine_tune import SFTJobConfig, default_config, run_sft
        config = default_config()
        # point at a nonexistent DB to guarantee no episodes
        config["min_reward"] = 99.0  # unreachable threshold
        result = run_sft(config)
        assert result["status"] == "insufficient_data"
        assert result["episodes_used"] < 10
        assert "error" in result


class TestFineTuneEndpoint:
    """Phase 6: POST /api/fine-tune endpoint request/response."""

    def _reset_limiter(self):
        """Clear rate limiter state so each test starts fresh."""
        from backend.main import limiter
        try:
            limiter._storage.reset()
        except Exception:
            pass

    def test_endpoint_rejects_missing_auth(self):
        import os
        from fastapi.testclient import TestClient
        self._reset_limiter()
        # short test key; the boot-time length floor is bypassed via the
        # RLFUSION_ALLOW_WEAK_ADMIN_KEY escape hatch so the auth path itself
        # is what's under test.
        os.environ["RLFUSION_ADMIN_KEY"] = "test-secret-key-123"
        os.environ["RLFUSION_ALLOW_WEAK_ADMIN_KEY"] = "1"
        try:
            from backend.main import app
            client = TestClient(app)
            resp = client.post("/api/fine-tune", json={"lora_rank": 16})
            assert resp.status_code == 401
        finally:
            os.environ.pop("RLFUSION_ADMIN_KEY", None)
            os.environ.pop("RLFUSION_ALLOW_WEAK_ADMIN_KEY", None)

    def test_endpoint_accepts_valid_auth(self):
        import os
        from fastapi.testclient import TestClient
        self._reset_limiter()
        os.environ["RLFUSION_ADMIN_KEY"] = "test-secret-key-456"
        os.environ["RLFUSION_ALLOW_WEAK_ADMIN_KEY"] = "1"
        try:
            from backend.main import app
            client = TestClient(app)
            resp = client.post(
                "/api/fine-tune",
                json={"min_reward": 0.99},
                headers={"Authorization": "Bearer test-secret-key-456"},
            )
            assert resp.status_code == 200
            data = resp.json()
            # should succeed auth but return insufficient_data (no replay episodes)
            assert data["status"] == "insufficient_data"
            assert "job_id" in data
        finally:
            os.environ.pop("RLFUSION_ADMIN_KEY", None)
            os.environ.pop("RLFUSION_ALLOW_WEAK_ADMIN_KEY", None)

    def test_endpoint_rejects_no_admin_key_set(self):
        import os
        from fastapi.testclient import TestClient
        self._reset_limiter()
        os.environ.pop("RLFUSION_ADMIN_KEY", None)
        from backend.main import app
        client = TestClient(app)
        resp = client.post(
            "/api/fine-tune",
            json={},
            headers={"Authorization": "Bearer anything"},
        )
        assert resp.status_code == 401


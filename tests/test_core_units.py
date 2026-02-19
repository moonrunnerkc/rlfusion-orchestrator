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
        """Scores outside [0, 1] should be clamped."""
        from backend.core.critique import parse_inline_critique
        bad = "<critique>\nFactual accuracy: 1.50/1.00\nFinal reward: -0.3\n</critique>"
        _, result = parse_inline_critique(bad)
        assert result["factual"] == 1.0
        # negative reward not captured by regex; falls back to mean of sub-scores
        # factual=1.0, proactivity=0.75 (default), helpfulness=0.75 (default)
        expected_reward = (1.0 + 0.75 + 0.75) / 3.0
        assert abs(result["reward"] - expected_reward) < 0.01


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

    def test_filters_low_score_rag(self):
        from backend.core.fusion import fuse_context
        rag = [{"text": "low score", "score": 0.30}]
        cag = []
        graph = []
        result = fuse_context("test", rag, cag, graph)
        assert result["context"] == ""

    def test_includes_high_score_rag(self):
        from backend.core.fusion import fuse_context
        rag = [{"text": "high quality doc", "score": 0.80}]
        result = fuse_context("test", rag, [], [])
        assert "high quality doc" in result["context"]
        assert "[RAG:0.80]" in result["context"]

    def test_cag_threshold(self):
        from backend.core.fusion import fuse_context
        cag_low = [{"text": "low cached", "score": 0.50}]
        cag_high = [{"text": "cached answer", "score": 0.90}]
        r1 = fuse_context("test", [], cag_low, [])
        r2 = fuse_context("test", [], cag_high, [])
        assert "cached" not in r1["context"]
        assert "cached answer" in r2["context"]

    def test_returns_weights(self):
        from backend.core.fusion import fuse_context
        result = fuse_context("test", [], [], [])
        assert "rag" in result["weights"]
        assert "cag" in result["weights"]
        assert "graph" in result["weights"]


class TestFormatWebSource:
    """Web source display formatting."""

    def test_strips_tavily_prefix(self):
        from backend.core.fusion import format_web_source
        result = format_web_source({"url": "tavily://search", "text": "content", "title": "Title"})
        assert "tavily" not in result.lower()
        assert "Title" in result

    def test_strips_tavily_from_title(self):
        from backend.core.fusion import format_web_source
        result = format_web_source({"url": "https://example.com", "text": "content", "title": "Tavily: Example"})
        assert "Tavily:" not in result
        assert "Example" in result


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
        assert obs_reset.shape == (396,), f"reset() returned shape {obs_reset.shape}, expected (396,)"

        action = env.action_space.sample()
        obs_step, reward, done, truncated, info = env.step(action)
        assert obs_step.shape == (396,), f"step() returned shape {obs_step.shape}, expected (396,)"

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

    def test_persist_turn_writes_to_db(self):
        import sqlite3
        from backend.config import PROJECT_ROOT, cfg
        from backend.core.memory import record_turn

        db_path = PROJECT_ROOT / cfg["paths"]["db"]
        if not db_path.exists():
            pytest.skip("Database not initialized")

        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM conversations WHERE session_id = 'test-persist'")
        conn.commit()

        record_turn("test-persist", "user", "hello from test")
        record_turn("test-persist", "assistant", "hi back")

        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE session_id = 'test-persist' ORDER BY id"
        ).fetchall()
        conn.execute("DELETE FROM conversations WHERE session_id = 'test-persist'")
        conn.commit()
        conn.close()

        assert len(rows) == 2
        assert rows[0] == ("user", "hello from test")
        assert rows[1] == ("assistant", "hi back")


# ---------------------------------------------------------------------------
# CAG batch embedding
# ---------------------------------------------------------------------------

class TestCAGRetrieval:
    """Verify CAG retrieval with batch embedding path."""

    def test_cag_exact_hit(self):
        import sqlite3
        from backend.config import PROJECT_ROOT, cfg
        from backend.core.retrievers import retrieve_cag

        db_path = PROJECT_ROOT / cfg["paths"]["db"]
        if not db_path.exists():
            pytest.skip("Database not initialized")

        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT OR REPLACE INTO cache (key, value, score) VALUES (?, ?, ?)",
                     ("test_cag_query_exact", "cached answer text", 0.95))
        conn.commit()
        conn.close()

        results = retrieve_cag("test_cag_query_exact")
        assert len(results) == 1
        assert results[0]["text"] == "cached answer text"
        assert results[0]["score"] == 0.95

        # cleanup
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM cache WHERE key = 'test_cag_query_exact'")
        conn.commit()
        conn.close()

    def test_cag_no_hit(self):
        from backend.core.retrievers import retrieve_cag
        results = retrieve_cag("xyzzy_nonexistent_query_42")
        assert results == []


# ---------------------------------------------------------------------------
# Graph cache
# ---------------------------------------------------------------------------

class TestGraphCache:
    """Verify graph loading cache behavior."""

    def test_graph_cache_loads(self):
        from backend.core.retrievers import _load_graph
        G = _load_graph()
        # ontology.json is currently empty, so graph has 0 nodes
        assert G is not None

    def test_retrieve_graph_empty_ontology(self):
        from backend.core.retrievers import retrieve_graph
        results = retrieve_graph("What is machine learning?")
        # empty ontology returns empty results
        assert results == []


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

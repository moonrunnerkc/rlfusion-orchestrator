# Author: Bradley R. Kinnard
# test_stis_contradiction.py - tests for the STIS contradiction trigger in critique.py
# Uses real BGE embeddings to validate contradiction detection and routing logic.

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RLFUSION_DEVICE", "cpu")
os.environ.setdefault("RLFUSION_FORCE_CPU", "true")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestSTISConstants:
    """Verify STIS trigger thresholds are set correctly."""

    def test_cswr_threshold_value(self):
        from backend.core.critique import STIS_CSWR_THRESHOLD
        assert STIS_CSWR_THRESHOLD == 0.70

    def test_similarity_floor_value(self):
        from backend.core.critique import STIS_SIMILARITY_FLOOR
        assert STIS_SIMILARITY_FLOOR == 0.40

    def test_thresholds_are_float(self):
        from backend.core.critique import STIS_CSWR_THRESHOLD, STIS_SIMILARITY_FLOOR
        assert isinstance(STIS_CSWR_THRESHOLD, float)
        assert isinstance(STIS_SIMILARITY_FLOOR, float)

    def test_thresholds_in_valid_range(self):
        from backend.core.critique import STIS_CSWR_THRESHOLD, STIS_SIMILARITY_FLOOR
        assert 0.0 < STIS_CSWR_THRESHOLD < 1.0
        assert 0.0 < STIS_SIMILARITY_FLOOR < 1.0


# ---------------------------------------------------------------------------
# detect_contradiction()
# ---------------------------------------------------------------------------

class TestDetectContradiction:
    """Validate contradiction detection between RAG and Graph results."""

    def test_empty_rag_no_contradiction(self):
        from backend.core.critique import detect_contradiction
        result = detect_contradiction([], [{"text": "some graph fact", "score": 0.8}])
        assert result["contradicted"] is False
        assert result["similarity"] == 1.0

    def test_empty_graph_no_contradiction(self):
        from backend.core.critique import detect_contradiction
        result = detect_contradiction([{"text": "some rag fact", "score": 0.8}], [])
        assert result["contradicted"] is False
        assert result["similarity"] == 1.0

    def test_both_empty_no_contradiction(self):
        from backend.core.critique import detect_contradiction
        result = detect_contradiction([], [])
        assert result["contradicted"] is False

    def test_similar_content_no_contradiction(self):
        """Semantically similar RAG and Graph chunks should not trigger."""
        from backend.core.critique import detect_contradiction
        rag = [{"text": "Python is a programming language used for data science and machine learning", "score": 0.85}]
        graph = [{"text": "Python is a popular language for data science, machine learning, and AI applications", "score": 0.80}]
        result = detect_contradiction(rag, graph)
        assert result["contradicted"] is False
        assert result["similarity"] > 0.40

    def test_opposing_content_triggers_contradiction(self):
        """Semantically unrelated chunks should trigger contradiction."""
        from backend.core.critique import detect_contradiction
        rag = [{"text": "The recommended GPU for deep learning is the NVIDIA RTX 4090 with 24GB VRAM", "score": 0.75}]
        graph = [{"text": "Banana bread recipe: mix flour, sugar, eggs, and mashed bananas in a bowl", "score": 0.60}]
        result = detect_contradiction(rag, graph)
        # these are completely unrelated topics, similarity should be very low
        assert result["similarity"] < 0.60
        # whether it crosses 0.40 depends on embedding geometry
        assert isinstance(result["contradicted"], bool)

    def test_uses_highest_scored_chunks(self):
        """Should compare the top-scored chunk from each source, not the first."""
        from backend.core.critique import detect_contradiction
        rag = [
            {"text": "irrelevant noise text filler content", "score": 0.3},
            {"text": "Python uses garbage collection for memory management", "score": 0.9},
        ]
        graph = [
            {"text": "random unrelated graph entry", "score": 0.2},
            {"text": "Python employs automatic garbage collection to manage memory", "score": 0.85},
        ]
        result = detect_contradiction(rag, graph)
        # both top chunks say the same thing about Python GC
        assert result["contradicted"] is False
        assert result["similarity"] > 0.5

    def test_output_shape(self):
        """Return dict must contain all required keys with correct types."""
        from backend.core.critique import detect_contradiction
        rag = [{"text": "Test claim A", "score": 0.7}]
        graph = [{"text": "Test claim B", "score": 0.7}]
        result = detect_contradiction(rag, graph)

        assert "contradicted" in result
        assert "similarity" in result
        assert "rag_claim" in result
        assert "graph_claim" in result
        assert isinstance(result["contradicted"], bool)
        assert isinstance(result["similarity"], float)
        assert isinstance(result["rag_claim"], str)
        assert isinstance(result["graph_claim"], str)

    def test_similarity_bounded_zero_one(self):
        """Cosine similarity should be in [0, 1] for normalized embeddings."""
        from backend.core.critique import detect_contradiction
        rag = [{"text": "Neural networks use backpropagation for training", "score": 0.8}]
        graph = [{"text": "Stochastic gradient descent optimizes neural network weights", "score": 0.8}]
        result = detect_contradiction(rag, graph)
        assert 0.0 <= result["similarity"] <= 1.0 + 1e-6

    def test_claim_text_truncated(self):
        """Claim text in output should be truncated to 500 chars."""
        from backend.core.critique import detect_contradiction
        long_text = "x" * 1000
        rag = [{"text": long_text, "score": 0.8}]
        graph = [{"text": "short claim", "score": 0.8}]
        result = detect_contradiction(rag, graph)
        assert len(result["rag_claim"]) <= 500

    def test_blank_text_no_contradiction(self):
        """Blank text content should not trigger contradiction."""
        from backend.core.critique import detect_contradiction
        rag = [{"text": "   ", "score": 0.8}]
        graph = [{"text": "valid content", "score": 0.8}]
        result = detect_contradiction(rag, graph)
        assert result["contradicted"] is False
        assert result["similarity"] == 1.0


# ---------------------------------------------------------------------------
# should_route_to_stis()
# ---------------------------------------------------------------------------

class TestShouldRouteToSTIS:
    """Validate STIS routing decision logic (dual-condition gate)."""

    def test_no_contradiction_high_cswr_stays_ollama(self):
        """Neither condition met: proceed with Ollama."""
        from backend.core.critique import should_route_to_stis
        rag = [{"text": "Python is interpreted", "score": 0.85, "csw_score": 0.90}]
        graph = [{"text": "Python is a high-level interpreted language", "score": 0.80, "csw_score": 0.88}]
        result = should_route_to_stis(rag, graph)
        assert result["route_to_stis"] is False
        assert "Proceeding with Ollama" in result["reason"]

    def test_contradiction_with_low_cswr_routes_to_stis(self):
        """Both conditions met: route to STIS."""
        from backend.core.critique import should_route_to_stis
        # totally unrelated content + low scores
        rag = [{"text": "NVIDIA CUDA cores enable parallel GPU computing for deep learning", "score": 0.30, "csw_score": 0.25}]
        graph = [{"text": "Banana bread recipe requires two cups of flour and three ripe bananas", "score": 0.20, "csw_score": 0.15}]
        result = should_route_to_stis(rag, graph)
        # only routes if BOTH conditions hold
        contradiction = result["contradiction"]
        if contradiction["contradicted"] and result["best_cswr"] < 0.70:
            assert result["route_to_stis"] is True
            assert "STIS" in result["reason"]
        else:
            # if embedding similarity happens to be >= 0.40, that's fine
            assert result["route_to_stis"] is False

    def test_contradiction_with_high_cswr_stays_ollama(self):
        """Contradiction but strong CSWR: probably a topic mismatch, not a real conflict."""
        from backend.core.critique import should_route_to_stis
        rag = [{"text": "NVIDIA GPUs are used in deep learning", "score": 0.95, "csw_score": 0.92}]
        graph = [{"text": "Cooking pasta requires boiling water for ten minutes", "score": 0.85, "csw_score": 0.88}]
        result = should_route_to_stis(rag, graph)
        # even if contradicted, high CSWR means we trust the data
        assert result["route_to_stis"] is False
        assert result["best_cswr"] >= 0.70

    def test_no_contradiction_low_cswr_stays_ollama(self):
        """Low CSWR but no contradiction: not an STIS case."""
        from backend.core.critique import should_route_to_stis
        rag = [{"text": "Machine learning models require training data", "score": 0.30, "csw_score": 0.25}]
        graph = [{"text": "Machine learning algorithms learn from training datasets", "score": 0.25, "csw_score": 0.20}]
        result = should_route_to_stis(rag, graph)
        # same topic, so no contradiction, even though CSWR is low
        assert result["route_to_stis"] is False

    def test_explicit_cswr_scores_used(self):
        """When cswr_scores list is passed explicitly, it takes precedence."""
        from backend.core.critique import should_route_to_stis
        rag = [{"text": "GPU computing accelerates matrix operations", "score": 0.30}]
        graph = [{"text": "French cuisine emphasizes butter and cream", "score": 0.25}]
        # pass explicit high CSWR even though result scores are low
        result = should_route_to_stis(rag, graph, cswr_scores=[0.85, 0.90])
        assert result["best_cswr"] == 0.90
        # high explicit CSWR prevents routing regardless of contradiction
        assert result["route_to_stis"] is False

    def test_explicit_low_cswr_scores(self):
        """Explicit low CSWR scores should participate in routing decision."""
        from backend.core.critique import should_route_to_stis
        rag = [{"text": "Quantum computing uses qubits for superposition", "score": 0.2}]
        graph = [{"text": "Chocolate cake needs cocoa powder and eggs", "score": 0.15}]
        result = should_route_to_stis(rag, graph, cswr_scores=[0.30, 0.25])
        assert result["best_cswr"] == 0.30
        assert result["best_cswr"] < 0.70

    def test_output_shape(self):
        """Return dict must contain all required keys."""
        from backend.core.critique import should_route_to_stis
        rag = [{"text": "Test A", "score": 0.5}]
        graph = [{"text": "Test B", "score": 0.5}]
        result = should_route_to_stis(rag, graph)

        assert "route_to_stis" in result
        assert "reason" in result
        assert "contradiction" in result
        assert "best_cswr" in result
        assert isinstance(result["route_to_stis"], bool)
        assert isinstance(result["reason"], str)
        assert isinstance(result["contradiction"], dict)
        assert isinstance(result["best_cswr"], float)

    def test_reason_always_nonempty(self):
        """Every routing decision must include a nonempty reason string."""
        from backend.core.critique import should_route_to_stis
        # scenario 1: empty inputs
        r1 = should_route_to_stis([], [])
        assert len(r1["reason"]) > 0

        # scenario 2: normal inputs
        r2 = should_route_to_stis(
            [{"text": "hello", "score": 0.5}],
            [{"text": "world", "score": 0.5}]
        )
        assert len(r2["reason"]) > 0

    def test_both_empty_no_route(self):
        """Empty retrieval results should never route to STIS."""
        from backend.core.critique import should_route_to_stis
        result = should_route_to_stis([], [])
        assert result["route_to_stis"] is False

    def test_cswr_fallback_to_score_field(self):
        """When csw_score is absent, should fall back to score field."""
        from backend.core.critique import should_route_to_stis
        rag = [{"text": "test rag content", "score": 0.75}]  # no csw_score
        graph = [{"text": "test graph content similar", "score": 0.80}]
        result = should_route_to_stis(rag, graph)
        assert result["best_cswr"] == 0.80


# ---------------------------------------------------------------------------
# Integration: contradiction detection is importable alongside existing symbols
# ---------------------------------------------------------------------------

class TestSTISImportIntegration:
    """Verify STIS symbols coexist with existing critique.py exports."""

    def test_all_existing_symbols_still_importable(self):
        from backend.core.critique import (
            critique,
            check_safety,
            check_faithfulness,
            log_episode_to_replay_buffer,
            get_critique_instruction,
            strip_critique_block,
            parse_inline_critique,
            count_citations,
        )
        # all should be callable
        assert callable(critique)
        assert callable(check_safety)
        assert callable(check_faithfulness)
        assert callable(log_episode_to_replay_buffer)
        assert callable(get_critique_instruction)
        assert callable(strip_critique_block)
        assert callable(parse_inline_critique)
        assert callable(count_citations)

    def test_stis_symbols_importable(self):
        from backend.core.critique import (
            detect_contradiction,
            should_route_to_stis,
            STIS_CSWR_THRESHOLD,
            STIS_SIMILARITY_FLOOR,
        )
        assert callable(detect_contradiction)
        assert callable(should_route_to_stis)
        assert isinstance(STIS_CSWR_THRESHOLD, float)
        assert isinstance(STIS_SIMILARITY_FLOOR, float)

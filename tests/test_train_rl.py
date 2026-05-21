# Author: Bradley R. Kinnard
# Smoke tests for the 2-path CQL trainer plus the obs/simplex helpers.

import os
import sys
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RLFUSION_DEVICE", "cpu")
os.environ.setdefault("RLFUSION_FORCE_CPU", "true")
os.environ.setdefault("RLFUSION_ENV_DRY_RUN", "1")


class TestObsBuilder:
    """obs_builder owns the 394-d observation contract."""

    def test_obs_shape(self):
        from backend.rl.obs_builder import OBS_DIM, build_observation
        embed = np.random.RandomState(0).rand(384).astype(np.float32)
        obs = build_observation("what is RLFusion?", embed, {"cag": [], "graph": []})
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_obs_rejects_wrong_embedding(self):
        from backend.rl.obs_builder import build_observation
        with pytest.raises(ValueError):
            build_observation("x", np.zeros(10, dtype=np.float32), {})

    def test_simplex_floor_holds(self):
        from backend.rl.obs_builder import SIMPLEX_FLOOR, project_to_simplex
        for action in ([10.0, -10.0], [-10.0, 10.0], [0.0, 0.0]):
            w = project_to_simplex(action)
            assert w.shape == (2,)
            assert w[0] >= SIMPLEX_FLOOR - 1e-6
            assert w[1] >= SIMPLEX_FLOOR - 1e-6
            assert abs(w.sum() - 1.0) < 1e-5

    def test_simplex_matches_softmax_when_unconstrained(self):
        """With no extreme inputs the projection differs from raw softmax
        only by the floor scaling. Verify the rescaling stays correct."""
        from backend.rl.obs_builder import SIMPLEX_FLOOR, project_to_simplex
        w = project_to_simplex([0.5, -0.5])
        # 1 - 2*floor + 2*floor == 1.0
        assert abs(w.sum() - 1.0) < 1e-6
        # cag should outweigh graph
        assert w[0] > w[1]
        # numerical sanity
        assert w[0] < 1.0 - SIMPLEX_FLOOR + 1e-6


class TestEpisodeLoading:
    """train_rl._load_episodes_two_path materializes rows into MDPDataset."""

    def test_no_db(self, tmp_path):
        from backend.rl.train_rl import _load_episodes_two_path
        result = _load_episodes_two_path(tmp_path / "missing.db")
        assert result is None

    def test_loads_rows(self, tmp_path):
        from backend.rl.train_rl import _load_episodes_two_path

        db = tmp_path / "ep.db"
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT, response TEXT, reward REAL,
                cag_weight REAL, graph_weight REAL,
                fused_context TEXT, proactive_suggestions TEXT,
                from_cache INTEGER DEFAULT 0,
                schema_version INTEGER DEFAULT 2
            )
        """)
        for q, c, g, r in [
            ("what is rlfusion?", 0.6, 0.4, 0.82),
            ("how does cswr work?", 0.4, 0.6, 0.76),
            ("define cql", 0.7, 0.3, 0.91),
        ]:
            conn.execute(
                "INSERT INTO episodes (query, cag_weight, graph_weight, reward) "
                "VALUES (?, ?, ?, ?)",
                (q, c, g, r),
            )
        conn.commit()
        conn.close()

        ds = _load_episodes_two_path(db)
        assert ds is not None
        assert ds.size() == 3

    def test_skips_from_cache_rows(self, tmp_path):
        from backend.rl.train_rl import _load_episodes_two_path

        db = tmp_path / "ep.db"
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT, response TEXT, reward REAL,
                cag_weight REAL, graph_weight REAL,
                fused_context TEXT, proactive_suggestions TEXT,
                from_cache INTEGER DEFAULT 0,
                schema_version INTEGER DEFAULT 2
            )
        """)
        conn.execute(
            "INSERT INTO episodes (query, cag_weight, graph_weight, reward, from_cache) "
            "VALUES (?, ?, ?, ?, ?)",
            ("cached q", 1.0, 0.0, 0.9, 1),
        )
        conn.execute(
            "INSERT INTO episodes (query, cag_weight, graph_weight, reward, from_cache) "
            "VALUES (?, ?, ?, ?, ?)",
            ("policy q", 0.5, 0.5, 0.8, 0),
        )
        conn.commit()
        conn.close()

        ds = _load_episodes_two_path(db)
        assert ds is not None
        assert ds.size() == 1  # only the non-cache row


class TestCritiqueScoreParsing:
    """F4.8: critique regex must accept signed floats."""

    def test_negative_factual_score_parses(self):
        from backend.core.critique import _FACTUAL_RE
        m = _FACTUAL_RE.search("factual accuracy: -0.05 some text")
        assert m is not None
        assert float(m.group(1)) == pytest.approx(-0.05)

    def test_positive_proactivity_score_parses(self):
        from backend.core.critique import _PROACTIVE_RE
        m = _PROACTIVE_RE.search("proactivity score: 0.82")
        assert m is not None
        assert float(m.group(1)) == pytest.approx(0.82)

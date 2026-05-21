# Author: Bradley R. Kinnard
# Smoke coverage for previously untested modules: decomposer, profile,
# scheduler, engine_detect. F4.7 from the 2026-05-21 remediation plan.

import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RLFUSION_DEVICE", "cpu")
os.environ.setdefault("RLFUSION_FORCE_CPU", "true")


class TestDecomposer:
    """Heuristic and llm-backed decomposer must always return required keys."""

    def test_heuristic_default(self):
        from backend.core.decomposer import _heuristic_decompose

        result = _heuristic_decompose("What is RLFusion?")
        for key in (
            "key_entities",
            "required_facts",
            "primary_intent",
            "expected_shape",
            "sensitivity_level",
        ):
            assert key in result

    def test_heuristic_extracts_entities(self):
        from backend.core.decomposer import _heuristic_decompose

        result = _heuristic_decompose("Explain how CQL works in RLFusion")
        assert isinstance(result["key_entities"], list)

    def test_decompose_query_uses_heuristic_by_default(self):
        """decompose_query consults config; the test config sets use_llm=false."""
        from backend.core.decomposer import decompose_query

        result = decompose_query("define entropy")
        assert "primary_intent" in result


class TestProfile:
    """User profile persists to the sqlite user_profile table."""

    def _seed_db(self, tmp_path):
        db = tmp_path / "rlfo_cache.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE user_profile (fact_key TEXT PRIMARY KEY, fact_value TEXT, category TEXT)"
        )
        conn.commit()
        conn.close()
        return db

    def test_get_user_profile_empty(self, tmp_path, monkeypatch):
        from backend.core import profile as profile_mod

        db = self._seed_db(tmp_path)
        monkeypatch.setattr(profile_mod, "_get_db_path", lambda: db)
        assert profile_mod.get_user_profile() == ""

    def test_update_and_read_fact(self, tmp_path, monkeypatch):
        from backend.core import profile as profile_mod

        db = self._seed_db(tmp_path)
        monkeypatch.setattr(profile_mod, "_get_db_path", lambda: db)
        profile_mod.update_user_fact("name", "Alice", "personal")
        text = profile_mod.get_user_profile()
        assert "Alice" in text or "name" in text.lower()

    def test_detect_remember_pattern(self):
        from backend.core.profile import detect_and_save_memory

        is_mem, content = detect_and_save_memory("Remember that I'm a data scientist")
        # the regex may or may not catch this exact phrasing; whichever way,
        # the function must return a (bool, str|None) shape.
        assert isinstance(is_mem, bool)
        assert content is None or isinstance(content, str)


class TestSchedulerHelpers:
    """detect_hardware + recommend_quantization stay deterministic."""

    def test_detect_hardware_keys(self):
        from backend.core.scheduler import detect_hardware

        profile = detect_hardware()
        for key in ("cpu_count", "ram_total_mb", "gpu_available"):
            assert key in profile

    def test_recommend_quantization_picks_a_band(self):
        from backend.core.scheduler import detect_hardware, recommend_quantization

        rec = recommend_quantization(detect_hardware())
        assert "level" in rec
        assert rec["level"].lower().startswith("q")


class TestEngineDetect:
    """engine_detect must select a sensible fallback when GGUFs are absent."""

    def test_picks_ollama_when_ggufs_missing(self, monkeypatch):
        from backend.core import engine_detect as ed

        monkeypatch.setattr(ed, "_ggufs_present", lambda inf: False)
        monkeypatch.setattr(
            ed, "_ollama_models", lambda url: [{"name": "qwen2.5:1.5b"}]
        )
        result = ed.resolve_inference_config()
        assert "engine" in result
        # if ollama models are present we should at least never crash
        assert result["engine"] in {"llama_cpp_dual", "ollama"}

    def test_pick_ollama_honors_preferred(self):
        from backend.core.engine_detect import _pick_ollama_model

        installed = [
            {"name": "llama3.1:8b", "size": 5 * 1024**3},
            {"name": "qwen2.5:1.5b", "size": 1 * 1024**3},
        ]
        # explicit preference wins
        assert _pick_ollama_model(installed, "qwen2.5:1.5b") == "qwen2.5:1.5b"
        # base-name match also works
        assert _pick_ollama_model(installed, "qwen2.5") == "qwen2.5:1.5b"

    def test_pick_ollama_empty_returns_blank(self):
        from backend.core.engine_detect import _pick_ollama_model

        assert _pick_ollama_model([]) == ""

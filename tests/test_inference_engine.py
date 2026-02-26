# Author: Bradley R. Kinnard
# test_inference_engine.py - unit tests for the multi-engine inference abstraction
# Tests engine selection, config handling, and the generate/stream interfaces.
# No external services (Ollama, vLLM) required.

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RLFUSION_DEVICE", "cpu")
os.environ.setdefault("RLFUSION_FORCE_CPU", "true")


class TestInferenceConfig:
    """Verify inference config loads with correct defaults and env overrides."""

    def test_default_config_loads(self):
        from backend.config import get_inference_config
        inf = get_inference_config()
        assert inf["engine"] == "ollama"
        assert "localhost" in inf["base_url"]
        assert isinstance(inf["model"], str)
        assert inf["max_concurrent"] >= 1
        assert inf["timeout_secs"] >= 1

    def test_env_override_engine(self):
        from backend.config import get_inference_config
        with patch.dict(os.environ, {"INFERENCE_ENGINE": "vllm"}):
            inf = get_inference_config()
            assert inf["engine"] == "vllm"

    def test_env_override_base_url(self):
        from backend.config import get_inference_config
        with patch.dict(os.environ, {"INFERENCE_BASE_URL": "http://gpu-box:8000"}):
            inf = get_inference_config()
            assert inf["base_url"] == "http://gpu-box:8000"

    def test_env_override_model(self):
        from backend.config import get_inference_config
        with patch.dict(os.environ, {"INFERENCE_MODEL": "llama3:70b"}):
            inf = get_inference_config()
            assert inf["model"] == "llama3:70b"

    def test_env_override_api_key(self):
        from backend.config import get_inference_config
        with patch.dict(os.environ, {"INFERENCE_API_KEY": "sk-test-key"}):
            inf = get_inference_config()
            assert inf["openai_api_key"] == "sk-test-key"


class TestInferenceEngineInit:
    """Verify InferenceEngine initializes correctly from config."""

    def test_engine_loads_from_config(self):
        from backend.core.model_router import InferenceEngine
        engine = InferenceEngine()
        assert engine.engine in ("ollama", "vllm", "tensorrt")
        assert isinstance(engine.model, str)
        assert isinstance(engine.base_url, str)

    def test_engine_defaults_to_ollama(self):
        from backend.core.model_router import InferenceEngine
        engine = InferenceEngine()
        assert engine.engine == "ollama"

    def test_engine_respects_env_override(self):
        from backend.core.model_router import InferenceEngine
        with patch.dict(os.environ, {"INFERENCE_ENGINE": "vllm", "INFERENCE_BASE_URL": "http://vllm:8000"}):
            engine = InferenceEngine()
            assert engine.engine == "vllm"
            assert engine.base_url == "http://vllm:8000"


class TestInferenceEngineOllama:
    """Test the Ollama backend path with mocked SDK calls."""

    def test_generate_calls_ollama_client(self):
        from backend.core.model_router import InferenceEngine

        engine = InferenceEngine()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "test response"}}

        with patch("backend.core.model_router.InferenceEngine._ollama_generate", wraps=engine._ollama_generate):
            with patch("ollama.Client", return_value=mock_client):
                result = engine.generate(
                    messages=[{"role": "user", "content": "hello"}],
                    temperature=0.5,
                )
                assert result == "test response"

    def test_stream_yields_chunks(self):
        from backend.core.model_router import InferenceEngine

        engine = InferenceEngine()
        chunks = [
            {"message": {"content": "Hello"}},
            {"message": {"content": " world"}},
        ]
        mock_client = MagicMock()
        mock_client.chat.return_value = iter(chunks)

        with patch("ollama.Client", return_value=mock_client):
            result = list(engine.stream(
                messages=[{"role": "user", "content": "hello"}],
            ))
            assert result == ["Hello", " world"]


class TestInferenceEngineOpenAI:
    """Test the OpenAI-compatible backend path (vLLM/TensorRT)."""

    def test_generate_calls_openai_endpoint(self):
        from backend.core.model_router import InferenceEngine

        with patch.dict(os.environ, {"INFERENCE_ENGINE": "vllm", "INFERENCE_BASE_URL": "http://vllm:8000"}):
            engine = InferenceEngine()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "vllm response"}}]
            }
            mock_response.raise_for_status = MagicMock()

            with patch("httpx.post", return_value=mock_response) as mock_post:
                result = engine.generate(
                    messages=[{"role": "user", "content": "test"}],
                    temperature=0.3,
                )
                assert result == "vllm response"
                call_args = mock_post.call_args
                assert "v1/chat/completions" in call_args[0][0]

    def test_generate_sends_correct_payload(self):
        from backend.core.model_router import InferenceEngine

        with patch.dict(os.environ, {"INFERENCE_ENGINE": "vllm", "INFERENCE_BASE_URL": "http://vllm:8000"}):
            engine = InferenceEngine()

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ok"}}]
            }
            mock_response.raise_for_status = MagicMock()

            with patch("httpx.post", return_value=mock_response) as mock_post:
                engine.generate(
                    messages=[{"role": "user", "content": "test"}],
                    temperature=0.7, num_predict=200,
                )
                payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
                assert payload["temperature"] == 0.7
                assert payload["max_tokens"] == 200
                assert payload["stream"] is False

    def test_api_key_sent_in_header(self):
        from backend.core.model_router import InferenceEngine

        with patch.dict(os.environ, {
            "INFERENCE_ENGINE": "vllm",
            "INFERENCE_BASE_URL": "http://vllm:8000",
            "INFERENCE_API_KEY": "sk-test-123",
        }):
            engine = InferenceEngine()

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ok"}}]
            }
            mock_response.raise_for_status = MagicMock()

            with patch("httpx.post", return_value=mock_response) as mock_post:
                engine.generate(
                    messages=[{"role": "user", "content": "test"}],
                )
                headers = mock_post.call_args.kwargs.get("headers") or mock_post.call_args[1].get("headers")
                assert "Bearer sk-test-123" in headers.get("Authorization", "")


class TestInferenceEngineHealthCheck:
    """Test health check dispatching for both engine types."""

    def test_ollama_health_check_success(self):
        from backend.core.model_router import InferenceEngine
        engine = InferenceEngine()

        mock_response = MagicMock()
        mock_response.json.return_value = {"models": [{"name": "dolphin-llama3:8b"}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            assert engine.check_health() is True

    def test_ollama_health_check_model_missing(self):
        from backend.core.model_router import InferenceEngine
        engine = InferenceEngine()

        mock_response = MagicMock()
        mock_response.json.return_value = {"models": [{"name": "other-model:7b"}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            assert engine.check_health() is False

    def test_vllm_health_check_success(self):
        from backend.core.model_router import InferenceEngine

        with patch.dict(os.environ, {"INFERENCE_ENGINE": "vllm", "INFERENCE_BASE_URL": "http://vllm:8000"}):
            engine = InferenceEngine()

            mock_response = MagicMock()
            mock_response.json.return_value = {"data": [{"id": "dolphin-llama3:8b"}]}
            mock_response.raise_for_status = MagicMock()

            with patch("httpx.get", return_value=mock_response):
                assert engine.check_health() is True

    def test_health_check_handles_connection_error(self):
        from backend.core.model_router import InferenceEngine
        engine = InferenceEngine()

        with patch("httpx.get", side_effect=ConnectionError("refused")):
            assert engine.check_health() is False


class TestGetEngineSingleton:
    """Verify get_engine() returns a stable singleton."""

    def test_returns_same_instance(self):
        import backend.core.model_router as mr
        mr._engine = None  # reset singleton
        e1 = mr.get_engine()
        e2 = mr.get_engine()
        assert e1 is e2
        mr._engine = None  # cleanup


class TestModelRouterPreserved:
    """Verify existing ModelRouter MoE-selection behavior is unchanged."""

    def test_select_model_returns_general_when_disabled(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        router._enabled = False
        assert router.select_model("explain") == router.general_model

    def test_select_model_returns_string(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        model = router.select_model("generation")
        assert isinstance(model, str)
        assert len(model) > 0

    def test_register_and_select(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        router.register_model("code-llama:7b", ["critique"], priority=10)
        assert router.select_model("critique") == "code-llama:7b"

    def test_unregister_model(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        router.register_model("temp-model:1b", ["summarize"], priority=10)
        assert router.unregister_model("temp-model:1b") is True
        assert router.unregister_model("nonexistent") is False

    def test_list_models_returns_pool(self):
        from backend.core.model_router import ModelRouter
        router = ModelRouter()
        models = router.list_models()
        assert isinstance(models, list)
        assert len(models) >= 1
        assert "name" in models[0]


class TestConfigToggle:
    """Verify the engine can be toggled between ollama and vllm via config."""

    def test_ollama_engine_uses_ollama_backend(self):
        from backend.core.model_router import InferenceEngine
        engine = InferenceEngine()
        assert engine.engine == "ollama"

    def test_vllm_engine_uses_openai_backend(self):
        from backend.core.model_router import InferenceEngine
        with patch.dict(os.environ, {"INFERENCE_ENGINE": "vllm"}):
            engine = InferenceEngine()
            assert engine.engine == "vllm"

    def test_tensorrt_engine_uses_openai_backend(self):
        from backend.core.model_router import InferenceEngine
        with patch.dict(os.environ, {"INFERENCE_ENGINE": "tensorrt"}):
            engine = InferenceEngine()
            assert engine.engine == "tensorrt"

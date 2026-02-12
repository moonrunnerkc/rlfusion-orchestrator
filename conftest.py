# conftest.py — shared pytest configuration
# Defines custom markers for test categorization

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring a CUDA GPU (skipped in CI)")
    config.addinivalue_line("markers", "slow: mark test as slow-running (>30s)")
    config.addinivalue_line("markers", "integration: mark test as requiring external services (Ollama, Redis)")

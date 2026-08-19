"""
Overrides the LLM provider dependency for the whole test session so tests
run in milliseconds, not the production 2s-per-call.
"""

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_cache_service, get_coalescer, get_llm_provider
from app.llm.mock_provider import MockLLMProvider
from app.main import app
from app.services.cache_service import NoOpCacheService
from app.services.coalescer import NoOpCoalescer

FAST_LATENCY = 0.05


@pytest.fixture
def fast_llm_provider():
    return MockLLMProvider(simulated_latency_seconds=FAST_LATENCY)


@pytest.fixture(autouse=True)
def override_dependencies(fast_llm_provider):
    app.dependency_overrides[get_llm_provider] = lambda: fast_llm_provider
    app.dependency_overrides[get_cache_service] = lambda: NoOpCacheService()
    app.dependency_overrides[get_coalescer] = lambda: NoOpCoalescer()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)
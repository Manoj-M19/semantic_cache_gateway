"""
Phase 5 tests: streaming cache support.

Uses TestClient (sync) rather than AsyncClient here — TestClient fully
consumes a StreamingResponse into `response.text` for an in-process
test, which is exactly what's needed to assert on the complete SSE
payload; genuine concurrency (Phase 3's concern) isn't what's being
tested in this file.
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_cache_service, get_llm_provider
from app.llm.base import UpstreamLLMProvider
from app.llm.mock_provider import MockLLMProvider
from app.main import app
from app.services.qdrant_vector_service import QdrantVectorService
from app.services.semantic_cache_service import SemanticCacheService

API_KEY = "devkey123"


def _parse_sse_events(raw_text: str) -> list[dict | str]:
    events = []
    for line in raw_text.split("\n\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        events.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return events


@pytest.fixture
def cache_service(tmp_path):
    vector_service = QdrantVectorService(
        qdrant_path=str(tmp_path / "qdrant"), collection_name=f"test_{uuid.uuid4().hex}"
    )
    return SemanticCacheService(vector_service)


@pytest.fixture
def client_with(cache_service):
    def _make(provider: UpstreamLLMProvider):
        app.dependency_overrides[get_llm_provider] = lambda: provider
        app.dependency_overrides[get_cache_service] = lambda: cache_service
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


def test_streaming_miss_forwards_chunks_and_populates_cache(client_with):
    provider = MockLLMProvider(simulated_latency_seconds=0.02)
    client = client_with(provider)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is a semantic cache?"}],
            "stream": True,
        },
        headers={"X-API-Key": API_KEY},
    )

    assert resp.status_code == 200
    assert resp.headers["x-cache-status"] == "miss"

    events = _parse_sse_events(resp.text)
    assert events[-1] == "[DONE]"
    content_chunks = [
        e["choices"][0]["delta"].get("content")
        for e in events
        if e != "[DONE]" and e["choices"][0]["delta"].get("content")
    ]
    assert len(content_chunks) > 1
    assert provider.call_count == 1


def test_streaming_hit_replays_identical_chunks_without_calling_upstream(client_with):
    provider = MockLLMProvider(simulated_latency_seconds=0.02)
    client = client_with(provider)
    headers = {"X-API-Key": API_KEY}

    first = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is a semantic cache?"}],
            "stream": True,
        },
        headers=headers,
    )
    first_chunks = [
        e["choices"][0]["delta"].get("content")
        for e in _parse_sse_events(first.text)
        if e != "[DONE]" and e["choices"][0]["delta"].get("content")
    ]
    assert provider.call_count == 1

    second = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Can you explain what a semantic cache is?"}
            ],
            "stream": True,
        },
        headers=headers,
    )
    assert second.headers["x-cache-status"] == "hit"
    second_chunks = [
        e["choices"][0]["delta"].get("content")
        for e in _parse_sse_events(second.text)
        if e != "[DONE]" and e["choices"][0]["delta"].get("content")
    ]

    assert second_chunks == first_chunks
    assert provider.call_count == 1


def test_non_streaming_cache_entry_does_not_satisfy_a_streaming_request(client_with):
    provider = MockLLMProvider(simulated_latency_seconds=0.02)
    client = client_with(provider)
    headers = {"X-API-Key": API_KEY}

    non_streaming = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "What is a semantic cache?"}]},
        headers=headers,
    )
    assert non_streaming.json()["cache_status"] == "miss"
    assert provider.call_count == 1

    streaming = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is a semantic cache?"}],
            "stream": True,
        },
        headers=headers,
    )
    assert streaming.headers["x-cache-status"] == "miss"
    assert provider.call_count == 2


def test_streaming_timeout_sends_error_event_and_does_not_cache_partial_response(client_with):
    import os

    from app.config import get_settings

    slow_provider = MockLLMProvider(simulated_latency_seconds=1.0)
    client = client_with(slow_provider)

    get_settings.cache_clear()
    os.environ["UPSTREAM_TIMEOUT_SECONDS"] = "0.05"
    get_settings.cache_clear()
    try:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "this will stall"}],
                "stream": True,
            },
            headers={"X-API-Key": API_KEY},
        )
    finally:
        del os.environ["UPSTREAM_TIMEOUT_SECONDS"]
        get_settings.cache_clear()

    events = _parse_sse_events(resp.text)
    error_events = [e for e in events if e != "[DONE]" and "error" in e]
    assert len(error_events) == 1
    assert error_events[0]["error"]["type"] == "upstream_timeout"

    slow_provider2 = MockLLMProvider(simulated_latency_seconds=0.02)
    app.dependency_overrides[get_llm_provider] = lambda: slow_provider2
    retry = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "this will stall"}], "stream": True},
        headers={"X-API-Key": API_KEY},
    )
    assert retry.headers["x-cache-status"] == "miss"
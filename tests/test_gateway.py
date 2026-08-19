from app.dependencies import get_llm_provider
from app.llm.mock_provider import MockLLMProvider
from app.main import app

API_KEY = "devkey123"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_chat_completions_requires_auth(client):
    resp = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 401


def test_chat_completions_success(client, fast_llm_provider):
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "What is a semantic cache?"}]},
        headers={"X-API-Key": API_KEY},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_status"] == "miss"
    assert "mock-llm" in body["choices"][0]["message"]["content"]
    assert body["latency_ms"] > 0


def test_no_cache_yet_every_request_hits_upstream(client, fast_llm_provider):
    """CacheService is wired in but is a NoOp in Phase 1 — this asserts
    that explicitly. When Phase 2 swaps in a real CacheService, this
    exact test should start failing at `after == before + 2`, which is
    the signal that real caching has landed."""
    before = fast_llm_provider.call_count
    payload = {"messages": [{"role": "user", "content": "identical prompt"}]}
    client.post("/v1/chat/completions", json=payload, headers={"X-API-Key": API_KEY})
    client.post("/v1/chat/completions", json=payload, headers={"X-API-Key": API_KEY})
    assert fast_llm_provider.call_count == before + 2


def test_missing_payload_returns_standardized_422(client):
    resp = client.post("/v1/chat/completions", json={}, headers={"X-API-Key": API_KEY})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error_type"] == "validation_error"
    assert "messages" in body["detail"]


def test_upstream_timeout_returns_504(client):
    """Forces a real timeout: a provider slower than the configured
    timeout. This exercises the exact failure path a production upstream
    outage would hit, not just the happy path."""
    slow_provider = MockLLMProvider(simulated_latency_seconds=5.0)
    app.dependency_overrides[get_llm_provider] = lambda: slow_provider

    from app.config import get_settings

    get_settings.cache_clear()
    import os

    os.environ["UPSTREAM_TIMEOUT_SECONDS"] = "0.1"
    get_settings.cache_clear()

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "this will time out"}]},
        headers={"X-API-Key": API_KEY},
    )

    del os.environ["UPSTREAM_TIMEOUT_SECONDS"]
    get_settings.cache_clear()

    assert resp.status_code == 504
    body = resp.json()
    assert body["error_type"] == "upstream_timeout"
"""
Phase 3 tests: in-flight request coalescing.

Uses fakeredis so these tests don't need a real Redis server, and
httpx.AsyncClient + ASGITransport + asyncio.gather so multiple requests
genuinely run concurrently within one event loop. A thread-pool-based
test would only prove multiple threads can call the app — it wouldn't
exercise the actual interleaving that coalescing has to handle correctly.
"""

import asyncio

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient

from app.dependencies import get_cache_service, get_coalescer, get_llm_provider
from app.llm.mock_provider import MockLLMProvider
from app.main import app
from app.services.cache_service import NoOpCacheService
from app.services.coalescer import RedisCoalescer

API_KEY = "devkey123"


@pytest.fixture
def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def coalescing_provider():
    return MockLLMProvider(simulated_latency_seconds=0.3)


@pytest.fixture
def coalescer(fake_redis):
    return RedisCoalescer(
        fake_redis, lock_ttl_seconds=5.0, poll_interval_seconds=0.01, max_wait_seconds=3.0
    )


@pytest.mark.asyncio
async def test_concurrent_identical_requests_coalesce_to_one_upstream_call(
    coalescing_provider, coalescer
):
    app.dependency_overrides[get_llm_provider] = lambda: coalescing_provider
    app.dependency_overrides[get_cache_service] = lambda: NoOpCacheService()
    app.dependency_overrides[get_coalescer] = lambda: coalescer
    try:
        payload = {"messages": [{"role": "user", "content": "identical concurrent prompt"}]}
        headers = {"X-API-Key": API_KEY}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            responses = await asyncio.gather(
                *[ac.post("/v1/chat/completions", json=payload, headers=headers) for _ in range(20)]
            )
    finally:
        app.dependency_overrides.clear()

    assert coalescing_provider.call_count == 1

    statuses = [r.json()["cache_status"] for r in responses]
    assert statuses.count("miss") == 1
    assert statuses.count("coalesced") == 19

    texts = {r.json()["choices"][0]["message"]["content"] for r in responses}
    assert len(texts) == 1


@pytest.mark.asyncio
async def test_different_prompts_do_not_coalesce(coalescing_provider, coalescer):
    app.dependency_overrides[get_llm_provider] = lambda: coalescing_provider
    app.dependency_overrides[get_cache_service] = lambda: NoOpCacheService()
    app.dependency_overrides[get_coalescer] = lambda: coalescer
    try:
        headers = {"X-API-Key": API_KEY}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            responses = await asyncio.gather(
                ac.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "prompt A"}]}, headers=headers),
                ac.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "prompt B"}]}, headers=headers),
            )
    finally:
        app.dependency_overrides.clear()

    assert coalescing_provider.call_count == 2
    assert all(r.json()["cache_status"] == "miss" for r in responses)
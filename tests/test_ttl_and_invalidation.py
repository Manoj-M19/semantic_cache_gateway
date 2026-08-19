"""
Phase 4 tests: TTL expiry and manual invalidation.

TTL tests talk to QdrantVectorService directly — waiting out a TTL
through 15 HTTP round trips per test would work but adds nothing over
testing the mechanism where it actually lives. The invalidation
endpoints ARE tested through the real HTTP layer, since proving the
admin API itself works end-to-end is the actual point there.
"""

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_cache_service
from app.main import app
from app.services.qdrant_vector_service import QdrantVectorService
from app.services.semantic_cache_service import SemanticCacheService

API_KEY = "devkey123"


@pytest.fixture
def vector_service(tmp_path):
    def _make(ttl_seconds: float = 3600.0):
        return QdrantVectorService(
            qdrant_path=str(tmp_path / "qdrant"),
            collection_name=f"test_{uuid.uuid4().hex}",
            cache_ttl_seconds=ttl_seconds,
        )

    return _make


@pytest.mark.asyncio
async def test_entry_expires_after_ttl(vector_service):
    vs = vector_service(ttl_seconds=0.2)
    embedding = await vs.embed("a prompt")
    await vs.upsert("a prompt", embedding, "an answer")

    hit = await vs.search(embedding, similarity_threshold=0.9)
    assert hit is not None
    assert hit.cached_response_text == "an answer"

    await asyncio.sleep(0.3)

    miss = await vs.search(embedding, similarity_threshold=0.9)
    assert miss is None


@pytest.mark.asyncio
async def test_delete_by_text_removes_only_the_matching_entry(vector_service):
    vs = vector_service()
    emb_a = await vs.embed("prompt A")
    emb_b = await vs.embed("prompt B")
    await vs.upsert("prompt A", emb_a, "answer A")
    await vs.upsert("prompt B", emb_b, "answer B")

    removed = await vs.delete_by_text("prompt A")
    assert removed is True

    assert await vs.search(emb_a, similarity_threshold=0.9) is None
    still_there = await vs.search(emb_b, similarity_threshold=0.9)
    assert still_there is not None
    assert still_there.cached_response_text == "answer B"


@pytest.mark.asyncio
async def test_delete_by_text_on_nonexistent_entry_returns_false(vector_service):
    vs = vector_service()
    removed = await vs.delete_by_text("never cached")
    assert removed is False


@pytest.mark.asyncio
async def test_clear_all_removes_everything_and_returns_count(vector_service):
    vs = vector_service()
    for i in range(4):
        emb = await vs.embed(f"prompt {i}")
        await vs.upsert(f"prompt {i}", emb, f"answer {i}")

    deleted = await vs.clear_all()
    assert deleted == 4

    emb = await vs.embed("prompt 0")
    assert await vs.search(emb, similarity_threshold=0.9) is None


@pytest.mark.asyncio
async def test_purge_expired_removes_only_expired_entries(vector_service):
    vs = vector_service(ttl_seconds=0.2)
    emb_short = await vs.embed("short-lived prompt")
    await vs.upsert("short-lived prompt", emb_short, "short answer")

    await asyncio.sleep(0.3)

    emb_fresh = await vs.embed("fresh prompt")
    await vs.upsert("fresh prompt", emb_fresh, "fresh answer")

    purged = await vs.purge_expired()
    assert purged == 1

    still_there = await vs.search(emb_fresh, similarity_threshold=0.9)
    assert still_there is not None


def test_invalidate_endpoint_evicts_a_semantically_matching_prompt(tmp_path):
    from app.dependencies import get_llm_provider
    from app.llm.mock_provider import MockLLMProvider

    vector_service = QdrantVectorService(
        qdrant_path=str(tmp_path / "qdrant"), collection_name=f"test_{uuid.uuid4().hex}"
    )
    cache_service = SemanticCacheService(vector_service)
    fast_provider = MockLLMProvider(simulated_latency_seconds=0.05)

    app.dependency_overrides[get_llm_provider] = lambda: fast_provider
    app.dependency_overrides[get_cache_service] = lambda: cache_service
    client = TestClient(app)
    headers = {"X-API-Key": API_KEY}

    try:
        first = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "What is a semantic cache?"}]},
            headers=headers,
        )
        assert first.json()["cache_status"] == "miss"

        confirm_cached = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Can you explain what a semantic cache is?"}]},
            headers=headers,
        )
        assert confirm_cached.json()["cache_status"] == "hit"

        inv = client.post(
            "/v1/cache/invalidate", json={"text": "What is a semantic cache?"}, headers=headers
        )
        assert inv.status_code == 200
        assert inv.json()["invalidated"] is True

        after_invalidate = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Can you explain what a semantic cache is?"}]},
            headers=headers,
        )
        assert after_invalidate.json()["cache_status"] == "miss"
        assert fast_provider.call_count == 2
    finally:
        app.dependency_overrides.clear()


def test_clear_cache_endpoint_requires_auth():
    client = TestClient(app)
    resp = client.delete("/v1/cache")
    assert resp.status_code == 401
"""
Phase 2 tests: real embeddings, real Qdrant search — nothing about the
caching logic is mocked, only the upstream LLM. Each test gets its own
isolated Qdrant path + collection name, so tests can't see each other's
cached entries. Similarity numbers below were measured empirically:
real paraphrases scored 0.955-0.984, unrelated prompts ~0.79-0.81.
"""

import uuid

import pytest

from app.dependencies import get_cache_service, get_llm_provider
from app.main import app
from app.services.qdrant_vector_service import QdrantVectorService
from app.services.semantic_cache_service import SemanticCacheService

API_KEY = "devkey123"


@pytest.fixture
def semantic_cache_service(tmp_path):
    vector_service = QdrantVectorService(
        qdrant_path=str(tmp_path / "qdrant"),
        collection_name=f"test_{uuid.uuid4().hex}",
    )
    return SemanticCacheService(vector_service)


@pytest.fixture
def semantic_client(client, fast_llm_provider, semantic_cache_service):
    app.dependency_overrides[get_llm_provider] = lambda: fast_llm_provider
    app.dependency_overrides[get_cache_service] = lambda: semantic_cache_service
    return client


def _ask(client, text):
    return client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": text}]},
        headers={"X-API-Key": API_KEY},
    )


def test_paraphrase_is_a_cache_hit(semantic_client, fast_llm_provider):
    first = _ask(semantic_client, "What is a semantic cache?")
    assert first.json()["cache_status"] == "miss"

    second = _ask(semantic_client, "Can you explain what a semantic cache is?")
    body = second.json()

    assert body["cache_status"] == "hit"
    assert body["similarity_score"] >= 0.95
    assert body["choices"][0]["message"]["content"] == first.json()["choices"][0]["message"]["content"]
    assert fast_llm_provider.call_count == 1


def test_unrelated_prompt_is_a_cache_miss(semantic_client, fast_llm_provider):
    _ask(semantic_client, "What is a semantic cache?")
    second = _ask(semantic_client, "What is the capital of France?")
    assert second.json()["cache_status"] == "miss"
    assert fast_llm_provider.call_count == 2


def test_first_request_to_empty_cache_is_always_a_miss(semantic_client, fast_llm_provider):
    resp = _ask(semantic_client, "Anything at all — the cache starts empty")
    assert resp.json()["cache_status"] == "miss"
    assert fast_llm_provider.call_count == 1
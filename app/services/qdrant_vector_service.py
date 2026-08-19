import asyncio
import logging
import time
import uuid

import spacy
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from app.config import get_settings
from app.services.vector_service import VectorSearchResult, VectorService

logger = logging.getLogger("semcache.vector_service")


class QdrantVectorService(VectorService):
    def __init__(
        self,
        qdrant_path: str | None = None,
        qdrant_url: str | None = None,
        collection_name: str | None = None,
        cache_ttl_seconds: float | None = None,
    ):
        settings = get_settings()

        self._nlp = spacy.load("en_core_web_md")
        self._vector_size = self._nlp.vocab.vectors_length

        url = qdrant_url if qdrant_url is not None else settings.qdrant_url
        path = qdrant_path if qdrant_path is not None else settings.qdrant_local_path
        self._client = AsyncQdrantClient(url=url) if url else AsyncQdrantClient(path=path)

        self._collection = collection_name or settings.qdrant_collection
        self._ttl_seconds = (
            cache_ttl_seconds if cache_ttl_seconds is not None else settings.cache_ttl_seconds
        )
        self._collection_ready = False

    async def _create_collection(self) -> None:
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
        )
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="expires_at",
            field_schema=PayloadSchemaType.FLOAT,
        )
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="text",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info(
            "Created Qdrant collection",
            extra={"extra_fields": {"collection": self._collection, "vector_size": self._vector_size}},
        )

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        existing = [c.name for c in (await self._client.get_collections()).collections]
        if self._collection not in existing:
            await self._create_collection()
        self._collection_ready = True

    async def embed(self, text: str) -> list[float]:
        doc = await asyncio.to_thread(self._nlp, text)
        return doc.vector.tolist()

    async def search(
        self, embedding: list[float], similarity_threshold: float
    ) -> VectorSearchResult | None:
        await self._ensure_collection()

        now = time.time()
        response = await self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            limit=1,
            query_filter=Filter(must=[FieldCondition(key="expires_at", range=Range(gt=now))]),
        )

        if not response.points:
            return None

        top = response.points[0]
        if top.score < similarity_threshold:
            return None

        return VectorSearchResult(
            cached_response_text=top.payload["response_text"],
            similarity_score=top.score,
            response_chunks=top.payload.get("response_chunks"),
        )
    
    async def upsert(
        self,
        text: str,
        embedding: list[float],
        response_text: str,
        response_chunks: list[str] | None = None,
    ) -> None:
        await self._ensure_collection()
        expires_at = time.time() + self._ttl_seconds
        payload = {"text": text, "response_text": response_text, "expires_at": expires_at}
        if response_chunks is not None:
            payload["response_chunks"] = response_chunks
        await self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=payload)],
        )

    async def delete_by_text(self, text: str) -> bool:
        await self._ensure_collection()
        text_filter = Filter(must=[FieldCondition(key="text", match=MatchValue(value=text))])

        existing = await self._client.count(collection_name=self._collection, count_filter=text_filter)
        if existing.count == 0:
            return False

        await self._client.delete(
            collection_name=self._collection, points_selector=FilterSelector(filter=text_filter)
        )
        return True

    async def clear_all(self) -> int:
        await self._ensure_collection()
        count_result = await self._client.count(collection_name=self._collection)
        total = count_result.count

        # Recreate rather than delete-by-empty-filter: dropping and
        # rebuilding the collection is an unambiguous "everything is
        # gone," rather than relying on an empty filter's match-all
        # semantics, which is the kind of detail worth not gambling on
        # for a destructive operation.
        await self._client.delete_collection(collection_name=self._collection)
        self._collection_ready = False
        await self._ensure_collection()

        return total

    async def purge_expired(self) -> int:
        await self._ensure_collection()
        now = time.time()
        expired_filter = Filter(must=[FieldCondition(key="expires_at", range=Range(lte=now))])

        count_result = await self._client.count(
            collection_name=self._collection, count_filter=expired_filter
        )
        expired_count = count_result.count
        if expired_count > 0:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=FilterSelector(filter=expired_filter),
            )

        return expired_count
from app.config import get_settings
from app.schemas import ChatCompletionRequest
from app.services.cache_service import CachedResult, CacheService
from app.services.vector_service import VectorService
from app.utils import extract_last_user_message


class SemanticCacheService(CacheService):
    def __init__(self, vector_service: VectorService):
        self._vector_service = vector_service

    async def get(self, request: ChatCompletionRequest) -> CachedResult | None:
        settings = get_settings()
        text = extract_last_user_message(request.messages)
        if not text:
            return None

        embedding = await self._vector_service.embed(text)
        result = await self._vector_service.search(embedding, settings.similarity_threshold)
        if result is None:
            return None

        return CachedResult(
            response_text=result.cached_response_text,
            similarity_score=result.similarity_score,
            response_chunks=result.response_chunks,
        )

    async def set(
        self,
        request: ChatCompletionRequest,
        response_text: str,
        response_chunks: list[str] | None = None,
    ) -> None:
        text = extract_last_user_message(request.messages)
        if not text:
            return

        embedding = await self._vector_service.embed(text)
        await self._vector_service.upsert(text, embedding, response_text, response_chunks)

    async def invalidate(self, text: str) -> bool:
        return await self._vector_service.delete_by_text(text)

    async def clear_all(self) -> int:
        return await self._vector_service.clear_all()

    async def purge_expired(self) -> int:
        return await self._vector_service.purge_expired()
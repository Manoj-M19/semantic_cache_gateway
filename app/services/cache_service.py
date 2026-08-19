from abc import ABC, abstractmethod

from app.schemas import ChatCompletionRequest

class CachedResult:
    """What a cache HIT returns: the cached text plus the similarity
    score that justified reusing it, plus the original streamed chunk
    sequence if this entry came from a streaming response."""

    __slots__ = ("response_chunks", "response_text", "similarity_score")

    def __init__(
        self,
        response_text: str,
        similarity_score: float,
        response_chunks: list[str] | None = None,
    ):
        self.response_text = response_text
        self.similarity_score = similarity_score
        self.response_chunks = response_chunks

class CacheService(ABC):
    @abstractmethod
    async def get(self, request: ChatCompletionRequest) -> CachedResult | None:
        raise NotImplementedError

    @abstractmethod
    async def set(
        self,
        request: ChatCompletionRequest,
        response_text: str,
        response_chunks: list[str] | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def invalidate(self, text: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def clear_all(self) -> int:
        raise NotImplementedError

    @abstractmethod
    async def purge_expired(self) -> int:
        raise NotImplementedError

class NoOpCacheService(CacheService):
    async def get(self, request: ChatCompletionRequest) -> CachedResult | None:
        return None

    async def set(
        self,
        request: ChatCompletionRequest,
        response_text: str,
        response_chunks: list[str] | None = None,
    ) -> None:
        return None

    async def invalidate(self, text: str) -> bool:
        return False

    async def clear_all(self) -> int:
        return 0

    async def purge_expired(self) -> int:
        return 0
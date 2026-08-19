"""
VectorService is the interface Phase 2 implements against Qdrant.
"""

from abc import ABC, abstractmethod
from typing import NamedTuple


class VectorSearchResult(NamedTuple):
    cached_response_text: str
    similarity_score: float
    response_chunks: list[str] | None = None


class VectorService(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    async def search(
        self, embedding: list[float], similarity_threshold: float
    ) -> VectorSearchResult | None:
        raise NotImplementedError

    @abstractmethod
    async def upsert(
        self,
        text: str,
        embedding: list[float],
        response_text: str,
        response_chunks: list[str] | None = None,
    ) -> None:
        """`response_chunks`, when given, is the original streamed
        chunk sequence — kept separately from `response_text` so a
        streaming cache HIT can replay the exact same chunk boundaries
        a second caller saw, not a re-chunked approximation."""
        raise NotImplementedError

    @abstractmethod
    async def delete_by_text(self, text: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def clear_all(self) -> int:
        raise NotImplementedError

    @abstractmethod
    async def purge_expired(self) -> int:
        raise NotImplementedError
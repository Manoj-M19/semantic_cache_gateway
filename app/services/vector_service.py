from abc import ABC, abstractmethod
from typing import NamedTuple

class VectorSearchResult(NamedTuple):
    cached_response_text: str
    similarity_score: float

class VectorService(ABC):
    @abstractmethod
    async def embed(self, text:str) -> list[float]:
        """Convert text into an embedding vector. Implementations choose
        the model; callers only depend on getting a fixed-length float
        vector back."""
        raise NotImplementedError

    @abstractmethod
    async def search(
        self, embedding: list[float], similarity_threshold: float
    ) -> VectorSearchResult | None:
        """Return the closest stored vector if its cosine similarity
        meets or exceeds `similarity_threshold`, else None. The threshold
        is passed in (not baked into the implementation) so it stays a
        single tunable in Settings, not scattered across the codebase."""
        raise NotImplementedError

    @abstractmethod
    async def upsert(self, text: str, embedding:list[float], response_text: str) -> None:
        """Store a new prompt/embedding/response triple for future
        similarity lookups."""
        raise NotImplementedError
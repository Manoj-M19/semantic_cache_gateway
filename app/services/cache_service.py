"""
CacheService is the seam Phase 2 plugs real caching into.

Phase 1 deliberately ships a NoOpCacheService — every lookup is a miss,
every write does nothing. That's not a placeholder to apologize for: it's
what lets the router be written *once*, against the interface, and never
touch again when Phase 2 swaps in a Qdrant-backed implementation. The
router doesn't know or care which implementation it's holding.
"""
from abc import ABC, abstractmethod

from app.schemas import ChatCompletionRequest

class CachedResult:
     """What a cache HIT returns: the cached text plus the similarity
    score that justified reusing it (useful for logging/debugging why a
    particular response was served)."""

     __slots__ = ("response_text", "similarity_score")

     def __init__(self, response_text: str, similarity_score: float):
          self.response_text = response_text
          self.similarity_score = similarity_score

class CacheService(ABC):
     @abstractmethod
     async def get(self, request: ChatCompletionRequest) -> CachedResult | None:
          """Return a CachedResult on a semantic-similarity hit, or None on
        a miss. None is the *safe* default — a false HIT returns a wrong
        answer to a real user, a false MISS just costs one extra upstream
        call. This service must be conservative."""
          raise NotImplementedError

     @abstractmethod
     async def set(self, request: ChatCompletionRequest, response_text: str) -> None:
           """Persist a fresh upstream response so a future semantically
        similar request can be served from cache."""
           raise NotImplementedError

class NoOpCacheService(CacheService):
      """Phase 1 implementation: always a miss, writes are discarded.
    Exists so the full request path — including the get/set calls — is
    exercised and tested *before* Phase 2 adds real caching logic on top
    of a already-proven code path."""

      async def get(self, request: ChatCompletionRequest) -> CachedResult | None:
           return None

      async def set(self, request:ChatCompletionRequest, response_text: str) -> None:
           return None
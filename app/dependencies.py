from app.llm.base import UpstreamLLMProvider
from app.llm.mock_provider import MockLLMProvider
from app.redis_client import get_redis_client
from app.services.cache_service import CacheService
from app.services.coalescer import Coalescer, RedisCoalescer
from app.services.qdrant_vector_service import QdrantVectorService
from app.services.semantic_cache_service import SemanticCacheService
from app.services.vector_service import VectorService

_llm_provider: UpstreamLLMProvider = MockLLMProvider()
_vector_service: VectorService = QdrantVectorService()
_cache_service: CacheService = SemanticCacheService(_vector_service)
_coalescer: Coalescer = RedisCoalescer(get_redis_client())

def get_llm_provider() -> UpstreamLLMProvider:
    return _llm_provider

def get_cache_service() -> CacheService:
    return _cache_service

def get_vector_service() -> VectorService:
    return _vector_service

def get_coalescer() -> Coalescer:
    return _coalescer
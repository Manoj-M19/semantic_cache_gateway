"""
Central settings, same pattern as InferGate: one object, env-driven, so the
same code runs unmodified in local dev, Docker, and production — only the
.env / environment variables change.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "SemCache"
    environment: str = "development"
    log_level: str = "INFO"

    api_keys: str = "devkey123"

    redis_url: str = "redis://localhost:6379/0"

    # Qdrant: if qdrant_url is unset, we run Qdrant in embedded/local mode
    # (good for dev + tests, zero extra infra). Set qdrant_url to point at
    # a real Qdrant server in docker-compose / production.
    qdrant_url: str = ""
    qdrant_local_path: str = "./qdrant_data"
    qdrant_collection: str = "semcache_prompts"

    # Cosine similarity floor for treating a cached entry as a safe reuse.
    # 0.95 is intentionally strict — a cache HIT that returns a wrong
    # answer is worse than a cache MISS, so we'd rather under-cache.
    similarity_threshold: float = 0.95

    # Any single upstream call — real or mocked — that exceeds this is
    # treated as a gateway-level failure, not left to hang. 10s gives the
    # 2s mock provider comfortable headroom while still being a real
    # bound in production.
    upstream_timeout_seconds: float = 10.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def api_keys_set(self) -> set:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

@lru_cache
def get_settings() -> Settings:
    return Settings()
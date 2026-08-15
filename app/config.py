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

    qdrant_url: str = ""
    qdrant_local_path: str = "./qdrant_data"
    qdrant_collection: str = "semcache_prompts"

    similarity_threshold: float = 0.95

    # How long a cached entry stays valid. After this, it's invisible to
    # search (TTL enforced at query time via a Qdrant filter) even
    # before a purge_expired() sweep actually deletes it from storage.
    cache_ttl_seconds: float = 3600.0

    upstream_timeout_seconds: float = 10.0

    coalesce_lock_ttl_seconds: float = 30.0
    coalesce_poll_interval_seconds: float = 0.05
    coalesce_max_wait_seconds: float = 15.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def api_keys_set(self) -> set:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

import redis.asyncio as redis

from app.exceptions import CoalesceTimeoutError

logger = logging.getLogger("semcache.coalescer")

WorkFn = Callable[[], Awaitable[dict]]


class Coalescer(ABC):
    @abstractmethod
    async def coalesce(self, key: str, work: WorkFn) -> tuple[dict, bool]:
        """Run `work()` for the first caller with a given `key`; every
        concurrent caller with the *same* key gets that call's result
        instead of running `work()` themselves. Returns (result,
        was_leader) — was_leader tells the caller whether they did the
        real work or reused someone else's."""
        raise NotImplementedError


class NoOpCoalescer(Coalescer):
    """No deduplication — every call runs `work()` independently and is
    always its own leader. The safe default for tests not specifically
    testing coalescing, exactly like NoOpCacheService was in Phase 1."""

    async def coalesce(self, key: str, work: WorkFn) -> tuple[dict, bool]:
        result = await work()
        return result, True


class RedisCoalescer(Coalescer):
    def __init__(
        self,
        redis_client: redis.Redis,
        lock_ttl_seconds: float = 30.0,
        poll_interval_seconds: float = 0.05,
        max_wait_seconds: float = 15.0,
    ):
        self._redis = redis_client
        self._lock_ttl_seconds = lock_ttl_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._max_wait_seconds = max_wait_seconds

    async def coalesce(self, key: str, work: WorkFn) -> tuple[dict, bool]:
        lock_key = f"coalesce:lock:{key}"
        result_key = f"coalesce:result:{key}"

        # int(), not float — redis-py's `ex` requires an int or
        # timedelta. max(1, ...) guards against a sub-second TTL
        # rounding down to 0, which Redis would reject.
        lock_ttl = max(1, round(self._lock_ttl_seconds))
        acquired = await self._redis.set(lock_key, "1", nx=True, ex=lock_ttl)

        if acquired:
            try:
                result = await work()
                # Short TTL — just long enough for any followers still
                # polling to grab it before it's cleaned up.
                await self._redis.set(result_key, json.dumps(result), ex=5)
                return result, True
            finally:
                await self._redis.delete(lock_key)

        result = await self._wait_for_result(key, result_key, lock_key)
        return result, False

    async def _wait_for_result(self, key: str, result_key: str, lock_key: str) -> dict:
        waited = 0.0
        while waited < self._max_wait_seconds:
            raw = await self._redis.get(result_key)
            if raw is not None:
                return json.loads(raw)

            if not await self._redis.exists(lock_key):
                # Leader is gone. Check once more before giving up — it
                # may have finished and written the result *between* our
                # last two reads (result written, then lock deleted);
                # without this, that legitimate success looks identical
                # to a crash.
                raw = await self._redis.get(result_key)
                if raw is not None:
                    return json.loads(raw)
                break

            await asyncio.sleep(self._poll_interval_seconds)
            waited += self._poll_interval_seconds

        logger.warning(
            "Coalesce follower timed out",
            extra={"extra_fields": {"key": key, "waited_seconds": round(waited, 2)}},
        )
        raise CoalesceTimeoutError(key=key, waited_seconds=waited)
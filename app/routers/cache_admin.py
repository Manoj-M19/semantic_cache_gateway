import logging

from fastapi import APIRouter, Depends

from app.auth import verify_api_key
from app.dependencies import get_cache_service
from app.schemas import (
    ClearCacheResponse,
    InvalidateRequest,
    InvalidateResponse,
    PurgeExpiredResponse,
)
from app.services.cache_service import CacheService

router = APIRouter(prefix="/v1/cache", tags=["cache-admin"])
logger = logging.getLogger("semcache.cache_admin")


@router.post("/invalidate", response_model=InvalidateResponse, dependencies=[Depends(verify_api_key)])
async def invalidate(
    body: InvalidateRequest, cache_service: CacheService = Depends(get_cache_service)
) -> InvalidateResponse:
    removed = await cache_service.invalidate(body.text)
    logger.info(
        "Cache entry invalidated" if removed else "Invalidate requested but nothing matched",
        extra={"extra_fields": {"removed": removed}},
    )
    return InvalidateResponse(invalidated=removed)


@router.delete("", response_model=ClearCacheResponse, dependencies=[Depends(verify_api_key)])
async def clear_cache(cache_service: CacheService = Depends(get_cache_service)) -> ClearCacheResponse:
    count = await cache_service.clear_all()
    logger.warning("Cache cleared entirely", extra={"extra_fields": {"deleted_count": count}})
    return ClearCacheResponse(deleted_count=count)


@router.post("/purge-expired", response_model=PurgeExpiredResponse, dependencies=[Depends(verify_api_key)])
async def purge_expired(cache_service: CacheService = Depends(get_cache_service)) -> PurgeExpiredResponse:
    count = await cache_service.purge_expired()
    logger.info("Expired entries purged", extra={"extra_fields": {"purged_count": count}})
    return PurgeExpiredResponse(purged_count=count)
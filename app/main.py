import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.exceptions import CoalesceTimeoutError, UpstreamTimeoutError
from app.logging_config import configure_logging
from app.routers import gateway, health, cache_admin
from app.schemas import ErrorResponse

configure_logging()
logger = logging.getLogger("semcache")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SemCache starting", extra={"extra_fields": {"environment": settings.environment}})
    yield
    logger.info("SemCache shutting down")


app = FastAPI(
    title=settings.app_name,
    description="A semantic caching gateway for LLM APIs — cuts cost and "
    "latency by recognizing near-duplicate prompts.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(gateway.router)
app.include_router(cache_admin.router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.info(
        "Request validation failed",
        extra={"extra_fields": {"path": str(request.url), "errors": exc.errors()}},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=ErrorResponse(
            error_type="validation_error",
            detail="; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()),
        ).model_dump(),
    )


@app.exception_handler(UpstreamTimeoutError)
async def upstream_timeout_handler(request: Request, exc: UpstreamTimeoutError) -> JSONResponse:
    logger.warning(
        "Upstream provider timed out",
        extra={"extra_fields": {"path": str(request.url), "provider": exc.provider_name, "timeout_seconds": exc.timeout_seconds}},
    )
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content=ErrorResponse(error_type="upstream_timeout", detail=str(exc)).model_dump(),
    )


@app.exception_handler(CoalesceTimeoutError)
async def coalesce_timeout_handler(request: Request, exc: CoalesceTimeoutError) -> JSONResponse:
    logger.warning(
        "Coalesce follower timed out",
        extra={"extra_fields": {"path": str(request.url), "key": exc.key, "waited_seconds": exc.waited_seconds}},
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(error_type="coalesce_timeout", detail=str(exc)).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception", extra={"extra_fields": {"path": str(request.url)}})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error_type="internal_error", detail="Internal server error").model_dump(),
    )
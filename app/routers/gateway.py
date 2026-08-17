import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.auth import verify_api_key
from app.config import get_settings
from app.dependencies import get_cache_service, get_coalescer, get_llm_provider
from app.exceptions import UpstreamTimeoutError
from app.llm.base import UpstreamLLMProvider
from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamChunk,
    ChatMessage,
    StreamDelta,
)
from app.services.cache_service import CacheService
from app.services.coalescer import Coalescer
from app.sse import SSE_DONE, format_sse_chunk
from app.utils import extract_last_user_message

router = APIRouter(prefix="/v1", tags=["gateway"])
logger = logging.getLogger("semcache.gateway")


@router.post("/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(
    body: ChatCompletionRequest,
    llm_provider: UpstreamLLMProvider = Depends(get_llm_provider),
    cache_service: CacheService = Depends(get_cache_service),
    coalescer: Coalescer = Depends(get_coalescer),
):
    if body.stream:
        return await _stream_chat_completion(body, llm_provider, cache_service)
    return await _non_streaming_chat_completion(body, llm_provider, cache_service, coalescer)


async def _non_streaming_chat_completion(
    body: ChatCompletionRequest,
    llm_provider: UpstreamLLMProvider,
    cache_service: CacheService,
    coalescer: Coalescer,
) -> ChatCompletionResponse:
    """Unchanged from Phase 3/4: cache-check -> coalesced upstream call
    on miss -> cache-write. Extracted into its own function once the
    route handler had to branch on streaming, purely for readability."""
    start = time.perf_counter()
    settings = get_settings()

    text = extract_last_user_message(body.messages)
    coalesce_key = hashlib.sha256(text.encode()).hexdigest()

    async def do_work() -> dict:
        cached = await cache_service.get(body)
        if cached is not None:
            return {
                "response_text": cached.response_text,
                "cache_status": "hit",
                "similarity_score": cached.similarity_score,
            }

        try:
            reply_text = await asyncio.wait_for(
                llm_provider.complete(body.messages),
                timeout=settings.upstream_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise UpstreamTimeoutError(
                provider_name=llm_provider.name,
                timeout_seconds=settings.upstream_timeout_seconds,
            )

        await cache_service.set(body, reply_text)
        return {"response_text": reply_text, "cache_status": "miss", "similarity_score": None}

    work_result, was_leader = await coalescer.coalesce(coalesce_key, do_work)
    final_cache_status = work_result["cache_status"] if was_leader else "coalesced"

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "chat completion served",
        extra={
            "extra_fields": {
                "cache_status": final_cache_status,
                "was_leader": was_leader,
                "latency_ms": round(elapsed_ms, 2),
            }
        },
    )

    return ChatCompletionResponse(
        model=body.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=work_result["response_text"])
            )
        ],
        cache_status=final_cache_status,
        similarity_score=work_result.get("similarity_score"),
        latency_ms=round(elapsed_ms, 2),
    )


async def _stream_chat_completion(
    body: ChatCompletionRequest,
    llm_provider: UpstreamLLMProvider,
    cache_service: CacheService,
) -> StreamingResponse:
    cached = await cache_service.get(body)

    # A match with no stored chunks (e.g. originally cached by a
    # *non-streaming* request) is deliberately treated as a miss here,
    # not synthesized into a fake one-chunk "stream" — real chunking
    # from a real upstream call is more honest than faking the shape.
    if cached is not None and cached.response_chunks is not None:
        logger.info("streaming from cache", extra={"extra_fields": {"cache_status": "hit"}})
        return StreamingResponse(
            _replay_cached_stream(body.model, cached.response_chunks),
            media_type="text/event-stream",
            headers={"X-Cache-Status": "hit"},
        )

    logger.info("streaming from upstream", extra={"extra_fields": {"cache_status": "miss"}})
    return StreamingResponse(
        _stream_and_cache(body, llm_provider, cache_service),
        media_type="text/event-stream",
        headers={"X-Cache-Status": "miss"},
    )


async def _replay_cached_stream(model: str, chunks: list[str]) -> AsyncIterator[str]:
    """Replays a previously-cached stream's *exact* chunk boundaries —
    the original chunks themselves, not the final text re-split — so a
    second caller sees the same token-by-token shape the first one did.
    Delivered as fast as the network allows, not throttled to match the
    original's timing: reproducing the original latency on a cache HIT
    would defeat the entire point of caching."""
    for chunk in chunks:
        yield format_sse_chunk(
            ChatCompletionStreamChunk(
                model=model,
                choices=[ChatCompletionStreamChoice(delta=StreamDelta(content=chunk))],
            ).model_dump()
        )
    yield format_sse_chunk(
        ChatCompletionStreamChunk(
            model=model,
            choices=[ChatCompletionStreamChoice(delta=StreamDelta(), finish_reason="stop")],
        ).model_dump()
    )
    yield SSE_DONE


async def _iterate_with_timeout(aiter: AsyncIterator[str], timeout: float) -> AsyncIterator[str]:
    """asyncio.wait_for wraps a single awaitable, not an async
    iterator — this drives one manually so each individual chunk gets
    the same timeout protection a single upstream call does in the
    non-streaming path. Without this, a stalled stream would hang the
    connection open indefinitely instead of failing predictably."""
    while True:
        try:
            chunk = await asyncio.wait_for(aiter.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return
        yield chunk


async def _stream_and_cache(
    body: ChatCompletionRequest,
    llm_provider: UpstreamLLMProvider,
    cache_service: CacheService,
) -> AsyncIterator[str]:
    """Forwards each chunk to the caller the instant it arrives — never
    buffers waiting for the full response — while collecting every
    chunk so the complete stream can be written to the cache exactly
    once, *after* the client already has everything. The cache write
    happens after the final yield, deliberately: nothing about caching
    should be able to delay the response the current caller is actually
    waiting on."""
    settings = get_settings()
    collected_chunks: list[str] = []

    try:
        async for chunk in _iterate_with_timeout(
            llm_provider.stream(body.messages), settings.upstream_timeout_seconds
        ):
            collected_chunks.append(chunk)
            yield format_sse_chunk(
                ChatCompletionStreamChunk(
                    model=body.model,
                    choices=[ChatCompletionStreamChoice(delta=StreamDelta(content=chunk))],
                ).model_dump()
            )
    except asyncio.TimeoutError:
        # The HTTP response already started (status 200, headers sent)
        # the moment the first chunk went out — there's no changing
        # that to a 504 at this point. The best available signal is an
        # in-stream error event, then a clean stop.
        logger.warning(
            "Upstream stream timed out mid-response",
            extra={"extra_fields": {"provider": llm_provider.name}},
        )
        yield format_sse_chunk(
            {"error": {"type": "upstream_timeout", "detail": "Upstream stalled mid-stream"}}
        )
        yield SSE_DONE
        return  # deliberately skip the cache write below — see docstring

    yield format_sse_chunk(
        ChatCompletionStreamChunk(
            model=body.model,
            choices=[ChatCompletionStreamChoice(delta=StreamDelta(), finish_reason="stop")],
        ).model_dump()
    )
    yield SSE_DONE

    # A truncated response is worse than no cached response at all — a
    # future cache HIT would silently serve a cut-off answer. Only a
    # stream that reached this line, fully intact, gets cached.
    full_text = "".join(collected_chunks)
    await cache_service.set(body, full_text, response_chunks=collected_chunks)
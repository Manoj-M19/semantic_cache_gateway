"""
Schemas deliberately mirror the OpenAI chat-completions shape
(`messages: [{role, content}]` in, `choices[].message.content` out).
"""

from typing import Literal

from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mock-llm"
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = False

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    model: str
    choices: list[ChatCompletionChoice]
    cache_status: Literal["hit", "miss", "coalesced"]
    similarity_score: float | None = None
    latency_ms: float

class InvalidateRequest(BaseModel):
    text: str = Field(..., min_length=1)

class InvalidateResponse(BaseModel):
    invalidated: bool

class ClearCacheResponse(BaseModel):
    deleted_count: int

class PurgeExpiredResponse(BaseModel):
    purged_count: int

class StreamDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None

class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: StreamDelta
    finish_reason: str | None = None

class ChatCompletionStreamChunk(BaseModel):
    model: str
    choices: list[ChatCompletionStreamChoice]

class ErrorResponse(BaseModel):
    error_type: str
    detail: str
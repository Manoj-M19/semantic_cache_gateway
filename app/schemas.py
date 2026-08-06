"""
Schemas deliberately mirror the OpenAI chat-completions shape
(`messages: [{role, content}]` in, `choices[].message.content` out).

That's not cosmetic — it means SemCache is a drop-in proxy in front of
*any* OpenAI-compatible endpoint (OpenAI itself, Together, Groq, a local
vLLM server), which is the same "vendor-agnostic gateway" idea from
InferGate applied to caching instead of routing.
"""
from typing import Literal

from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: Literal["system","user","assistent"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mock-llm"
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = False

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = 'stop'

class ChatCompletionResponse(BaseModel):
    model: str
    choices: list[ChatCompletionChoice]
    cache_status: literal["hit", "miss"]
    similarity_score: float | None = None
    latency_ms: float

class ErrorResponse(BaseModel):
    """Every error path in this gateway — validation, timeout, or an
    unhandled internal failure — returns this same shape. A caller
    should never need three different parsers for three different kinds
    of failure."""

    error_type: str
    detail: str
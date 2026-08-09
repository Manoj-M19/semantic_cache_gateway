"""
The upstream LLM provider interface.

Same idea as InferGate's ModelRunner: the gateway's routing, auth, and
(in later phases) caching logic never know or care which LLM actually
answered the request. Swapping OpenAI for Anthropic for a local vLLM
server means writing one new class here — nothing else in the codebase
changes.
"""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.schemas import ChatMessage

class UpstreamLLMProvider(ABC):
    name: str

    @abstractmethod
    async def complete(self, message: list[ChatMessage]) -> str:
         """Non-streaming completion: return the full response text."""
         raise NotImplementedError

    @abstractmethod
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
         """Streaming completion: yield response text chunk by chunk."""
         raise NotImplementedError
         yield
"""
A deterministic mock LLM — deliberately NOT a toy. It has real, measurable
latency (2s by default, matching realistic LLM response time) so caching's
benefit is actually observable in tests/demos, and it counts every call it
receives (`call_count`), which is how later phases *prove* the cache and
the coalescing logic are working: if you fire 50 identical requests and
`call_count` only goes up by 1, the architecture is doing its job.
"""

import asyncio
import hashlib
from collections.abc import AsyncIterator

from app.llm.base import UpstreamLLMProvider
from app.schemas import ChatMessage
from app.utils import extract_last_user_message


class MockLLMProvider(UpstreamLLMProvider):
    name = "mock-llm"

    def __init__(self, simulated_latency_seconds: float = 2.0):
        self.simulated_latency_seconds = simulated_latency_seconds
        self.call_count = 0

    def _deterministic_reply(self, messages: list[ChatMessage]) -> str:
        last_user_msg = extract_last_user_message(messages)
        digest = hashlib.sha256(last_user_msg.encode()).hexdigest()[:8]
        return (
            f"[mock-llm response id={digest}] Here's a considered answer "
            f'to: "{last_user_msg[:80]}"'
        )

    async def complete(self, messages: list[ChatMessage]) -> str:
        self.call_count += 1
        await asyncio.sleep(self.simulated_latency_seconds)
        return self._deterministic_reply(messages)

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.call_count += 1
        full_reply = self._deterministic_reply(messages)
        for word in full_reply.split(" "):
            await asyncio.sleep(self.simulated_latency_seconds / 20)
            yield word + " "
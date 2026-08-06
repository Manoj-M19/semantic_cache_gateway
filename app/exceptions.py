"""
Gateway-specific exceptions. Kept separate from generic Python exceptions
so the handlers in main.py can distinguish "upstream took too long"
(recoverable, the caller should retry) from "something inside the
gateway broke" (a bug — log full detail server-side, never leak it).
"""
class UpstreamTimeoutError(Exception):
    """Raised when a call to the upstream LLM provider exceeds
    settings.upstream_timeout_seconds. Maps to HTTP 504."""

    def __init__(self, provider_name: str, timeout_seconds: float):
        self.provider_name = provider_name
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Upstream provider '{provider_name}' exceeded {timeout_seconds}s timeout")
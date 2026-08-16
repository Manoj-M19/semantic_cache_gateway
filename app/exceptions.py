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

class CoalesceTimeoutError(Exception):
    """Raised when a follower gives up waiting for the leader handling
    the same request to produce a result — either the leader is
    genuinely still slow past max_wait_seconds, or it crashed without
    ever writing a result. Maps to HTTP 503: a contention/availability
    problem, distinct from an upstream timeout."""

    def __init__(self, key: str, waited_seconds: float):
        self.key = key
        self.waited_seconds = waited_seconds
        super().__init__(
            f"Timed out after {waited_seconds:.2f}s waiting for in-flight request '{key}' to complete"
        )
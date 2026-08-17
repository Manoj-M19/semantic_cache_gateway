import json
from typing import Any

SSE_DONE = "data: [DONE]\n\n"


def format_sse_chunk(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
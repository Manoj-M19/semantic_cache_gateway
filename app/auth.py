from fastapi import Header, HTTPException, status

from app.config import get_settings

async def verify_api_key(x_api_key: str | None = Header(default=None, alias='X-API-Key')) -> str:
    settings = get_settings()
    if not x_api_key or x_api_key not in settings.api_keys_set:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key
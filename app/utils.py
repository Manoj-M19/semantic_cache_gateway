from app.schemas import ChatMessage


def extract_last_user_message(messages: list[ChatMessage]) -> str:
    return next((m.content for m in reversed(messages) if m.role == "user"), "")
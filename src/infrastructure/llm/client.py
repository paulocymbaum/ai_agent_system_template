from src.config.settings import settings


def create_llm_client():
    """Factory function to create a Groq LLM client.

    Import `ChatGroq` lazily to avoid requiring the dependency at module
    import time (useful for running tests without installing the provider).
    """
    try:
        from langchain_groq import ChatGroq
    except Exception:  # pragma: no cover - optional provider
        return None

    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.GROQ_MODEL,
        temperature=0.7,
        max_tokens=2048,
    )

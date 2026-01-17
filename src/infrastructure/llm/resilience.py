from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from src.config.settings import settings


def create_retry_decorator():
    """Create a retry decorator for LLM calls."""

    return retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1, min=settings.RETRY_MIN_WAIT, max=settings.RETRY_MAX_WAIT
        ),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )


# Decorator instance
robust_llm_call = create_retry_decorator()

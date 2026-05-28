"""Retry and rate-limit handling for Azure OpenAI calls."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import structlog
from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Tenacity retry decorator pre-configured for Azure OpenAI
_RETRY_DECORATOR = retry(
    retry=retry_if_exception_type(
        (RateLimitError, APITimeoutError, APIConnectionError)
    ),
    wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
    stop=stop_after_attempt(5),
    before_sleep=lambda retry_state: logger.warning(
        "llm.retry",
        attempt=retry_state.attempt_number,
        exception=str(retry_state.outcome.exception()) if retry_state.outcome else "",
    ),
)


def with_retry(fn: F) -> F:
    """Decorate an LLM-calling method with exponential-backoff retry."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _RETRY_DECORATOR(fn)(*args, **kwargs)

    return wrapper  # type: ignore[return-value]

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 5.0


@dataclass(slots=True, frozen=True)
class RetryPolicy:
    """Configures retry behavior for external source requests."""

    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE

    def execute(
        self,
        operation: Callable[[], T],
        is_empty: Callable[[T], bool],
        context: str = "request",
    ) -> T:
        """Execute an operation with retry on empty results.

        Args:
            operation: Callable that returns the result.
            is_empty: Predicate that returns True if the result should trigger a retry.
            context: Human-readable label for log messages.

        Returns:
            The result of the operation (may be empty if all retries exhausted).
        """
        for attempt in range(self.max_retries + 1):
            result = operation()
            if not is_empty(result):
                return result
            if attempt < self.max_retries:
                delay = self.backoff_base * (attempt + 1)
                logger.warning(
                    "%s returned empty (attempt %d/%d), retrying in %.0fs",
                    context,
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                )
                time.sleep(delay)
        return result  # type: ignore[possibly-undefined]

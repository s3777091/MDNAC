from __future__ import annotations

from http.client import IncompleteRead
import logging
import time
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from libs.data.contracts import HttpTransport

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({403, 429, 500, 502, 503, 504})


class UrllibHttpTransport(HttpTransport):
    def __init__(
        self,
        timeout_seconds: float = 30.0,
        user_agent: str = "MicrobialDNACompiler/0.2",
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._max_retries = max_retries
        self._backoff_base = backoff_base

    def get_text(
        self,
        url: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> str:
        query_string = ""
        if params:
            query_string = urlencode(params, doseq=True)

        target_url = url if not query_string else f"{url}?{query_string}"
        request_headers = {"User-Agent": self._user_agent}
        if headers:
            request_headers.update(headers)

        request = Request(target_url, headers=request_headers, method="GET")

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                with urlopen(request, timeout=self._timeout_seconds) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    return response.read().decode(charset)
            except HTTPError as exc:
                last_error = exc
                if exc.code not in _RETRYABLE_STATUS_CODES or attempt == self._max_retries:
                    raise
                delay = self._backoff_base ** attempt
                logger.warning("HTTP %s from %s — retrying in %.1fs (attempt %d/%d)", exc.code, url, delay, attempt + 1, self._max_retries)
                time.sleep(delay)
            except (IncompleteRead, URLError, TimeoutError, ConnectionResetError, OSError) as exc:
                last_error = exc
                if attempt == self._max_retries:
                    raise
                delay = self._backoff_base ** attempt
                logger.warning("%s from %s — retrying in %.1fs (attempt %d/%d)", exc, url, delay, attempt + 1, self._max_retries)
                time.sleep(delay)

        raise last_error  # type: ignore[misc]

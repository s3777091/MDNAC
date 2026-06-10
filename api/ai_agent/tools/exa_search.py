from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from ai_agent.config.settings import ExaSettings


class ExaSearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExaSearchResult:
    title: str
    url: str
    published_date: str | None = None
    text: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "title": self.title,
            "url": self.url,
            "published_date": self.published_date,
            "text": self.text,
        }


class ExaSearchTool:
    def __init__(
        self,
        settings: ExaSettings,
        *,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory or _create_client

    def search(self, query: str) -> list[ExaSearchResult]:
        api_key = os.environ.get(self._settings.api_key_env)
        if not api_key:
            raise ExaSearchError(
                f"Missing Exa API key. Set environment variable "
                f"`{self._settings.api_key_env}`."
            )
        client = self._client_factory(api_key)
        try:
            response = client.search_and_contents(
                query,
                type=self._settings.search_type,
                num_results=self._settings.max_results,
                text=True,
            )
        except AttributeError:
            response = client.search(
                query,
                type=self._settings.search_type,
                num_results=self._settings.max_results,
            )
        except Exception as exc:
            raise ExaSearchError(f"Exa search failed: {exc}") from exc
        return normalize_exa_results(response)


def normalize_exa_results(response: Any) -> list[ExaSearchResult]:
    raw_results = _get_value(response, "results") or response
    if not isinstance(raw_results, list):
        return []
    normalized: list[ExaSearchResult] = []
    for item in raw_results:
        url = _clean_optional(_get_value(item, "url"))
        if not url:
            continue
        normalized.append(
            ExaSearchResult(
                title=_clean_optional(_get_value(item, "title")) or url,
                url=url,
                published_date=_clean_optional(
                    _get_value(item, "published_date")
                    or _get_value(item, "publishedDate")
                    or _get_value(item, "published")
                ),
                text=_clean_optional(_get_value(item, "text")),
            )
        )
    return normalized


def _create_client(api_key: str) -> Any:
    try:
        from exa_py import Exa
    except ImportError as exc:
        raise RuntimeError(
            "Exa search requires the `exa-py` package. Install the ai agent extra dependencies."
        ) from exc
    return Exa(api_key)


def _get_value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None

from __future__ import annotations

import json
import re
from typing import Any

from ai_agent.config.settings import AISettings
from ai_agent.skills import render_selected_skills


URL_PATTERN = re.compile(r"https?://[^\s\])}>\"']+")


def build_answer_messages(settings: AISettings, state: dict[str, Any]) -> list[dict[str, str]]:
    search_results = state.get("search_results") or []
    search_error = state.get("search_error")
    selected_skills = state.get("selected_skills") or []
    skills_text = render_selected_skills(selected_skills)
    evidence = {
        "context": state.get("context") or "",
        "search_results": search_results,
        "search_error": search_error,
    }
    user_content = (
        "Task:\n"
        f"{state.get('user_input') or ''}\n\n"
        "Selected Skills:\n"
        f"{skills_text or 'No additional skills selected.'}\n\n"
        "Evidence JSON:\n"
        f"{json.dumps(evidence, ensure_ascii=True, indent=2)}\n\n"
        "Return a direct answer. Include citations only when their URLs are present in "
        "`search_results`."
    )
    return [
        {"role": "system", "content": settings.system_prompt},
        {"role": "user", "content": user_content},
    ]


def allowed_citations(search_results: list[dict[str, Any]]) -> list[str]:
    citations: list[str] = []
    for result in search_results:
        url = str(result.get("url") or "").strip()
        if url and url not in citations:
            citations.append(url)
    return citations


def sanitize_answer_urls(answer: str, *, allowed_urls: list[str]) -> str:
    allowed = set(allowed_urls)

    def replace(match: re.Match[str]) -> str:
        url = match.group(0)
        if url in allowed:
            return url
        return "[unsupported URL removed]"

    return URL_PATTERN.sub(replace, answer)

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ai_agent.agent.prompts import (
    allowed_citations,
    build_answer_messages,
    sanitize_answer_urls,
)
from ai_agent.config.settings import AISettings
from ai_agent.models.base import ChatModel
from ai_agent.skills import select_agent_skills
from ai_agent.tools.exa_search import ExaSearchTool


SEARCH_KEYWORDS = (
    "current",
    "latest",
    "recent",
    "today",
    "news",
    "search",
    "source",
    "sources",
    "cite",
    "citation",
    "verify",
)


def plan_node(state: dict[str, Any], *, settings: AISettings) -> dict[str, Any]:
    user_input = (state.get("user_input") or "").lower()
    context = (state.get("context") or "").strip()
    wants_fresh_evidence = any(keyword in user_input for keyword in SEARCH_KEYWORDS)
    search_needed = bool(wants_fresh_evidence or not context)
    if len(state.get("tool_calls") or []) >= settings.agent.max_tool_calls:
        search_needed = False
    selected_skills = select_agent_skills(
        user_input=state.get("user_input") or "",
        context=context,
        search_needed=search_needed,
    )
    tool_calls = list(state.get("tool_calls") or [])
    tool_calls.append(
        {
            "tool": "agent_skills",
            "status": "completed",
            "selected": [skill["name"] for skill in selected_skills],
        }
    )
    return {
        **state,
        "search_needed": search_needed,
        "selected_skills": selected_skills,
        "tool_calls": tool_calls,
    }


def exa_search_node(
    state: dict[str, Any],
    *,
    settings: AISettings,
    search_tool: ExaSearchTool,
) -> dict[str, Any]:
    del settings
    tool_calls = list(state.get("tool_calls") or [])
    call_record: dict[str, Any] = {
        "tool": "exa_search",
        "query": state.get("user_input") or "",
        "status": "completed",
    }
    try:
        results = search_tool.search(state.get("user_input") or "")
        normalized = [result.to_dict() for result in results]
        call_record["result_count"] = len(normalized)
        return {
            **state,
            "search_results": normalized,
            "tool_calls": [*tool_calls, call_record],
            "search_error": None,
        }
    except Exception as exc:
        call_record["status"] = "failed"
        call_record["error"] = str(exc)
        return {
            **state,
            "search_results": [],
            "tool_calls": [*tool_calls, call_record],
            "search_error": str(exc),
        }


def answer_node(
    state: dict[str, Any],
    *,
    settings: AISettings,
    model: ChatModel,
) -> dict[str, Any]:
    messages = build_answer_messages(settings, state)
    try:
        raw_answer = model.generate(
            messages,
            temperature=0.0,
        )
    except Exception as exc:
        return {**state, "messages": messages, "error": f"Model generation failed: {exc}"}

    citations = allowed_citations(state.get("search_results") or [])
    answer = sanitize_answer_urls(str(raw_answer).strip(), allowed_urls=citations)
    if state.get("search_error"):
        answer = (
            f"Search failed: {state['search_error']}\n\n"
            "Answering from provided context only.\n\n"
            f"{answer}"
        )
    return {
        **state,
        "messages": messages,
        "draft_answer": answer,
        "final_answer": answer,
        "citations": citations,
        "error": state.get("error"),
    }


def human_review_node(state: dict[str, Any], *, settings: AISettings) -> dict[str, Any]:
    low_confidence = not (state.get("context") or state.get("search_results"))
    needs_approval = bool(settings.agent.require_human_approval or low_confidence)
    return {
        **state,
        "needs_human_approval": needs_approval,
        "approval_id": uuid4().hex if needs_approval else None,
    }

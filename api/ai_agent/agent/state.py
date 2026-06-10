from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    context: str
    messages: list[dict[str, str]]
    search_results: list[dict[str, Any]]
    search_needed: bool
    search_error: str | None
    selected_skills: list[dict[str, str]]
    tool_calls: list[dict[str, Any]]
    draft_answer: str
    final_answer: str
    citations: list[str]
    needs_human_approval: bool
    approval_id: str | None
    error: str | None

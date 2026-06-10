from __future__ import annotations

from typing import Any

from ai_agent.agent.human_in_loop import InMemoryApprovalStore
from ai_agent.agent.nodes import (
    answer_node,
    exa_search_node,
    human_review_node,
    plan_node,
)
from ai_agent.config.settings import AISettings
from ai_agent.models.base import ChatModel
from ai_agent.schemas import AgentRunResponse
from ai_agent.tools.exa_search import ExaSearchTool


class _SequentialAgentGraph:
    def __init__(
        self,
        *,
        settings: AISettings,
        model: ChatModel,
        search_tool: ExaSearchTool,
    ) -> None:
        self._settings = settings
        self._model = model
        self._search_tool = search_tool

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        state = plan_node(state, settings=self._settings)
        if state.get("search_needed"):
            state = exa_search_node(
                state,
                settings=self._settings,
                search_tool=self._search_tool,
            )
        state = answer_node(state, settings=self._settings, model=self._model)
        return human_review_node(state, settings=self._settings)


def build_agent_graph(
    *,
    settings: AISettings,
    model: ChatModel,
    search_tool: ExaSearchTool,
) -> Any:
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return _SequentialAgentGraph(
            settings=settings,
            model=model,
            search_tool=search_tool,
        )

    def route_after_plan(state: dict[str, Any]) -> str:
        return "search" if state.get("search_needed") else "answer"

    graph = StateGraph(dict)
    graph.add_node("plan", lambda state: plan_node(state, settings=settings))
    graph.add_node(
        "exa_search",
        lambda state: exa_search_node(
            state,
            settings=settings,
            search_tool=search_tool,
        ),
    )
    graph.add_node("answer", lambda state: answer_node(state, settings=settings, model=model))
    graph.add_node("human_review", lambda state: human_review_node(state, settings=settings))
    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"search": "exa_search", "answer": "answer"},
    )
    graph.add_edge("exa_search", "answer")
    graph.add_edge("answer", "human_review")
    graph.add_edge("human_review", END)
    return graph.compile()


def run_agent(
    *,
    settings: AISettings,
    model: ChatModel,
    search_tool: ExaSearchTool,
    approval_store: InMemoryApprovalStore,
    user_input: str,
    context: str,
) -> AgentRunResponse:
    initial_state: dict[str, Any] = {
        "user_input": user_input,
        "context": context,
        "messages": [],
        "search_results": [],
        "selected_skills": [],
        "tool_calls": [],
        "draft_answer": "",
        "final_answer": "",
        "citations": [],
        "needs_human_approval": False,
        "approval_id": None,
        "error": None,
    }
    graph = build_agent_graph(settings=settings, model=model, search_tool=search_tool)
    final_state = graph.invoke(initial_state)

    if final_state.get("error") and not final_state.get("draft_answer"):
        return AgentRunResponse(
            status="failed",
            answer="",
            citations=[],
            tool_calls=final_state.get("tool_calls") or [],
            needs_approval=False,
            approval_id=None,
            error=final_state.get("error"),
        )

    if final_state.get("needs_human_approval"):
        pending = approval_store.save(
            approval_id=final_state.get("approval_id"),
            draft_answer=final_state.get("draft_answer") or "",
            citations=final_state.get("citations") or [],
            tool_calls=final_state.get("tool_calls") or [],
            user_input=user_input,
            context=context,
        )
        return AgentRunResponse(
            status="waiting_for_human",
            answer=pending.draft_answer,
            citations=pending.citations,
            tool_calls=pending.tool_calls,
            needs_approval=True,
            approval_id=pending.approval_id,
            error=final_state.get("error"),
        )

    return AgentRunResponse(
        status="completed",
        answer=final_state.get("final_answer") or final_state.get("draft_answer") or "",
        citations=final_state.get("citations") or [],
        tool_calls=final_state.get("tool_calls") or [],
        needs_approval=False,
        approval_id=None,
        error=final_state.get("error"),
    )

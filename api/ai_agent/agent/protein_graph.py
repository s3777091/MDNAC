"""A LangGraph agent that turns a vague biological goal into a protein
span-completion prompt, with a real human-in-the-loop clarification step.

Flow (matches the product goal):

    user goal (vague)
        -> public_research   : Exa tool, understand the domain
        -> clarify           : propose a sharper query; interrupt() and ask the
                               human when the goal is too vague; loop on "revise"
        -> build_protein_span: fetch protein records (NCBI/ENA), rank them with
                               the local semantic search (accurate, not guessing),
                               then cut the correct mask span and build the
                               `instruction` / `input` for span completion.

The Exa tool is "outside" knowledge to refine the goal with the human; the
protein semantic search is the "inside" tool that selects the real protein and
the exact span to fill. Raw protein records are only handled inside a single
node so they never need to be serialized into the checkpointer across the
interrupt -- only JSON-safe state is persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypedDict

from ai_agent.agent.clarify import clarify_protein_request
from ai_agent.config.settings import AISettings
from ai_agent.models.base import ChatModel
from ai_agent.tools.exa_search import ExaSearchTool
from ai_agent.tools.protein_semantic_search import (
    choose_semantic_mask_span,
    rank_protein_records,
)


DEFAULT_MAX_CLARIFY_ROUNDS = 3

DEFAULT_PARAMS: dict[str, Any] = {
    "source": "ncbi",
    "limit": 5,
    "semantic_top_k": 3,
    "mask_policy": "random_span",
    "mask_start": None,
    "mask_length": 48,
    "left_flank_size": 64,
    "right_flank_size": 64,
    "require_clarification": True,
    "research_with_exa": True,
    "max_clarify_rounds": DEFAULT_MAX_CLARIFY_ROUNDS,
}


class ProteinSpanState(TypedDict, total=False):
    user_input: str
    context: str
    params: dict[str, Any]
    evidence: list[dict[str, Any]]
    search_error: str | None
    clarify_rounds: int
    needs_clarification: bool
    clarification: dict[str, Any]
    refined_query: str
    clarify_done: bool
    matches: list[dict[str, Any]]
    selected: dict[str, Any] | None
    span: dict[str, Any] | None
    instruction: str
    input: str
    tool_calls: list[dict[str, Any]]
    status: str
    error: str | None
    note: str


# Record fetcher: (query, source, limit) -> list of protein record objects, where
# each record exposes `.sequence`, `.accession`, `.description`, `.organism`,
# `.metadata` (the shape produced by libs.data sequence sources).
FetchRecords = Callable[[str, str, int], list[Any]]
BuildSourceRow = Callable[[Any, str], dict[str, Any]]


@dataclass
class ProteinSpanDeps:
    """External collaborators injected by the server (keeps the graph testable)."""

    model: ChatModel
    settings: AISettings
    search_tool: ExaSearchTool
    fetch_records: FetchRecords
    build_source_row: BuildSourceRow
    make_span_example: Callable[..., dict[str, Any]]


def initial_protein_span_state(
    *,
    user_input: str,
    context: str = "",
    params: dict[str, Any] | None = None,
) -> ProteinSpanState:
    merged = {**DEFAULT_PARAMS, **(params or {})}
    return ProteinSpanState(
        user_input=user_input,
        context=context,
        params=merged,
        evidence=[],
        search_error=None,
        clarify_rounds=0,
        needs_clarification=False,
        refined_query="",
        clarify_done=False,
        matches=[],
        selected=None,
        span=None,
        instruction="",
        input="",
        tool_calls=[],
        status="running",
        error=None,
    )


def _append_tool(state: ProteinSpanState, entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [*(state.get("tool_calls") or []), entry]


def _public_research_node(state: ProteinSpanState, deps: ProteinSpanDeps) -> dict[str, Any]:
    """Exa tool: gather public background evidence about the (raw) goal."""
    params = state.get("params") or {}
    if not params.get("research_with_exa", True):
        return {
            "evidence": [],
            "search_error": None,
            "tool_calls": _append_tool(state, {"tool": "exa_search", "status": "skipped"}),
        }
    try:
        results = deps.search_tool.search(state.get("user_input") or "")
        evidence = [result.to_dict() for result in results]
        return {
            "evidence": evidence,
            "search_error": None,
            "tool_calls": _append_tool(
                state,
                {"tool": "exa_search", "status": "completed", "result_count": len(evidence)},
            ),
        }
    except Exception as exc:  # network/key errors must not abort the workflow
        return {
            "evidence": [],
            "search_error": str(exc),
            "tool_calls": _append_tool(
                state, {"tool": "exa_search", "status": "failed", "error": str(exc)}
            ),
        }


def _clarify_node(state: ProteinSpanState, deps: ProteinSpanDeps) -> dict[str, Any]:
    """Human-in-the-loop: refine the goal, pausing for the human when it is vague."""
    from langgraph.types import interrupt

    params = state.get("params") or {}
    rounds = int(state.get("clarify_rounds") or 0)
    user_input = state.get("user_input") or ""

    if not params.get("require_clarification", True):
        return {"needs_clarification": False, "refined_query": user_input, "clarify_done": True}

    clarification = clarify_protein_request(
        deps.model,
        deps.settings,
        user_input,
        state.get("context") or "",
        state.get("evidence") or [],
    )
    proposed = str(clarification.get("proposed_query") or user_input)

    if not clarification.get("needs_clarification"):
        return {
            "needs_clarification": False,
            "refined_query": _pick_search_query(user_input, proposed, rounds),
            "clarification": clarification,
            "clarify_done": True,
        }

    # Goal is vague. Stop runaway loops by accepting the proposal after the cap.
    if rounds >= int(params.get("max_clarify_rounds", DEFAULT_MAX_CLARIFY_ROUNDS)):
        return {
            "needs_clarification": False,
            "refined_query": _pick_search_query(user_input, proposed, rounds),
            "clarification": clarification,
            "clarify_done": True,
            "note": "max clarification rounds reached; used proposed query",
        }

    decision = interrupt(
        {
            "type": "clarification_request",
            "message": clarification.get("message"),
            "proposed_query": proposed,
            "user_input": user_input,
            "evidence": state.get("evidence") or [],
            "expected_actions": ["approve", "revise", "cancel"],
        }
    )

    decision = decision or {}
    action = str(decision.get("action") or "").strip().lower()
    if action in {"approve", "ok", "continue"}:
        return {
            "needs_clarification": False,
            "refined_query": _pick_search_query(user_input, proposed, rounds),
            "clarification": clarification,
            "clarify_done": True,
        }
    if action in {"revise", "edit"}:
        revised = str(decision.get("user_input") or "").strip()
        if not revised:
            return {
                "clarify_done": True,
                "status": "failed",
                "error": "revise action requires a non-empty user_input.",
            }
        # Loop back into clarify with the revised goal.
        return {
            "user_input": revised,
            "clarify_rounds": rounds + 1,
            "needs_clarification": True,
            "clarify_done": False,
        }
    if action == "cancel":
        return {
            "clarify_done": True,
            "status": "cancelled",
            "error": "Workflow cancelled by user.",
        }
    return {
        "clarify_done": True,
        "status": "failed",
        "error": "Unknown clarification action. Use approve, revise, or cancel.",
    }


def _build_protein_span_node(state: ProteinSpanState, deps: ProteinSpanDeps) -> dict[str, Any]:
    """Inside tool + span cutting: fetch -> semantic rank -> choose span -> build prompt."""
    params = state.get("params") or {}
    query = state.get("refined_query") or state.get("user_input") or ""

    # 1) Fetch real protein records (NCBI/ENA) for the refined goal.
    try:
        records = deps.fetch_records(
            query, str(params.get("source", "ncbi")), int(params.get("limit", 5))
        )
    except Exception as exc:
        return {"status": "failed", "error": f"Protein fetch failed: {exc}"}
    records = [record for record in records if _compact(getattr(record, "sequence", ""))]
    if not records:
        return {"status": "failed", "error": "No fetched protein record contains a sequence."}

    # 2) Semantic ranking over real records -- accurate selection, not a guess.
    evidence = state.get("evidence") or []
    matches = rank_protein_records(
        query,
        records,
        evidence_texts=[item.get("text") or item.get("title") or "" for item in evidence],
        min_length=int(params.get("mask_length", 48)),
        top_k=int(params.get("semantic_top_k", 3)),
    )
    if not matches:
        mask_len = int(params.get("mask_length", 48))
        longest = max(
            (len(_compact(getattr(record, "sequence", ""))) for record in records),
            default=0,
        )
        return {
            "status": "failed",
            "error": (
                f"None of the {len(records)} fetched protein record(s) is long enough for a "
                f"{mask_len}-residue mask span (longest fetched sequence is {longest} aa). "
                f"Lower mask_length to {longest} or less, or refine the query toward longer proteins."
            ),
        }
    top_match = matches[0]

    # 3) Cut the correct span to fill (standard amino acids, useful flanks).
    try:
        span_choice = choose_semantic_mask_span(
            top_match.record.sequence,
            mask_length=int(params.get("mask_length", 48)),
            requested_start=params.get("mask_start"),
            left_flank_size=int(params.get("left_flank_size", 64)),
            right_flank_size=int(params.get("right_flank_size", 64)),
        )
    except Exception as exc:
        return {"status": "failed", "error": f"Span selection failed: {exc}"}

    # 4) Build the span-completion prompt (never reveals the hidden output span).
    try:
        source_row = deps.build_source_row(top_match.record, query)
        span_row = deps.make_span_example(
            source_row,
            source_index=0,
            mask_start=span_choice.start,
            mask_end=span_choice.end,
            mask_policy=str(params.get("mask_policy", "random_span")),
            left_flank_size=int(params.get("left_flank_size", 64)),
            right_flank_size=int(params.get("right_flank_size", 64)),
        )
    except Exception as exc:
        return {"status": "failed", "error": f"Span example build failed: {exc}"}

    return {
        "refined_query": query,
        "matches": [match.to_dict() for match in matches],
        "selected": top_match.to_dict(),
        "span": span_choice.to_dict(),
        "instruction": span_row["instruction"],
        "input": span_row["input"],
        "status": "completed",
        "tool_calls": _append_tool(
            state,
            {
                "tool": "protein_semantic_search",
                "status": "completed",
                "selected": top_match.to_dict().get("accession"),
            },
        ),
    }


def _strip_placeholders(text: str) -> str:
    import re

    cleaned = re.sub(r"\[[^\]]*\]", " ", text or "")
    return " ".join(cleaned.split())


def _pick_search_query(user_input: str, proposed: str, rounds: int) -> str:
    """Choose the protein-database search query.

    After a human revise (rounds > 0) the user's own concise wording is the
    authoritative search query; otherwise use the model's proposed keyword query.
    Either way, strip bracketed placeholders so NCBI esearch gets clean terms.
    """
    chosen = user_input if rounds > 0 else proposed
    return _strip_placeholders(chosen) or _strip_placeholders(proposed) or user_input


def _route_after_clarify(state: ProteinSpanState) -> str:
    if state.get("status") in {"failed", "cancelled"} or state.get("error"):
        return "end"
    if not state.get("clarify_done"):
        return "clarify"  # revise loop -> re-run clarification on the new goal
    if state.get("refined_query"):
        return "build"
    return "end"


def build_protein_span_graph(deps: ProteinSpanDeps, *, checkpointer: Any | None = None) -> Any:
    """Compile the LangGraph agent. A checkpointer is required for interrupt()."""
    from langgraph.graph import END, START, StateGraph

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    graph = StateGraph(ProteinSpanState)
    graph.add_node("public_research", lambda state: _public_research_node(state, deps))
    graph.add_node("clarify", lambda state: _clarify_node(state, deps))
    graph.add_node("build_protein_span", lambda state: _build_protein_span_node(state, deps))
    graph.add_edge(START, "public_research")
    graph.add_edge("public_research", "clarify")
    graph.add_conditional_edges(
        "clarify",
        _route_after_clarify,
        {"clarify": "clarify", "build": "build_protein_span", "end": END},
    )
    graph.add_edge("build_protein_span", END)
    return graph.compile(checkpointer=checkpointer)


def interpret_result(result: dict[str, Any]) -> dict[str, Any]:
    """Translate a graph invoke() result into an API-friendly response.

    Detects the interrupt that the clarify node raises and surfaces the human
    prompt; otherwise returns the completed/failed/cancelled outcome.
    """
    interrupts = result.get("__interrupt__")
    public = {key: value for key, value in result.items() if not key.startswith("__")}
    if interrupts:
        first = interrupts[0]
        payload = getattr(first, "value", first)
        return {"status": "waiting_for_human", "interrupt": payload, "state": public}
    return {"status": public.get("status") or "completed", "state": public}


def _compact(value: Any) -> str:
    return "".join(str(value or "").split()).upper()


__all__ = [
    "DEFAULT_PARAMS",
    "ProteinSpanDeps",
    "ProteinSpanState",
    "build_protein_span_graph",
    "initial_protein_span_state",
    "interpret_result",
]

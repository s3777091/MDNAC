from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Literal

from ai_agent.agent.graph import run_agent
from ai_agent.agent.human_in_loop import InMemoryApprovalStore
from ai_agent.config.settings import (
    AGENT_CONFIG_PATH_ENV_VAR,
    AGENT_ENVIRONMENT_ENV_VAR,
    AISettings,
    load_settings,
)
from ai_agent.models.base import ChatModel
from ai_agent.models.openai_model import OpenAIChatModel
from ai_agent.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    ApprovalRequest,
    ProteinSpanResumeRequest,
    ProteinSpanStartRequest,
    RejectionRequest,
)
from ai_agent.agent.clarify import clarify_protein_request as _clarify_protein_request
from ai_agent.agent.protein_graph import (
    ProteinSpanDeps,
    build_protein_span_graph,
    initial_protein_span_state,
    interpret_result,
)
from ai_agent.skills import load_agent_skills
from ai_agent.tools.exa_search import ExaSearchTool
from ai_agent.tools.protein_semantic_search import (
    choose_semantic_mask_span,
    rank_protein_records,
)
from interfere.config import (
    CONFIG_PATH_ENV_VAR,
    ENVIRONMENT_ENV_VAR,
    load_config,
)
from interfere.inference import InferenceAPI
from interfere.server import (
    SPAN_COMPLETION_ROUTE,
    _build_source_query,
    _build_span_source_row,
    _merge_generation_request,
    _new_http_transport,
    _span_completion_data_dependencies,
)
from libs.protein_completion.masking import make_span_completion_example


PROTEIN_SPAN_COMPLETION_WS_ROUTE = "/protein-span-completion/ws"

_MODEL_CACHE: dict[str, ChatModel] = {}
_TOOL_CACHE: dict[str, ExaSearchTool] = {}
_INFERENCE_CACHE: dict[str, InferenceAPI] = {}


def create_app(
    *,
    config_path: str | Path | None = None,
    environment: str | None = None,
    protein_config_path: str | Path | None = None,
    agent_config_path: str | Path | None = None,
    protein_environment: str | None = None,
    agent_environment: str | None = None,
):
    from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel, Field

    globals()["WebSocket"] = WebSocket
    globals()["WebSocketDisconnect"] = WebSocketDisconnect

    class GenerateRequest(BaseModel):
        prompt: str = ""
        max_new_tokens: int | None = Field(default=None, ge=0)
        temperature: float | None = Field(default=None, ge=0.0)
        top_k: int | None = Field(default=None, gt=0)
        seed: int | None = None
        stop_at_endoftext: bool | None = None
        ensure_protein_prompt: bool | None = None

    class SpanCompletionRequest(BaseModel):
        raw_input: str = Field(..., min_length=1)
        source: Literal["ncbi", "ena"] = "ncbi"
        limit: int = Field(default=1, gt=0)
        mask_policy: str = "random_span"
        mask_start: int = Field(default=0, ge=0)
        mask_length: int = Field(default=48, gt=0)
        left_flank_size: int = Field(default=64, ge=0)
        right_flank_size: int = Field(default=64, ge=0)

    class ProteinSpanSocketRequest(BaseModel):
        user_input: str = Field(..., min_length=1)
        context: str | list[str] | dict[str, Any] | None = None
        source: Literal["ncbi", "ena", "auto"] = "ncbi"
        limit: int = Field(default=5, gt=0)
        semantic_top_k: int = Field(default=3, gt=0)
        mask_policy: str = "random_span"
        mask_start: int | None = Field(default=None, ge=0)
        mask_length: int = Field(default=48, gt=0)
        left_flank_size: int = Field(default=64, ge=0)
        right_flank_size: int = Field(default=64, ge=0)
        require_clarification: bool = True
        research_with_exa: bool = True

    protein_settings = load_config(
        config_path=protein_config_path or config_path,
        environment=protein_environment or environment,
    )
    agent_settings = load_settings(
        config_path=agent_config_path,
        environment=agent_environment or environment,
    )
    approval_store = InMemoryApprovalStore()

    app = FastAPI(title="MDNAC Unified API", version="0.2.0")
    app.state.protein_settings = protein_settings
    app.state.agent_settings = agent_settings
    app.state.approval_store = approval_store

    def get_inference_api() -> InferenceAPI:
        cache_key = f"{protein_settings.model.path}|{protein_settings.model.device}"
        if cache_key not in _INFERENCE_CACHE:
            _INFERENCE_CACHE[cache_key] = InferenceAPI.load(
                model_path=protein_settings.model.path,
                device_name=protein_settings.model.device,
            )
        return _INFERENCE_CACHE[cache_key]

    def get_model() -> ChatModel:
        cache_key = _model_cache_key(agent_settings)
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = _create_model(agent_settings)
        return _MODEL_CACHE[cache_key]

    def get_search_tool() -> ExaSearchTool:
        cache_key = (
            f"{agent_settings.exa.api_key_env}|{agent_settings.exa.search_type}|"
            f"{agent_settings.exa.max_results}"
        )
        if cache_key not in _TOOL_CACHE:
            _TOOL_CACHE[cache_key] = ExaSearchTool(agent_settings.exa)
        return _TOOL_CACHE[cache_key]

    protein_span_graph_cache: dict[str, Any] = {}

    def _fetch_protein_records(query: str, source: str, limit: int) -> list[Any]:
        if source not in {"ncbi", "ena", "auto"}:
            raise _HttpError(
                400,
                f"Unsupported protein source '{source}'. Use 'ncbi', 'ena', or 'auto'.",
            )
        return _fetch_sequence_records(query, source=source, limit=limit)

    def get_protein_span_graph() -> Any:
        # Built lazily so health/ready checks do not require the OpenAI key.
        # The compiled graph owns an in-process checkpointer that holds paused
        # (waiting-for-human) threads between /start and /resume calls.
        if "graph" not in protein_span_graph_cache:
            deps = ProteinSpanDeps(
                model=get_model(),
                settings=agent_settings,
                search_tool=get_search_tool(),
                fetch_records=_fetch_protein_records,
                build_source_row=lambda record, raw_input: _build_span_source_row(
                    record, raw_input=raw_input
                ),
                make_span_example=make_span_completion_example,
            )
            protein_span_graph_cache["graph"] = build_protein_span_graph(deps)
        return protein_span_graph_cache["graph"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "protein": {
                "environment": protein_settings.environment,
                "model_path": str(protein_settings.model.path),
                "device": protein_settings.model.device,
            },
            "agent": {
                "environment": agent_settings.environment,
                "provider": agent_settings.provider,
                "model": _configured_model_name(agent_settings),
            },
        }

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        readiness: dict[str, Any] = {"status": "ready", "protein": {}, "agent": {}}
        try:
            api = get_inference_api()
            readiness["protein"] = {
                "status": "ready",
                "model_path": str(api.session.onnx_path),
                "providers": list(api.session.providers),
            }
        except Exception as exc:
            readiness["status"] = "degraded"
            readiness["protein"] = {"status": "not_ready", "error": str(exc)}

        try:
            get_model().ensure_ready()
            readiness["agent"] = {
                "status": "ready",
                "provider": agent_settings.provider,
                "model": _configured_model_name(agent_settings),
            }
        except Exception as exc:
            readiness["status"] = "degraded"
            readiness["agent"] = {"status": "not_ready", "error": str(exc)}

        if readiness["status"] != "ready":
            raise HTTPException(status_code=503, detail=readiness)
        return readiness

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        payload = _model_dump(request)
        kwargs = _merge_generation_request(protein_settings, payload)
        try:
            result = get_inference_api().generate_protein(request.prompt, **kwargs)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    @app.post(SPAN_COMPLETION_ROUTE)
    def span_completion(request: SpanCompletionRequest = Body(...)) -> dict[str, str]:
        try:
            return _build_span_completion_prompt(
                raw_input=request.raw_input,
                source=request.source,
                limit=request.limit,
                mask_policy=request.mask_policy,
                mask_start=request.mask_start,
                mask_length=request.mask_length,
                left_flank_size=request.left_flank_size,
                right_flank_size=request.right_flank_size,
            )
        except _HttpError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/agent/run", response_model=AgentRunResponse)
    def run(request: AgentRunRequest) -> AgentRunResponse:
        try:
            return run_agent(
                settings=agent_settings,
                model=get_model(),
                search_tool=get_search_tool(),
                approval_store=approval_store,
                user_input=request.user_input,
                context=_normalize_context(request.context),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/agent/skills")
    def skills() -> dict[str, Any]:
        loaded = load_agent_skills()
        return {
            "skills": [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "path": str(skill.path),
                }
                for skill in loaded
            ]
        }

    @app.post("/agent/approve", response_model=AgentRunResponse)
    def approve(request: ApprovalRequest) -> AgentRunResponse:
        pending = approval_store.approve(request.approval_id)
        if pending is None:
            raise HTTPException(status_code=404, detail="Unknown approval_id.")
        return AgentRunResponse(
            status="completed",
            answer=pending.draft_answer,
            citations=pending.citations,
            tool_calls=pending.tool_calls,
            needs_approval=False,
            approval_id=None,
        )

    @app.post("/agent/reject", response_model=AgentRunResponse)
    def reject(request: RejectionRequest) -> AgentRunResponse:
        pending = approval_store.reject(request.approval_id)
        if pending is None:
            raise HTTPException(status_code=404, detail="Unknown approval_id.")
        revised = request.revised_instruction or "Submit a new /agent/run request."
        reason = f" Reason: {request.reason}" if request.reason else ""
        return AgentRunResponse(
            status="failed",
            answer=f"Draft rejected.{reason} Revised instruction: {revised}",
            citations=[],
            tool_calls=pending.tool_calls,
            needs_approval=False,
            approval_id=None,
        )

    @app.post("/agent/protein-span/start")
    def protein_span_start(request: ProteinSpanStartRequest) -> dict[str, Any]:
        """Start the protein-span LangGraph agent.

        Runs Exa research then clarification. Returns either
        ``status=waiting_for_human`` with the question + a thread_id to resume,
        or a completed result with ``instruction``/``input``.
        """
        from uuid import uuid4

        thread_id = uuid4().hex
        state = initial_protein_span_state(
            user_input=request.user_input,
            context=_normalize_context(request.context),
            params={
                "source": request.source,
                "limit": request.limit,
                "semantic_top_k": request.semantic_top_k,
                "mask_policy": request.mask_policy,
                "mask_start": request.mask_start,
                "mask_length": request.mask_length,
                "left_flank_size": request.left_flank_size,
                "right_flank_size": request.right_flank_size,
                "require_clarification": request.require_clarification,
                "research_with_exa": request.research_with_exa,
            },
        )
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = interpret_result(get_protein_span_graph().invoke(state, config))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"thread_id": thread_id, **result}

    @app.post("/agent/protein-span/resume")
    def protein_span_resume(request: ProteinSpanResumeRequest) -> dict[str, Any]:
        """Resume a paused protein-span agent thread with the human's decision."""
        from langgraph.types import Command

        decision: dict[str, Any] = {"action": request.action}
        if request.user_input is not None:
            decision["user_input"] = request.user_input
        config = {"configurable": {"thread_id": request.thread_id}}
        try:
            result = interpret_result(
                get_protein_span_graph().invoke(Command(resume=decision), config)
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"thread_id": request.thread_id, **result}

    @app.websocket(PROTEIN_SPAN_COMPLETION_WS_ROUTE)
    async def protein_span_completion_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            initial_payload = await websocket.receive_json()
            request = ProteinSpanSocketRequest(**initial_payload)
            await _send_event(
                websocket,
                "accepted",
                {
                    "message": "Protein span completion workflow accepted.",
                    "route": PROTEIN_SPAN_COMPLETION_WS_ROUTE,
                },
            )

            # Step 1: run Exa public research on the original (possibly vague)
            # user input first, so the clarifying question can propose suggestions
            # grounded in outside domain evidence.
            evidence = await _run_public_research(
                websocket=websocket,
                request=request,
                query=request.user_input,
                search_tool_factory=get_search_tool,
            )

            # Step 2: ask the human to confirm/refine the query, using the Exa
            # evidence to power the proposed suggestions.
            clarification = await _resolve_clarification(
                websocket=websocket,
                request=request,
                model_factory=get_model,
                settings=agent_settings,
                evidence=evidence,
            )
            refined_query = clarification["query"]
            request = clarification["request"]

            await _send_event(
                websocket,
                "fetch_started",
                {
                    "source": request.source,
                    "query": refined_query,
                    "limit": request.limit,
                },
            )
            records = await asyncio.to_thread(
                _fetch_protein_records,
                refined_query,
                request.source,
                request.limit,
            )
            await _send_event(
                websocket,
                "fetch_completed",
                {"record_count": len(records), "source": request.source},
            )

            await _send_event(
                websocket,
                "semantic_search_started",
                {
                    "message": "Ranking fetched protein records with local semantic search.",
                    "top_k": request.semantic_top_k,
                },
            )
            matches = rank_protein_records(
                refined_query,
                records,
                evidence_texts=[item.get("text") or item.get("title") or "" for item in evidence],
                min_length=request.mask_length,
                top_k=request.semantic_top_k,
            )
            if not matches:
                longest = max(
                    (len(_compact_protein_sequence(record.sequence)) for record in records),
                    default=0,
                )
                raise ValueError(
                    f"None of the {len(records)} fetched protein record(s) is long enough "
                    f"for a {request.mask_length}-residue mask span (longest fetched sequence "
                    f"is {longest} aa). Lower mask_length to {longest} or less, or refine the "
                    f"query toward longer proteins."
                )

            top_match = matches[0]
            await _send_event(
                websocket,
                "semantic_search_completed",
                {
                    "selected": top_match.to_dict(),
                    "matches": [match.to_dict() for match in matches],
                },
            )

            span_choice = choose_semantic_mask_span(
                top_match.record.sequence,
                mask_length=request.mask_length,
                requested_start=request.mask_start,
                left_flank_size=request.left_flank_size,
                right_flank_size=request.right_flank_size,
            )
            await _send_event(websocket, "span_selected", span_choice.to_dict())

            source_row = _build_span_source_row(top_match.record, raw_input=refined_query)
            span_row = make_span_completion_example(
                source_row,
                source_index=0,
                mask_start=span_choice.start,
                mask_end=span_choice.end,
                mask_policy=request.mask_policy,
                left_flank_size=request.left_flank_size,
                right_flank_size=request.right_flank_size,
            )

            await _send_event(
                websocket,
                "completed",
                {
                    "instruction": span_row["instruction"],
                    "input": span_row["input"],
                    "refined_query": refined_query,
                    "selected_record": top_match.to_dict(),
                    "semantic_matches": [match.to_dict() for match in matches],
                    "span": span_choice.to_dict(),
                    "public_research": evidence,
                },
            )
        except WebSocketDisconnect:
            return
        except _WorkflowCancelled:
            return
        except Exception as exc:
            await _send_event(websocket, "error", {"detail": str(exc)})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the unified MDNAC HTTP and WebSocket API.",
    )
    parser.add_argument("--config", default=None, help="Path to api/config.yaml.")
    parser.add_argument("--agent-config", default=None, help="Path to ai_agent/config/agent.yaml.")
    parser.add_argument("--env", default=None, help="Environment name shared by both configs.")
    parser.add_argument("--protein-env", default=None, help="Environment name from api/config.yaml.")
    parser.add_argument("--agent-env", default=None, help="Environment name from agent.yaml.")
    parser.add_argument("--host", default=None, help="Override configured host.")
    parser.add_argument("--port", type=int, default=None, help="Override configured port.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    args = parser.parse_args()

    import uvicorn

    if args.config:
        os.environ[CONFIG_PATH_ENV_VAR] = str(Path(args.config).expanduser().resolve())
    if args.agent_config:
        os.environ[AGENT_CONFIG_PATH_ENV_VAR] = str(Path(args.agent_config).expanduser().resolve())
    if args.env:
        os.environ[ENVIRONMENT_ENV_VAR] = str(args.env)
        os.environ[AGENT_ENVIRONMENT_ENV_VAR] = str(args.env)
    if args.protein_env:
        os.environ[ENVIRONMENT_ENV_VAR] = str(args.protein_env)
    if args.agent_env:
        os.environ[AGENT_ENVIRONMENT_ENV_VAR] = str(args.agent_env)

    protein_settings = load_config(
        config_path=args.config,
        environment=args.protein_env or args.env,
    )
    agent_settings = load_settings(
        config_path=args.agent_config,
        environment=args.agent_env or args.env,
    )
    reload_enabled = bool(
        args.reload or protein_settings.server.reload or agent_settings.server.reload
    )
    run_kwargs: dict[str, Any] = {
        "factory": True,
        "host": args.host or protein_settings.server.host or agent_settings.server.host,
        "port": args.port or protein_settings.server.port,
        "reload": reload_enabled,
    }
    if reload_enabled:
        # WatchFiles registers each reload dir recursively and crashes with an
        # EIO ("os error 5") the moment it descends into a host-mounted
        # virtualenv -- e.g. the Windows .venv-win shared into this Linux
        # container via the repo bind-mount. uvicorn's reload_excludes only
        # filter events, they do not stop the walk, so we instead point the
        # watcher at first-party source dirs only and never at any .venv*/data.
        app_dir = Path(__file__).resolve().parent
        repo_root = app_dir.parent
        candidate_dirs = [
            app_dir / "ai_agent",
            app_dir / "interfere",
            app_dir / "structure_predictor",
            app_dir / "tools",
            repo_root / "libs",
        ]
        watch_dirs = [str(path) for path in candidate_dirs if path.is_dir()]
        if watch_dirs:
            run_kwargs["reload_dirs"] = watch_dirs
    uvicorn.run("server:create_app", **run_kwargs)


class _HttpError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _WorkflowCancelled(RuntimeError):
    pass


async def _resolve_clarification(
    *,
    websocket: Any,
    request: Any,
    model_factory: Any,
    settings: AISettings,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    current_request = request
    for _ in range(3):
        if not current_request.require_clarification:
            return {"request": current_request, "query": current_request.user_input}

        await _send_event(
            websocket,
            "clarification_started",
            {"user_input": current_request.user_input},
        )
        clarification = await asyncio.to_thread(
            _clarify_protein_request,
            model_factory(),
            settings,
            current_request.user_input,
            _normalize_context(current_request.context),
            evidence or [],
        )
        await _send_event(websocket, "clarification_completed", clarification)

        if not clarification.get("needs_clarification"):
            return {
                "request": current_request,
                "query": str(clarification.get("proposed_query") or current_request.user_input),
            }

        await _send_event(
            websocket,
            "waiting_for_user",
            {
                "message": clarification.get("message"),
                "proposed_query": clarification.get("proposed_query"),
                "expected_actions": ["approve", "revise", "cancel"],
            },
        )
        decision = await websocket.receive_json()
        action = str(decision.get("action") or "").strip().lower()
        if action in {"approve", "ok", "continue"}:
            return {
                "request": current_request,
                "query": str(clarification.get("proposed_query") or current_request.user_input),
            }
        if action in {"revise", "edit"}:
            revised_input = str(decision.get("user_input") or "").strip()
            if not revised_input:
                raise ValueError("revise action requires a non-empty user_input.")
            current_request = _copy_model(current_request, user_input=revised_input)
            continue
        if action == "cancel":
            await _send_event(websocket, "cancelled", {"message": "Workflow cancelled by user."})
            raise _WorkflowCancelled()
        raise ValueError("Unknown clarification action. Use approve, revise, or cancel.")

    raise ValueError("Clarification did not converge after 3 attempts.")


async def _run_public_research(
    *,
    websocket: Any,
    request: Any,
    query: str,
    search_tool_factory: Any,
) -> list[dict[str, Any]]:
    if not request.research_with_exa:
        await _send_event(websocket, "public_research_skipped", {"reason": "disabled"})
        return []

    await _send_event(
        websocket,
        "public_research_started",
        {
            "tool": "exa_search",
            "message": "Using Exa only for public background evidence, not final protein ranking.",
            "query": query,
        },
    )
    try:
        results = await asyncio.to_thread(search_tool_factory().search, query)
    except Exception as exc:
        await _send_event(
            websocket,
            "public_research_failed",
            {"tool": "exa_search", "error": str(exc), "continuing": True},
        )
        return []

    evidence = [result.to_dict() for result in results]
    await _send_event(
        websocket,
        "public_research_completed",
        {"tool": "exa_search", "result_count": len(evidence), "results": evidence},
    )
    return evidence


def _build_span_completion_prompt(
    *,
    raw_input: str,
    source: Literal["ncbi", "ena"],
    limit: int,
    mask_policy: str,
    mask_start: int,
    mask_length: int,
    left_flank_size: int,
    right_flank_size: int,
) -> dict[str, str]:
    mask_end = mask_start + mask_length
    records = _fetch_sequence_records(raw_input, source=source, limit=limit)
    records = [record for record in records if _compact_protein_sequence(record.sequence)]
    if not records:
        raise _HttpError(404, "No sequence records found.")

    long_enough = [
        record
        for record in records
        if len(_compact_protein_sequence(record.sequence)) > mask_end
    ]
    if not long_enough:
        raise _HttpError(400, "No fetched sequence is long enough for the requested mask span.")

    selected = long_enough[0]
    source_row = _build_span_source_row(selected, raw_input=raw_input)
    span_row = make_span_completion_example(
        source_row,
        source_index=0,
        mask_start=mask_start,
        mask_end=mask_end,
        mask_policy=mask_policy,
        left_flank_size=left_flank_size,
        right_flank_size=right_flank_size,
    )
    return {
        "instruction": span_row["instruction"],
        "input": span_row["input"],
    }


def _fetch_sequence_records(
    raw_input: str,
    *,
    source: Literal["ncbi", "ena", "auto"],
    limit: int,
) -> list[Any]:
    if source not in {"ncbi", "ena", "auto"}:
        raise _HttpError(
            400,
            f"Unsupported sequence source '{source}'. Use 'ncbi', 'ena', or 'auto'.",
        )
    (
        FetchRequest,
        DataNotFoundError,
        SourceConfigurationError,
        NcbiSequenceSource,
        EnaSequenceSource,
    ) = _span_completion_data_dependencies()

    source_names = ("ncbi", "ena") if source == "auto" else (source,)
    errors: list[str] = []
    records: list[Any] = []
    for source_name in source_names:
        try:
            query = _build_source_query(raw_input, source_name=source_name)
            transport = _new_http_transport()
            sequence_source = (
                NcbiSequenceSource(transport=transport)
                if source_name == "ncbi"
                else EnaSequenceSource(transport=transport)
            )
            fetch_request = FetchRequest(
                dataset_name="protein-span-completion",
                query=query,
                limit=limit,
                extra_fields=("gene", "product", "host", "keywords"),
            )
            records.extend(sequence_source.fetch(fetch_request))
        except DataNotFoundError as exc:
            errors.append(str(exc))
        except SourceConfigurationError as exc:
            raise _HttpError(400, str(exc)) from exc
        except Exception as exc:
            errors.append(f"{source_name.upper()} source fetch failed: {exc}")

    if records:
        return records
    detail = "; ".join(errors) if errors else "No sequence records found."
    raise _HttpError(404, detail)


async def _send_event(websocket: Any, event: str, payload: dict[str, Any]) -> None:
    await websocket.send_json({"event": event, **payload})


def _create_model(settings: AISettings) -> ChatModel:
    return OpenAIChatModel(
        model=settings.openai.model,
        api_key=settings.require_openai_api_key(),
    )


def _model_cache_key(settings: AISettings) -> str:
    return f"openai|{settings.openai.model}|{settings.openai.api_key_env}"


def _configured_model_name(settings: AISettings) -> str:
    return settings.openai.model


def _normalize_context(context: str | list[str] | dict[str, Any] | None) -> str:
    if context is None:
        return ""
    if isinstance(context, str):
        return context
    return json.dumps(context, ensure_ascii=True, indent=2)


def _model_dump(model: Any) -> dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _copy_model(model: Any, **updates: Any) -> Any:
    if hasattr(model, "model_copy"):
        return model.model_copy(update=updates)
    return model.copy(update=updates)


def _compact_protein_sequence(sequence: str) -> str:
    return "".join(str(sequence or "").split()).upper()


__all__ = [
    "PROTEIN_SPAN_COMPLETION_WS_ROUTE",
    "SPAN_COMPLETION_ROUTE",
    "create_app",
    "main",
]


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

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
    RejectionRequest,
)
from ai_agent.skills import load_agent_skills
from ai_agent.tools.exa_search import ExaSearchTool


_MODEL_CACHE: dict[str, ChatModel] = {}
_TOOL_CACHE: dict[str, ExaSearchTool] = {}


def create_app(
    config_path: str | Path | None = None,
    environment: str | None = None,
):
    from fastapi import FastAPI, HTTPException

    settings = load_settings(config_path=config_path, environment=environment)
    approval_store = InMemoryApprovalStore()
    app = FastAPI(title="MDNAC AI Agent API", version="0.1.0")
    app.state.settings = settings
    app.state.approval_store = approval_store

    def get_model() -> ChatModel:
        cache_key = _model_cache_key(settings)
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = _create_model(settings)
        return _MODEL_CACHE[cache_key]

    def get_search_tool() -> ExaSearchTool:
        cache_key = (
            f"{settings.exa.api_key_env}|{settings.exa.search_type}|"
            f"{settings.exa.max_results}"
        )
        if cache_key not in _TOOL_CACHE:
            _TOOL_CACHE[cache_key] = ExaSearchTool(settings.exa)
        return _TOOL_CACHE[cache_key]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "environment": settings.environment,
            "provider": settings.provider,
        }

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        try:
            get_model().ensure_ready()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "status": "ready",
            "environment": settings.environment,
            "provider": settings.provider,
            "model": _configured_model_name(settings),
        }

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

    @app.post("/agent/run", response_model=AgentRunResponse)
    def run(request: AgentRunRequest) -> AgentRunResponse:
        try:
            return run_agent(
                settings=settings,
                model=get_model(),
                search_tool=get_search_tool(),
                approval_store=approval_store,
                user_input=request.user_input,
                context=_normalize_context(request.context),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the local MDNAC AI agent HTTP API.",
    )
    parser.add_argument("--config", default=None, help="Path to ai_agent/config/agent.yaml.")
    parser.add_argument("--env", default=None, help="Environment name from agent.yaml.")
    parser.add_argument("--host", default=None, help="Override configured host.")
    parser.add_argument("--port", type=int, default=None, help="Override configured port.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    args = parser.parse_args()

    import uvicorn

    if args.config:
        os.environ[AGENT_CONFIG_PATH_ENV_VAR] = str(Path(args.config).expanduser().resolve())
    if args.env:
        os.environ[AGENT_ENVIRONMENT_ENV_VAR] = str(args.env)

    settings = load_settings(config_path=args.config, environment=args.env)
    uvicorn.run(
        "ai_agent.server:create_app",
        factory=True,
        host=args.host or settings.server.host,
        port=args.port or settings.server.port,
        reload=bool(args.reload or settings.server.reload),
    )


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


__all__ = ["create_app", "main"]


if __name__ == "__main__":
    main()

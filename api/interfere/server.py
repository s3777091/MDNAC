from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .config import (
    APISettings,
    CONFIG_PATH_ENV_VAR,
    ENVIRONMENT_ENV_VAR,
    generation_kwargs,
    load_config,
)
from .inference import InferenceAPI


def create_app(
    *,
    config_path: str | Path | None = None,
    environment: str | None = None,
):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "Serving the API requires FastAPI and Pydantic. "
            "Install the api project with the local or production extra."
        ) from exc

    settings = load_config(config_path=config_path, environment=environment)
    api_cache: dict[str, InferenceAPI] = {}

    class GenerateRequest(BaseModel):
        prompt: str = ""
        max_new_tokens: int | None = Field(default=None, ge=0)
        temperature: float | None = Field(default=None, ge=0.0)
        top_k: int | None = Field(default=None, gt=0)
        seed: int | None = None
        stop_at_endoftext: bool | None = None
        ensure_protein_prompt: bool | None = None

    app = FastAPI(
        title="MDNAC Protein API",
        version="0.1.0",
    )
    app.state.settings = settings

    def get_api() -> InferenceAPI:
        cache_key = f"{settings.model.path}|{settings.model.device}"
        if cache_key not in api_cache:
            api_cache[cache_key] = InferenceAPI.load(
                model_path=settings.model.path,
                device_name=settings.model.device,
            )
        return api_cache[cache_key]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "environment": settings.environment,
            "model_path": str(settings.model.path),
            "device": settings.model.device,
        }

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        try:
            api = get_api()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "status": "ready",
            "environment": settings.environment,
            "model_path": str(api.session.onnx_path),
            "providers": list(api.session.providers),
        }

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        kwargs = _merge_generation_request(settings, payload)
        try:
            result = get_api().generate_protein(request.prompt, **kwargs)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the local MDNAC protein HTTP API.",
    )
    parser.add_argument("--config", default=None, help="Path to api/config.yaml.")
    parser.add_argument("--env", default=None, help="Environment name from config.yaml.")
    parser.add_argument("--host", default=None, help="Override configured host.")
    parser.add_argument("--port", type=int, default=None, help="Override configured port.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Running the local server requires uvicorn. Install the api local extra."
        ) from exc

    if args.config:
        os.environ[CONFIG_PATH_ENV_VAR] = str(Path(args.config).expanduser().resolve())
    if args.env:
        os.environ[ENVIRONMENT_ENV_VAR] = str(args.env)

    settings = load_config(config_path=args.config, environment=args.env)
    uvicorn.run(
        "interfere.server:create_app",
        factory=True,
        host=args.host or settings.server.host,
        port=args.port or settings.server.port,
        reload=bool(args.reload or settings.server.reload),
    )


def _merge_generation_request(
    settings: APISettings,
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    kwargs = generation_kwargs(settings.generation)
    for key in tuple(kwargs):
        if request_payload.get(key) is not None:
            kwargs[key] = request_payload[key]
    return kwargs


__all__ = ["create_app", "main"]


if __name__ == "__main__":
    main()

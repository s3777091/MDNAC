from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .config import CONFIG_PATH_ENV_VAR, ENVIRONMENT_ENV_VAR, load_config
from .openfold import OpenFoldRunner, StructurePredictionRequest


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
            "Serving the structure API requires FastAPI and Pydantic. "
            "Install the api project with the structure extra."
        ) from exc

    settings = load_config(config_path=config_path, environment=environment)
    runner_cache: dict[str, OpenFoldRunner] = {}

    class PredictStructureRequest(BaseModel):
        sequence: str = Field(min_length=1)
        name: str = "candidate"
        output_format: str | None = None
        config_preset: str | None = None
        include_structure_text: bool | None = None
        job_id: str | None = None

    app = FastAPI(
        title="MDNAC OpenFold Structure Predictor API",
        version="0.1.0",
    )
    app.state.settings = settings

    def get_runner() -> OpenFoldRunner:
        cache_key = f"{settings.openfold.repo_path}|{settings.openfold.config_preset}"
        if cache_key not in runner_cache:
            runner_cache[cache_key] = OpenFoldRunner(settings.openfold)
        return runner_cache[cache_key]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "environment": settings.environment,
            "model": "OpenFold",
            "config_preset": settings.openfold.config_preset,
            "model_device": settings.openfold.model_device,
        }

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        readiness = get_runner().readiness()
        if not readiness["script_exists"]:
            raise HTTPException(
                status_code=503,
                detail=f"OpenFold script not found: {readiness['script_path']}",
            )
        return {
            "status": "ready",
            "environment": settings.environment,
            **readiness,
        }

    @app.post("/predict-structure")
    def predict_structure(request: PredictStructureRequest) -> dict[str, Any]:
        payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        try:
            result = get_runner().predict(StructurePredictionRequest(**payload))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the local MDNAC OpenFold structure HTTP API.",
    )
    parser.add_argument("--config", default=None, help="Path to api/config.structure.yaml.")
    parser.add_argument("--env", default=None, help="Environment name from config.structure.yaml.")
    parser.add_argument("--host", default=None, help="Override configured host.")
    parser.add_argument("--port", type=int, default=None, help="Override configured port.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Running the local structure server requires uvicorn. Install the api structure extra."
        ) from exc

    if args.config:
        os.environ[CONFIG_PATH_ENV_VAR] = str(Path(args.config).expanduser().resolve())
    if args.env:
        os.environ[ENVIRONMENT_ENV_VAR] = str(args.env)

    settings = load_config(config_path=args.config, environment=args.env)
    uvicorn.run(
        "structure_predictor.server:create_app",
        factory=True,
        host=args.host or settings.server.host,
        port=args.port or settings.server.port,
        reload=bool(args.reload or settings.server.reload),
    )


__all__ = ["create_app", "main"]


if __name__ == "__main__":
    main()

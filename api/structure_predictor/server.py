from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import Any, Literal

from .config import CONFIG_PATH_ENV_VAR, ENVIRONMENT_ENV_VAR, load_config
from .openfold import OpenFoldRunner, StructurePredictionRequest
from .simulation.dispatcher import enqueue_local_simulation_job
from .simulation.jobs import TASK_OPENMM_SIMULATION, SimulationJobStore
from .simulation.runpod import submit_runpod_simulation_job


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

    class SimulationEnvironmentRequest(BaseModel):
        solvent: str = "water"
        temperature_k: float = Field(default=300.0, gt=0.0)
        ph: float = Field(default=7.4, ge=0.0, le=14.0)
        salt_m: float = Field(default=0.15, ge=0.0)
        steps: int = Field(default=50_000, gt=0)
        report_interval: int = Field(default=500, gt=0)
        gpu_device: int = Field(default=0, ge=0)

    class SimulationJobRequest(BaseModel):
        structure_job_id: str | int
        pdb_path: str | None = None
        cif_path: str | None = None
        run_target: Literal["local", "runpod"] = "local"
        environment: SimulationEnvironmentRequest = Field(
            default_factory=SimulationEnvironmentRequest
        )

    app = FastAPI(
        title="MDNAC OpenFold Structure Predictor API",
        version="0.1.0",
    )
    app.state.settings = settings
    app.state.simulation_jobs = SimulationJobStore(settings.simulation.jobs_root)

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

    def predict_structure(request) -> dict[str, Any]:
        payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        try:
            result = get_runner().predict(StructurePredictionRequest(**payload))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    predict_structure.__annotations__["request"] = PredictStructureRequest
    app.post("/predict-structure")(predict_structure)

    def create_simulation_job(request) -> dict[str, Any]:
        payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        payload = _validate_simulation_payload(payload, http_exception=HTTPException)
        job_id = uuid.uuid4().hex
        payload["job_id"] = job_id
        payload["task"] = TASK_OPENMM_SIMULATION
        run_target = str(payload["run_target"])

        external_job_id = None
        if run_target == "runpod":
            external_job_id = submit_runpod_simulation_job(payload)

        job_state = app.state.simulation_jobs.create_job(
            job_id=job_id,
            payload=payload,
            run_target=run_target,
            external_job_id=external_job_id,
        )
        if run_target == "local":
            try:
                enqueue_local_simulation_job(payload, settings=settings)
            except Exception as exc:
                app.state.simulation_jobs.update_job(
                    job_id,
                    status="failed",
                    step="enqueue",
                    error=str(exc),
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to enqueue simulation job: {exc}",
                ) from exc

        response = {
            "job_id": job_id,
            "status": job_state["status"],
            "task": TASK_OPENMM_SIMULATION,
            "run_target": run_target,
        }
        if external_job_id:
            response["external_job_id"] = external_job_id
        return response

    create_simulation_job.__annotations__["request"] = SimulationJobRequest
    app.post("/api/v1/predict/simulation", include_in_schema=False)(create_simulation_job)
    app.post("/predict/simulation")(create_simulation_job)

    @app.get("/api/v1/predict/jobs/{job_id}", include_in_schema=False)
    @app.get("/predict/jobs/{job_id}")
    def get_simulation_job(job_id: str) -> dict[str, Any]:
        job_state = _load_simulation_job_or_404(app.state.simulation_jobs, job_id, HTTPException)
        response = {
            "job_id": job_state["job_id"],
            "task": job_state["task"],
            "status": job_state["status"],
            "step": job_state.get("step"),
            "progress": job_state.get("progress"),
            "error": job_state.get("error"),
        }
        if job_state.get("external_job_id"):
            response["external_job_id"] = job_state["external_job_id"]
        return response

    @app.get("/api/v1/predict/jobs/{job_id}/result", include_in_schema=False)
    @app.get("/predict/jobs/{job_id}/result")
    def get_simulation_job_result(job_id: str) -> dict[str, Any]:
        job_state = _load_simulation_job_or_404(app.state.simulation_jobs, job_id, HTTPException)
        if job_state["status"] != "completed":
            return {
                "job_id": job_state["job_id"],
                "status": job_state["status"],
                "message": "Job is not completed yet",
            }
        return {
            "job_id": job_state["job_id"],
            "status": job_state["status"],
            "task": job_state["task"],
            "result": job_state.get("result"),
        }

    return app


def _validate_simulation_payload(payload: dict[str, Any], *, http_exception):
    structure_job_id = str(payload.get("structure_job_id") or "").strip()
    if not structure_job_id:
        raise http_exception(status_code=400, detail="structure_job_id is required.")
    payload["structure_job_id"] = structure_job_id

    pdb_path = payload.get("pdb_path")
    cif_path = payload.get("cif_path")
    if not pdb_path and not cif_path:
        raise http_exception(status_code=400, detail="Either pdb_path or cif_path is required.")
    if not pdb_path and cif_path:
        raise http_exception(
            status_code=400,
            detail="CIF simulation is not supported yet; please provide pdb_path.",
        )

    environment = payload.get("environment") or {}
    if environment.get("solvent") != "water":
        raise http_exception(
            status_code=400,
            detail="Only solvent='water' is supported for OpenMM simulation.",
        )
    if int(environment["report_interval"]) > int(environment["steps"]):
        raise http_exception(
            status_code=400,
            detail="report_interval must be less than or equal to steps.",
        )
    return payload


def _load_simulation_job_or_404(store: SimulationJobStore, job_id: str, http_exception):
    try:
        return store.get_job(job_id)
    except FileNotFoundError as exc:
        raise http_exception(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise http_exception(status_code=400, detail=str(exc)) from exc


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

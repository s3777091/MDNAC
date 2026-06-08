from __future__ import annotations

from typing import Any

from runpod_flash import CpuInstanceType, Endpoint, GpuGroup


def _load_settings():
    from structure_predictor.config import load_config

    return load_config(environment="production")


def _endpoint_kwargs(settings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "name": settings.runpod.endpoint_name,
        "workers": (settings.runpod.workers_min, settings.runpod.workers_max),
        "idle_timeout": settings.runpod.idle_timeout,
        "flashboot": settings.runpod.flashboot,
        "dependencies": list(settings.runpod.dependencies),
    }
    if settings.runpod.gpu:
        kwargs["gpu"] = getattr(GpuGroup, settings.runpod.gpu)
    elif settings.runpod.cpu:
        kwargs["cpu"] = getattr(CpuInstanceType, settings.runpod.cpu)
    else:
        kwargs["gpu"] = GpuGroup.ANY
    return kwargs


SETTINGS = _load_settings()
endpoint = Endpoint(**_endpoint_kwargs(SETTINGS))
_RUNNER_CACHE: dict[str, object] = {}


def _get_runner():
    from structure_predictor.openfold import OpenFoldRunner

    cache_key = f"{SETTINGS.openfold.repo_path}|{SETTINGS.openfold.config_preset}"
    if cache_key not in _RUNNER_CACHE:
        _RUNNER_CACHE[cache_key] = OpenFoldRunner(SETTINGS.openfold)
    return _RUNNER_CACHE[cache_key]


@endpoint.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "environment": SETTINGS.environment,
        "model": "OpenFold",
        "config_preset": SETTINGS.openfold.config_preset,
        "model_device": SETTINGS.openfold.model_device,
    }


@endpoint.get("/ready")
async def ready() -> dict[str, Any]:
    readiness = _get_runner().readiness()
    status = "ready" if readiness["script_exists"] else "not_ready"
    return {
        "status": status,
        "environment": SETTINGS.environment,
        **readiness,
    }


@endpoint.post("/predict-structure")
async def predict_structure(data: dict[str, Any]) -> dict[str, Any]:
    from structure_predictor.openfold import StructurePredictionRequest

    result = _get_runner().predict(
        StructurePredictionRequest(
            sequence=str(data.get("sequence") or ""),
            name=str(data.get("name") or "candidate"),
            output_format=data.get("output_format"),
            config_preset=data.get("config_preset"),
            include_structure_text=data.get("include_structure_text"),
            job_id=data.get("job_id"),
        )
    )
    return result.to_dict()

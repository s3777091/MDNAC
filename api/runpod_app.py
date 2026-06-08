from __future__ import annotations

from pathlib import Path
from typing import Any

from runpod_flash import CpuInstanceType, Endpoint, GpuGroup


def _load_settings():
    from interfere.config import load_config

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
_API_CACHE: dict[str, object] = {}


def _get_api():
    from interfere.inference import InferenceAPI

    cache_key = f"{SETTINGS.model.path}|{SETTINGS.model.device}"
    if cache_key not in _API_CACHE:
        _API_CACHE[cache_key] = InferenceAPI.load(
            model_path=SETTINGS.model.path,
            device_name=SETTINGS.model.device,
        )
    return _API_CACHE[cache_key]


def _generation_kwargs(data: dict[str, Any]) -> dict[str, Any]:
    from interfere.config import generation_kwargs

    kwargs = generation_kwargs(SETTINGS.generation)
    for key in tuple(kwargs):
        if key in data and data[key] is not None:
            kwargs[key] = data[key]
    return kwargs


@endpoint.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "environment": SETTINGS.environment,
        "model_path": str(SETTINGS.model.path),
        "device": SETTINGS.model.device,
    }


@endpoint.get("/ready")
async def ready() -> dict[str, Any]:
    api = _get_api()
    return {
        "status": "ready",
        "environment": SETTINGS.environment,
        "model_path": str(api.session.onnx_path),
        "providers": list(api.session.providers),
    }


@endpoint.post("/generate")
async def generate(data: dict[str, Any]) -> dict[str, Any]:
    prompt = str(data.get("prompt") or "")
    result = _get_api().generate_protein(prompt, **_generation_kwargs(data))
    return result.to_dict()


def model_volume_path() -> Path:
    return SETTINGS.model.path

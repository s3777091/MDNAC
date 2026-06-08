from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


API_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = API_ROOT / "config.yaml"
CONFIG_PATH_ENV_VAR = "MDNAC_API_CONFIG"
ENVIRONMENT_ENV_VAR = "MDNAC_API_ENV"


@dataclass(frozen=True)
class ModelSettings:
    path: Path
    device: str


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int
    temperature: float
    top_k: int | None
    seed: int | None
    stop_at_endoftext: bool
    ensure_protein_prompt: bool


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    reload: bool


@dataclass(frozen=True)
class RunpodSettings:
    enabled: bool
    endpoint_name: str
    gpu: str | None
    cpu: str | None
    workers_min: int
    workers_max: int
    idle_timeout: int
    flashboot: bool
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class APISettings:
    environment: str
    config_path: Path
    model: ModelSettings
    generation: GenerationSettings
    server: ServerSettings
    runpod: RunpodSettings


def load_config(
    *,
    config_path: str | Path | None = None,
    environment: str | None = None,
) -> APISettings:
    resolved_config_path = Path(
        config_path or os.environ.get(CONFIG_PATH_ENV_VAR) or DEFAULT_CONFIG_PATH
    ).expanduser()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"API config.yaml not found: {resolved_config_path}")

    payload = _read_yaml(resolved_config_path)
    environments = payload.get("environments")
    if not isinstance(environments, dict) or not environments:
        raise ValueError("config.yaml must contain a non-empty `environments` mapping.")

    selected_environment = (
        environment
        or os.environ.get(ENVIRONMENT_ENV_VAR)
        or payload.get("environment")
        or "local"
    )
    selected_environment = str(selected_environment).strip()
    if selected_environment not in environments:
        available = ", ".join(sorted(str(name) for name in environments))
        raise ValueError(
            f"Unknown API environment {selected_environment!r}. Available: {available}"
        )

    raw_environment = environments[selected_environment]
    if not isinstance(raw_environment, dict):
        raise ValueError(f"Environment {selected_environment!r} must be a mapping.")

    return APISettings(
        environment=selected_environment,
        config_path=resolved_config_path.resolve(),
        model=_load_model_settings(raw_environment, base_dir=resolved_config_path.parent),
        generation=_load_generation_settings(raw_environment),
        server=_load_server_settings(raw_environment),
        runpod=_load_runpod_settings(raw_environment),
    )


def generation_kwargs(settings: GenerationSettings) -> dict[str, Any]:
    return {
        "max_new_tokens": settings.max_new_tokens,
        "temperature": settings.temperature,
        "top_k": settings.top_k,
        "seed": settings.seed,
        "stop_at_endoftext": settings.stop_at_endoftext,
        "ensure_protein_prompt": settings.ensure_protein_prompt,
    }


def _load_model_settings(raw_environment: dict[str, Any], *, base_dir: Path) -> ModelSettings:
    raw_model = raw_environment.get("model") or {}
    if not isinstance(raw_model, dict):
        raise ValueError("Environment `model` must be a mapping.")

    raw_path = raw_model.get("path") or "data/model"
    model_path = _resolve_config_path(str(raw_path), base_dir=base_dir)
    return ModelSettings(
        path=model_path,
        device=str(raw_model.get("device") or "auto"),
    )


def _load_generation_settings(raw_environment: dict[str, Any]) -> GenerationSettings:
    raw_generation = raw_environment.get("generation") or {}
    if not isinstance(raw_generation, dict):
        raise ValueError("Environment `generation` must be a mapping.")

    return GenerationSettings(
        max_new_tokens=int(raw_generation.get("max_new_tokens") or 128),
        temperature=float(raw_generation.get("temperature") or 0.0),
        top_k=_optional_int(raw_generation.get("top_k")),
        seed=_optional_int(raw_generation.get("seed")),
        stop_at_endoftext=bool(raw_generation.get("stop_at_endoftext", True)),
        ensure_protein_prompt=bool(raw_generation.get("ensure_protein_prompt", True)),
    )


def _load_server_settings(raw_environment: dict[str, Any]) -> ServerSettings:
    raw_server = raw_environment.get("server") or {}
    if not isinstance(raw_server, dict):
        raise ValueError("Environment `server` must be a mapping.")

    return ServerSettings(
        host=str(raw_server.get("host") or "127.0.0.1"),
        port=int(raw_server.get("port") or 8000),
        reload=bool(raw_server.get("reload", False)),
    )


def _load_runpod_settings(raw_environment: dict[str, Any]) -> RunpodSettings:
    raw_runpod = raw_environment.get("runpod") or {}
    if not isinstance(raw_runpod, dict):
        raise ValueError("Environment `runpod` must be a mapping.")

    raw_workers = raw_runpod.get("workers") or {}
    if not isinstance(raw_workers, dict):
        raw_workers = {}

    raw_dependencies = raw_runpod.get("dependencies") or ()
    if isinstance(raw_dependencies, str):
        dependencies = (raw_dependencies,)
    else:
        dependencies = tuple(str(item) for item in raw_dependencies)

    return RunpodSettings(
        enabled=bool(raw_runpod.get("enabled", False)),
        endpoint_name=str(raw_runpod.get("endpoint_name") or "mdnac-protein-api"),
        gpu=_optional_str(raw_runpod.get("gpu")),
        cpu=_optional_str(raw_runpod.get("cpu")),
        workers_min=int(raw_workers.get("min") or 0),
        workers_max=int(raw_workers.get("max") or 1),
        idle_timeout=int(raw_runpod.get("idle_timeout") or 60),
        flashboot=bool(raw_runpod.get("flashboot", True)),
        dependencies=dependencies,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Reading api/config.yaml requires PyYAML. Install the api project dependencies."
        ) from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object in {path}.")
    return payload


def _resolve_config_path(raw_path: str, *, base_dir: Path) -> Path:
    expanded = Path(os.path.expandvars(raw_path)).expanduser()
    if expanded.is_absolute():
        return expanded
    return (base_dir / expanded).resolve()


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


__all__ = [
    "APISettings",
    "CONFIG_PATH_ENV_VAR",
    "DEFAULT_CONFIG_PATH",
    "ENVIRONMENT_ENV_VAR",
    "GenerationSettings",
    "ModelSettings",
    "RunpodSettings",
    "ServerSettings",
    "generation_kwargs",
    "load_config",
]

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


API_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = API_ROOT / "config.structure.yaml"
CONFIG_PATH_ENV_VAR = "MDNAC_STRUCTURE_CONFIG"
ENVIRONMENT_ENV_VAR = "MDNAC_STRUCTURE_ENV"


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
class ServerSettings:
    host: str
    port: int
    reload: bool


@dataclass(frozen=True)
class OpenFoldSettings:
    repo_path: Path
    python_executable: str
    output_root: Path
    template_mmcif_dir: Path
    config_preset: str
    model_device: str
    output_format: str
    timeout_seconds: int
    cpus: int
    min_sequence_length: int
    max_sequence_length: int
    include_structure_text: bool
    max_response_structure_bytes: int
    use_precomputed_alignments: bool
    precomputed_alignments_dir: Path | None
    openfold_checkpoint_path: Path | None
    jax_param_path: Path | None
    data_random_seed: int | None
    skip_relaxation: bool
    long_sequence_inference: bool
    use_single_seq_mode: bool
    database_paths: dict[str, Path]
    binary_paths: dict[str, Path]
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class StructureAPISettings:
    environment: str
    config_path: Path
    server: ServerSettings
    runpod: RunpodSettings
    openfold: OpenFoldSettings


def load_config(
    *,
    config_path: str | Path | None = None,
    environment: str | None = None,
) -> StructureAPISettings:
    resolved_config_path = Path(
        config_path or os.environ.get(CONFIG_PATH_ENV_VAR) or DEFAULT_CONFIG_PATH
    ).expanduser()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"Structure config.yaml not found: {resolved_config_path}")

    payload = _read_yaml(resolved_config_path)
    environments = payload.get("environments")
    if not isinstance(environments, dict) or not environments:
        raise ValueError("config.structure.yaml must contain a non-empty `environments` mapping.")

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
            f"Unknown structure API environment {selected_environment!r}. Available: {available}"
        )

    raw_environment = environments[selected_environment]
    if not isinstance(raw_environment, dict):
        raise ValueError(f"Environment {selected_environment!r} must be a mapping.")

    return StructureAPISettings(
        environment=selected_environment,
        config_path=resolved_config_path.resolve(),
        server=_load_server_settings(raw_environment),
        runpod=_load_runpod_settings(raw_environment),
        openfold=_load_openfold_settings(
            raw_environment,
            base_dir=resolved_config_path.parent,
        ),
    )


def _load_server_settings(raw_environment: dict[str, Any]) -> ServerSettings:
    raw_server = raw_environment.get("server") or {}
    if not isinstance(raw_server, dict):
        raise ValueError("Environment `server` must be a mapping.")

    return ServerSettings(
        host=str(raw_server.get("host") or "127.0.0.1"),
        port=int(raw_server.get("port") or 8010),
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
        endpoint_name=str(raw_runpod.get("endpoint_name") or "mdnac-structure-openfold-api"),
        gpu=_optional_str(raw_runpod.get("gpu")),
        cpu=_optional_str(raw_runpod.get("cpu")),
        workers_min=int(raw_workers.get("min") or 0),
        workers_max=int(raw_workers.get("max") or 1),
        idle_timeout=int(raw_runpod.get("idle_timeout") or 300),
        flashboot=bool(raw_runpod.get("flashboot", True)),
        dependencies=dependencies,
    )


def _load_openfold_settings(
    raw_environment: dict[str, Any],
    *,
    base_dir: Path,
) -> OpenFoldSettings:
    raw_openfold = raw_environment.get("openfold") or {}
    if not isinstance(raw_openfold, dict):
        raise ValueError("Environment `openfold` must be a mapping.")

    raw_databases = raw_openfold.get("database_paths") or {}
    if not isinstance(raw_databases, dict):
        raise ValueError("Environment `openfold.database_paths` must be a mapping.")

    raw_binaries = raw_openfold.get("binary_paths") or {}
    if not isinstance(raw_binaries, dict):
        raise ValueError("Environment `openfold.binary_paths` must be a mapping.")

    output_format = str(raw_openfold.get("output_format") or "pdb").lower()
    if output_format not in {"pdb", "cif"}:
        raise ValueError("openfold.output_format must be `pdb` or `cif`.")

    extra_args = raw_openfold.get("extra_args") or ()
    if isinstance(extra_args, str):
        extra_args = (extra_args,)

    return OpenFoldSettings(
        repo_path=_resolve_config_path(str(raw_openfold.get("repo_path") or "../openfold"), base_dir),
        python_executable=str(raw_openfold.get("python_executable") or "python"),
        output_root=_resolve_config_path(
            str(raw_openfold.get("output_root") or "data/structure_predictions"),
            base_dir,
        ),
        template_mmcif_dir=_resolve_config_path(
            str(raw_openfold.get("template_mmcif_dir") or "data/openfold/pdb_mmcif/mmcif_files"),
            base_dir,
        ),
        config_preset=str(raw_openfold.get("config_preset") or "seq_model_esm1b_ptm"),
        model_device=str(raw_openfold.get("model_device") or "cuda:0"),
        output_format=output_format,
        timeout_seconds=int(raw_openfold.get("timeout_seconds") or 7200),
        cpus=int(raw_openfold.get("cpus") or 4),
        min_sequence_length=int(raw_openfold.get("min_sequence_length") or 1),
        max_sequence_length=int(raw_openfold.get("max_sequence_length") or 1022),
        include_structure_text=bool(raw_openfold.get("include_structure_text", True)),
        max_response_structure_bytes=int(
            raw_openfold.get("max_response_structure_bytes") or 2_000_000
        ),
        use_precomputed_alignments=bool(raw_openfold.get("use_precomputed_alignments", False)),
        precomputed_alignments_dir=_optional_path(
            raw_openfold.get("precomputed_alignments_dir"),
            base_dir=base_dir,
        ),
        openfold_checkpoint_path=_optional_path(
            raw_openfold.get("openfold_checkpoint_path"),
            base_dir=base_dir,
        ),
        jax_param_path=_optional_path(raw_openfold.get("jax_param_path"), base_dir=base_dir),
        data_random_seed=_optional_int(raw_openfold.get("data_random_seed")),
        skip_relaxation=bool(raw_openfold.get("skip_relaxation", False)),
        long_sequence_inference=bool(raw_openfold.get("long_sequence_inference", False)),
        use_single_seq_mode=bool(raw_openfold.get("use_single_seq_mode", False)),
        database_paths={
            str(key): _resolve_config_path(str(value), base_dir)
            for key, value in raw_databases.items()
            if value not in (None, "")
        },
        binary_paths={
            str(key): _resolve_config_path(str(value), base_dir)
            for key, value in raw_binaries.items()
            if value not in (None, "")
        },
        extra_args=tuple(str(item) for item in extra_args),
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Reading config.structure.yaml requires PyYAML. Install the api project dependencies."
        ) from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object in {path}.")
    return payload


def _resolve_config_path(raw_path: str, base_dir: Path) -> Path:
    expanded = Path(os.path.expandvars(raw_path)).expanduser()
    if expanded.is_absolute():
        return expanded
    return (base_dir / expanded).resolve()


def _optional_path(value: Any, *, base_dir: Path) -> Path | None:
    if value is None or value == "":
        return None
    return _resolve_config_path(str(value), base_dir)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None

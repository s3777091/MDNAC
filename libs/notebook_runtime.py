from __future__ import annotations

import os
import platform
import sys
from copy import deepcopy
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any

import torch


def is_colab_runtime() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def is_notebook_runtime() -> bool:
    return "ipykernel" in sys.modules or is_colab_runtime()


def platform_name() -> str:
    if is_colab_runtime():
        return "Google Colab"
    if platform.system() == "Linux":
        try:
            release = platform.freedesktop_os_release()
        except (AttributeError, OSError):
            return "Linux"
        return release.get("PRETTY_NAME") or release.get("NAME") or "Linux"
    return platform.system()


def load_env_file(path: Path | str) -> None:
    resolved_path = Path(path).expanduser()
    if not resolved_path.exists():
        return
    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def find_repo_dir(start: Path | str | None = None, *, env_var: str = "MDNAC_REPO_DIR") -> Path:
    candidates: list[Path] = []
    if start is not None:
        start_path = Path(start).expanduser().resolve()
        candidates.extend([start_path, *start_path.parents])

    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(Path(env_value).expanduser())

    if is_colab_runtime():
        candidates.extend(
            [
                Path("/content/MDNAC"),
                Path("/content/drive/MyDrive/MDNAC"),
            ]
        )

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "pyproject.toml").exists() and (resolved / "libs").is_dir():
            return resolved

    raise RuntimeError(
        "Could not locate the repo. Run from inside the project, set MDNAC_REPO_DIR, "
        "or in Colab clone/mount the repo under /content or /content/drive/MyDrive."
    )


def bootstrap_notebook(
    start: Path | str | None = None,
    *,
    chdir: bool = True,
    load_dotenv: bool = True,
) -> dict[str, Any]:
    repo_dir = find_repo_dir(Path.cwd() if start is None else start)
    if chdir:
        os.chdir(repo_dir)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    if load_dotenv:
        load_env_file(repo_dir / ".env")

    os.environ.setdefault("MICROBIAL_DATA_STORAGE_MODE", "local")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if platform.system() == "Windows":
        os.environ.setdefault("USE_LIBUV", "0")

    return runtime_summary(repo_dir)


def runtime_summary(repo_dir: Path | str) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count())
    return {
        "repo_dir": str(Path(repo_dir).resolve()),
        "platform": platform.system(),
        "platform_name": platform_name(),
        "is_colab": is_colab_runtime(),
        "is_notebook": is_notebook_runtime(),
        "python": sys.version.split()[0],
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
    }


def notebook_num_workers(configured_workers: int) -> int:
    if platform.system() == "Windows" and is_notebook_runtime():
        return 0
    return int(configured_workers)


def notebook_multi_gpu_mode(configured_mode: str) -> str:
    mode = str(configured_mode)
    if is_notebook_runtime() and mode == "ddp":
        return "data_parallel"
    return mode


def apply_instruction_notebook_overrides(config):
    updates: dict[str, Any] = {}
    resolved_workers = notebook_num_workers(getattr(config, "num_workers", 0))
    if resolved_workers != getattr(config, "num_workers", 0):
        updates["num_workers"] = resolved_workers
    resolved_mode = notebook_multi_gpu_mode(getattr(config, "multi_gpu_mode", "auto"))
    if resolved_mode != getattr(config, "multi_gpu_mode", "auto"):
        updates["multi_gpu_mode"] = resolved_mode
    if not updates:
        return config
    if is_dataclass(config):
        return replace(config, **updates)
    for key, value in updates.items():
        setattr(config, key, value)
    return config


def apply_training_config_notebook_overrides(config: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(config)
    data_cfg = dict(resolved.get("data", {}))
    runtime_cfg = dict(resolved.get("runtime", {}))
    data_cfg["num_workers"] = notebook_num_workers(int(data_cfg.get("num_workers", 0)))
    runtime_cfg["multi_gpu_mode"] = notebook_multi_gpu_mode(runtime_cfg.get("multi_gpu_mode", "auto"))
    resolved["data"] = data_cfg
    resolved["runtime"] = runtime_cfg
    return resolved


def materialize_notebook_training_config(
    config_path: Path | str,
    *,
    project_root: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> tuple[Path, list[str]]:
    """Write a small notebook-safe config copy when platform overrides are needed."""
    import yaml

    resolved_config_path = Path(config_path).expanduser()
    if not resolved_config_path.is_absolute():
        base_dir = find_repo_dir(project_root) if project_root is not None else Path.cwd()
        resolved_config_path = (base_dir / resolved_config_path).resolve()
        if not resolved_config_path.exists():
            config_path_candidate = (base_dir / "config" / Path(config_path).expanduser()).resolve()
            if config_path_candidate.exists():
                resolved_config_path = config_path_candidate
    else:
        resolved_config_path = resolved_config_path.resolve()

    raw_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"Expected a YAML mapping in {resolved_config_path}")

    adjusted_config = deepcopy(raw_config)
    changes: list[str] = []

    data_cfg = adjusted_config.setdefault("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("training config field data must be a mapping")
    configured_workers = int(data_cfg.get("num_workers", 0) or 0)
    resolved_workers = notebook_num_workers(configured_workers)
    if resolved_workers != configured_workers:
        data_cfg["num_workers"] = resolved_workers
        changes.append(f"data.num_workers: {configured_workers} -> {resolved_workers}")

    runtime_cfg = adjusted_config.setdefault("runtime", {})
    if not isinstance(runtime_cfg, dict):
        raise ValueError("training config field runtime must be a mapping")
    configured_mode = str(runtime_cfg.get("multi_gpu_mode", "auto") or "auto")
    resolved_mode = notebook_multi_gpu_mode(configured_mode)
    if resolved_mode != configured_mode:
        runtime_cfg["multi_gpu_mode"] = resolved_mode
        changes.append(f"runtime.multi_gpu_mode: {configured_mode} -> {resolved_mode}")

    if not changes:
        return resolved_config_path, changes

    repo_dir = find_repo_dir(project_root) if project_root is not None else find_repo_dir(resolved_config_path.parent)
    target_dir = Path(output_dir).expanduser() if output_dir is not None else repo_dir / ".notebook_runtime"
    if not target_dir.is_absolute():
        target_dir = (repo_dir / target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / f"{resolved_config_path.stem}.notebook{resolved_config_path.suffix}"
    target_path.write_text(
        yaml.safe_dump(adjusted_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target_path, changes

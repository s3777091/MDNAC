from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping


DEFAULT_QWEN_DIRNAME = "Qwen3.5-0.8B"
DEFAULT_QWEN_REPO_ID = "Qwen/Qwen3.5-0.8B"
QWEN_REPO_ID_2B = "Qwen/Qwen3.5-2B"
GENERIC_ENV_VARS = (
    "QWEN3_5_MODEL_DIR",
    "QWEN_MODEL_DIR",
    "HF_QWEN_DIR",
)
DEFAULT_ENV_VARS = ("QWEN3_5_0_8B_DIR", *GENERIC_ENV_VARS)


class _QwenProjectPaths:
    def __init__(self, project_root: str | Path) -> None:
        self.root = Path(project_root).resolve()
        self.pretrained_root = self.root / "data" / "pretrained"
        self.models_root = self.root / "models"
        self.checkpoints_root = self.root / "data" / "checkpoints"
        self.tokenizers_root = self.root / "data" / "tokenizers"

    def qwen_tokenizer_dir(self, *, model_dirname: str) -> Path:
        return self.tokenizers_root / model_dirname

    def ensure_qwen_tokenizer_dir(self, *, model_dirname: str) -> Path:
        path = self.qwen_tokenizer_dir(model_dirname=model_dirname)
        path.mkdir(parents=True, exist_ok=True)
        return path


def _repo_id_basename(repo_id: str) -> str:
    return repo_id.replace("\\", "/").rstrip("/").split("/")[-1].strip()


def _infer_qwen_repo_id(model_config: Mapping[str, Any] | None) -> str | None:
    if model_config is None:
        return None

    emb_dim = model_config.get("emb_dim")
    hidden_dim = model_config.get("hidden_dim")
    if emb_dim == 2_048 and hidden_dim == 6_144:
        return QWEN_REPO_ID_2B
    if emb_dim == 1_024 and hidden_dim == 3_584:
        return DEFAULT_QWEN_REPO_ID
    return None


def resolve_qwen_repo_id(
    repo_id: str | None = None,
    model_config: Mapping[str, Any] | None = None,
) -> str:
    normalized = str(repo_id or "").strip()
    inferred = _infer_qwen_repo_id(model_config)

    if inferred and normalized:
        normalized_basename = _repo_id_basename(normalized)
        inferred_basename = _repo_id_basename(inferred)
        if normalized_basename != inferred_basename and normalized_basename.startswith("Qwen3.5-"):
            return inferred

    if normalized:
        return normalized

    if inferred:
        return inferred

    return DEFAULT_QWEN_REPO_ID


def qwen_repo_id_to_dirname(repo_id: str | None = None) -> str:
    normalized = resolve_qwen_repo_id(repo_id)
    candidate = _repo_id_basename(normalized)
    return candidate or DEFAULT_QWEN_DIRNAME


def _model_dirname_to_slug(model_dirname: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model_dirname.casefold()).strip("_")
    if slug:
        return slug
    return re.sub(r"[^a-z0-9]+", "_", DEFAULT_QWEN_DIRNAME.casefold()).strip("_")


def qwen_repo_id_env_vars(repo_id: str | None = None) -> tuple[str, ...]:
    model_dirname = qwen_repo_id_to_dirname(repo_id)
    specific = f"{re.sub(r'[^A-Z0-9]+', '_', model_dirname.upper()).strip('_')}_DIR"
    return tuple(dict.fromkeys((specific, *GENERIC_ENV_VARS)))


def _default_project_model_dirs(
    project_root: str | Path,
    *,
    repo_id: str = DEFAULT_QWEN_REPO_ID,
) -> tuple[Path, ...]:
    data_paths = _QwenProjectPaths(project_root)
    model_dirname = qwen_repo_id_to_dirname(repo_id)
    model_slug = _model_dirname_to_slug(model_dirname)
    return (
        data_paths.qwen_tokenizer_dir(model_dirname=model_dirname),
        data_paths.pretrained_root / model_dirname,
        data_paths.models_root / model_dirname,
        data_paths.checkpoints_root / f"{model_slug}_avepoint_continue" / "hf_last",
    )


def _looks_like_local_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weights = (
        (path / "pytorch_model.bin").exists()
        or (path / "model.safetensors").exists()
        or any(path.glob("*.safetensors"))
        or any(path.glob("pytorch_model*.bin"))
    )
    return has_config and has_weights


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def iter_qwen_candidate_sources(
    project_root: str | Path,
    *,
    extra_sources: list[str | Path] | None = None,
    env_vars: tuple[str, ...] | None = None,
    include_repo_id: bool = False,
    repo_id: str = DEFAULT_QWEN_REPO_ID,
) -> list[str]:
    root = Path(project_root).resolve()
    resolved_repo_id = resolve_qwen_repo_id(repo_id)
    resolved_env_vars = qwen_repo_id_env_vars(resolved_repo_id) if env_vars is None else env_vars
    candidates: list[str] = []

    for env_name in resolved_env_vars:
        env_value = (os.getenv(env_name) or "").strip()
        if env_value:
            candidates.append(str(Path(env_value).expanduser()))

    for candidate in _default_project_model_dirs(root, repo_id=resolved_repo_id):
        if candidate.exists():
            candidates.append(str(candidate))

    if extra_sources:
        for source in extra_sources:
            if source is None:
                continue
            candidates.append(str(Path(source).expanduser()))

    if include_repo_id:
        candidates.append(resolved_repo_id)

    return _unique_strings(candidates)


def resolve_qwen_tokenizer_json(
    project_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    repo_id: str = DEFAULT_QWEN_REPO_ID,
    local_files_only: bool = True,
    allow_download: bool = False,
    extra_sources: list[str | Path] | None = None,
) -> Path:
    root = Path(project_root).resolve()
    data_paths = _QwenProjectPaths(root)
    resolved_repo_id = resolve_qwen_repo_id(repo_id)
    model_dirname = qwen_repo_id_to_dirname(resolved_repo_id)
    destination_dir = (
        data_paths.ensure_qwen_tokenizer_dir(model_dirname=model_dirname)
        if output_dir is None
        else Path(output_dir).resolve()
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "tokenizer.json"

    if destination.exists():
        return destination

    sources = iter_qwen_candidate_sources(
        root,
        extra_sources=extra_sources,
        include_repo_id=allow_download,
        repo_id=resolved_repo_id,
    )

    for raw_source in sources:
        source_path = Path(raw_source)
        if source_path.is_file() and source_path.name == "tokenizer.json":
            shutil.copy2(source_path, destination)
            return destination
        if source_path.is_dir():
            tokenizer_json = source_path / "tokenizer.json"
            if tokenizer_json.exists():
                shutil.copy2(tokenizer_json, destination)
                return destination

    if allow_download:
        try:
            from ..modeling import download_from_huggingface

            downloaded_path = download_from_huggingface(
                repo_id=resolved_repo_id,
                filename="tokenizer.json",
                local_dir=str(destination_dir),
            )
            return Path(downloaded_path)
        except Exception:
            pass

    download_hint = (
        f"Set allow_download=True or place the tokenizer in data/pretrained/{model_dirname}."
        if not allow_download
        else "Verify the repo id, credentials, or network access."
    )
    raise RuntimeError(
        f"Unable to resolve tokenizer.json for {resolved_repo_id}.\n"
        f"Looked in local env/project paths and optional Hugging Face sources.\n"
        f"{download_hint}\n"
        "No transformers fallback is used here."
    )


def resolve_qwen_model_source(
    project_root: str | Path,
    *,
    repo_id: str = DEFAULT_QWEN_REPO_ID,
    local_files_only: bool = True,
    allow_download: bool = False,
    extra_sources: list[str | Path] | None = None,
) -> str:
    root = Path(project_root).resolve()
    resolved_repo_id = resolve_qwen_repo_id(repo_id)
    model_dirname = qwen_repo_id_to_dirname(resolved_repo_id)

    for source in iter_qwen_candidate_sources(
        root,
        extra_sources=extra_sources,
        include_repo_id=False,
        repo_id=resolved_repo_id,
    ):
        source_path = Path(source)
        if _looks_like_local_model_dir(source_path):
            return source

    if allow_download:
        return resolved_repo_id

    download_hint = (
        f"Set allow_download=True or copy the model into data/pretrained/{model_dirname}."
        if not allow_download
        else "Verify the repo id, credentials, or network access."
    )
    raise RuntimeError(
        f"Unable to resolve a local/cached source for {resolved_repo_id}.\n"
        f"{download_hint}\n"
        "No transformers fallback is used here."
    )

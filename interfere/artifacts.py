from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from train.pipeline.runtime.artifacts import load_checkpoint


ArtifactFormat = Literal["pytorch", "onnx"]
DEFAULT_CHECKPOINT_PATH = Path("data") / "checkpoints" / "checkpoint_best.pt"
PREFERRED_CHECKPOINT_FILES = ("checkpoint_best.pt", "checkpoint_last.pt")


@dataclass(frozen=True)
class ResolvedArtifact:
    source_path: Path
    artifact_format: ArtifactFormat
    model_family: str
    model_config: dict[str, Any]
    metadata: dict[str, Any]
    checkpoint: dict[str, Any] | None
    checkpoint_path: Path | None
    onnx_path: Path | None
    metadata_path: Path | None
    reference_path: Path


def detect_model_family(metadata: dict[str, Any]) -> str:
    family = str(metadata.get("model_family") or "").strip().lower()
    if family == "qwen3_5" or "qwen" in family or family.endswith("_0_8b"):
        return "qwen3_5"

    model_config = metadata.get("model_config") or {}
    if isinstance(model_config, dict) and (
        "layer_types" in model_config or "partial_rotary_factor" in model_config
    ):
        return "qwen3_5"

    raise ValueError(
        "Unsupported inference artifact: only Qwen3.5 checkpoints/exports are accepted in this repo. "
        f"Could not detect a supported model family from keys: {sorted(metadata.keys())}"
    )


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON metadata file: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object metadata in {path}.")
    return payload


def _resolve_input_path(source_path: str | Path) -> Path:
    resolved = Path(source_path).expanduser()
    if not resolved.is_dir():
        return resolved

    for checkpoint_name in PREFERRED_CHECKPOINT_FILES:
        checkpoint_file = resolved / checkpoint_name
        if checkpoint_file.is_file():
            return checkpoint_file

    onnx_files = sorted(resolved.glob("*.onnx"))
    if onnx_files:
        return onnx_files[0]

    metadata_files = sorted(resolved.glob("*.json"))
    for metadata_file in metadata_files:
        try:
            metadata = _read_json_file(metadata_file)
        except ValueError:
            continue
        if "onnx_path" in metadata:
            return metadata_file

    return resolved / PREFERRED_CHECKPOINT_FILES[0]


def _resolve_onnx_metadata_path(onnx_path: Path) -> Path:
    metadata_path = onnx_path.with_suffix(".json")
    if metadata_path.is_file():
        return metadata_path.resolve()
    raise FileNotFoundError(
        "ONNX inference requires the export sidecar metadata JSON next to the .onnx file.\n"
        f"Expected: {metadata_path}"
    )


def _resolve_onnx_path_from_metadata(metadata_path: Path, metadata: dict[str, Any]) -> Path:
    candidates: list[Path] = []
    if metadata.get("onnx_path"):
        raw_onnx_path = Path(str(metadata["onnx_path"])).expanduser()
        if raw_onnx_path.is_absolute():
            candidates.append(raw_onnx_path)
        candidates.append(metadata_path.parent / raw_onnx_path)
    candidates.append(metadata_path.with_suffix(".onnx"))

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.is_file():
            return normalized

    searched = "\n".join(f"- {item}" for item in seen)
    raise FileNotFoundError(
        "Unable to resolve the ONNX file for this metadata sidecar.\n"
        f"Searched:\n{searched}"
    )


def load_inference_artifact(source_path: str | Path) -> ResolvedArtifact:
    artifact_path = _resolve_input_path(source_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Inference artifact not found: {artifact_path}")

    suffix = artifact_path.suffix.lower()
    if suffix == ".pt":
        checkpoint_path = artifact_path.resolve()
        checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError(
                "Expected a training checkpoint containing a `model_config` dictionary."
            )
        return ResolvedArtifact(
            source_path=checkpoint_path,
            artifact_format="pytorch",
            model_family=detect_model_family(checkpoint),
            model_config=model_config,
            metadata=checkpoint,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            onnx_path=None,
            metadata_path=None,
            reference_path=checkpoint_path,
        )

    if suffix == ".onnx":
        onnx_path = artifact_path.resolve()
        metadata_path = _resolve_onnx_metadata_path(onnx_path)
        metadata = _read_json_file(metadata_path)
        model_config = metadata.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError("ONNX metadata is missing a `model_config` dictionary.")
        checkpoint_path = (
            Path(str(metadata["checkpoint_path"])).expanduser().resolve()
            if metadata.get("checkpoint_path")
            else None
        )
        return ResolvedArtifact(
            source_path=onnx_path,
            artifact_format="onnx",
            model_family=detect_model_family(metadata),
            model_config=model_config,
            metadata=metadata,
            checkpoint=None,
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            metadata_path=metadata_path,
            reference_path=checkpoint_path or metadata_path,
        )

    if suffix == ".json":
        metadata_path = artifact_path.resolve()
        metadata = _read_json_file(metadata_path)
        if "onnx_path" not in metadata and not metadata_path.with_suffix(".onnx").is_file():
            raise ValueError(
                "JSON inference metadata must contain `onnx_path` or sit next to a matching .onnx file."
            )
        model_config = metadata.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError("ONNX metadata is missing a `model_config` dictionary.")
        onnx_path = _resolve_onnx_path_from_metadata(metadata_path, metadata)
        checkpoint_path = (
            Path(str(metadata["checkpoint_path"])).expanduser().resolve()
            if metadata.get("checkpoint_path")
            else None
        )
        return ResolvedArtifact(
            source_path=metadata_path,
            artifact_format="onnx",
            model_family=detect_model_family(metadata),
            model_config=model_config,
            metadata=metadata,
            checkpoint=None,
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            metadata_path=metadata_path,
            reference_path=checkpoint_path or metadata_path,
        )

    raise ValueError(
        "Unsupported inference artifact. Expected a checkpoint `.pt`, an ONNX `.onnx`, "
        "or an ONNX sidecar `.json`."
    )

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


API_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = API_ROOT / "data" / "model"
MODEL_PATH_ENV_VAR = "MDNAC_ONNX_MODEL"
SUPPORTED_MODEL_FAMILIES = {"progen_protein_lm", "mdc_protein_lm"}


@dataclass(frozen=True)
class ResolvedArtifact:
    source_path: Path
    onnx_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    model_config: dict[str, Any]
    tokenizer_payload: dict[str, Any]
    model_family: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    context_length: int


def detect_model_family(metadata: dict[str, Any]) -> str:
    family = str(metadata.get("model_family") or "").strip()
    if family in SUPPORTED_MODEL_FAMILIES:
        return family

    if not family and isinstance(metadata.get("model_config"), dict):
        return "progen_protein_lm"

    supported = ", ".join(sorted(SUPPORTED_MODEL_FAMILIES))
    raise ValueError(
        "Unsupported ONNX inference artifact. "
        f"Expected a protein model family in {{{supported}}}; received {family!r}."
    )


def load_inference_artifact(source_path: str | Path | None = None) -> ResolvedArtifact:
    artifact_path = _resolve_source_path(source_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Inference artifact not found: {artifact_path}")

    suffix = artifact_path.suffix.lower()
    if suffix == ".onnx":
        onnx_path = artifact_path.resolve()
        metadata_path = _metadata_path_for_onnx(onnx_path)
        metadata = _read_json_file(metadata_path)
    elif suffix == ".json":
        metadata_path = artifact_path.resolve()
        metadata = _read_json_file(metadata_path)
        onnx_path = _resolve_onnx_path_from_metadata(metadata_path, metadata)
    else:
        raise ValueError(
            "Unsupported inference artifact. Expected an ONNX `.onnx` file, "
            "a sidecar `.json`, or a directory containing exactly one model."
        )

    model_config = metadata.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(f"ONNX metadata is missing a `model_config` object: {metadata_path}")

    context_length = int(model_config.get("context_length") or 0)
    if context_length <= 0:
        raise ValueError("ONNX metadata model_config must contain a positive `context_length`.")

    tokenizer_payload = _load_tokenizer_payload(metadata_path, metadata)
    return ResolvedArtifact(
        source_path=artifact_path.resolve(),
        onnx_path=onnx_path,
        metadata_path=metadata_path,
        metadata=metadata,
        model_config=dict(model_config),
        tokenizer_payload=tokenizer_payload,
        model_family=detect_model_family(metadata),
        input_names=_metadata_names(metadata, "input_names", default=("input_ids",)),
        output_names=_metadata_names(metadata, "output_names", default=("logits",)),
        context_length=context_length,
    )


def _resolve_source_path(source_path: str | Path | None) -> Path:
    if source_path is None:
        source_path = os.environ.get(MODEL_PATH_ENV_VAR) or DEFAULT_MODEL_DIR

    path = Path(source_path).expanduser()
    if path.is_dir():
        return _resolve_model_from_dir(path)
    return path


def _resolve_model_from_dir(model_dir: Path) -> Path:
    onnx_files = sorted(model_dir.glob("*.onnx"))
    if len(onnx_files) == 1:
        return onnx_files[0]
    if len(onnx_files) > 1:
        names = "\n".join(f"- {path.name}" for path in onnx_files)
        raise ValueError(
            "Multiple ONNX models were found. Pass the exact model path or set "
            f"`{MODEL_PATH_ENV_VAR}` so the API cannot load the wrong model.\n{names}"
        )

    metadata_files = sorted(model_dir.glob("*.json"))
    if len(metadata_files) == 1:
        return metadata_files[0]
    if len(metadata_files) > 1:
        names = "\n".join(f"- {path.name}" for path in metadata_files)
        raise ValueError(
            "Multiple ONNX metadata files were found. Pass the exact metadata path or set "
            f"`{MODEL_PATH_ENV_VAR}`.\n{names}"
        )

    raise FileNotFoundError(
        "No ONNX model was found for inference.\n"
        f"Expected exactly one `.onnx` file in: {model_dir}\n"
        f"Or set `{MODEL_PATH_ENV_VAR}` to the exact `.onnx` or sidecar `.json` path."
    )


def _metadata_path_for_onnx(onnx_path: Path) -> Path:
    metadata_path = onnx_path.with_suffix(".json")
    if metadata_path.is_file():
        return metadata_path.resolve()
    raise FileNotFoundError(
        "ONNX inference requires a sidecar metadata JSON next to the model.\n"
        f"Expected: {metadata_path}"
    )


def _resolve_onnx_path_from_metadata(metadata_path: Path, metadata: dict[str, Any]) -> Path:
    candidates: list[Path] = []
    raw_value = metadata.get("onnx_path")
    if raw_value:
        raw_path = Path(str(raw_value)).expanduser()
        if raw_path.is_absolute():
            candidates.append(raw_path)
            candidates.append(metadata_path.parent / raw_path.name)
        else:
            candidates.append(metadata_path.parent / raw_path)
    candidates.append(metadata_path.with_suffix(".onnx"))

    for candidate in _unique_paths(candidates):
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(f"- {path}" for path in _unique_paths(candidates))
    raise FileNotFoundError(
        "Unable to resolve the ONNX file for this metadata sidecar.\n"
        f"Searched:\n{searched}"
    )


def _load_tokenizer_payload(metadata_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    embedded = metadata.get("tokenizer_map") or metadata.get("tokenizer")
    if isinstance(embedded, dict):
        return dict(embedded)

    candidates: list[Path] = []
    for key in ("tokenizer_map_path", "tokenizer_path"):
        raw_value = metadata.get(key)
        if not raw_value:
            continue
        raw_path = Path(str(raw_value)).expanduser()
        if raw_path.is_absolute():
            candidates.append(raw_path)
            candidates.append(metadata_path.parent / raw_path.name)
        else:
            candidates.append(metadata_path.parent / raw_path)

    candidates.append(metadata_path.parent / "tokenizer_map.json")
    for candidate in _unique_paths(candidates):
        if candidate.is_file():
            payload = _read_json_file(candidate)
            return dict(payload.get("tokenizer", payload))

    searched = "\n".join(f"- {path}" for path in _unique_paths(candidates))
    raise FileNotFoundError(
        "ONNX metadata does not contain an embedded `tokenizer_map`, and no tokenizer map "
        f"could be found next to the model.\nSearched:\n{searched}"
    )


def _metadata_names(
    metadata: dict[str, Any],
    key: str,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    export = metadata.get("export")
    value = export.get(key) if isinstance(export, dict) else None
    value = value or metadata.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        normalized = path.expanduser()
        try:
            normalized = normalized.resolve(strict=False)
        except OSError:
            pass
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique

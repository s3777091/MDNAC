from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from libs.data.config import DataConfig
from libs.data.entities import FetchRequest, PreparationSessionArtifact
from libs.data.training.normalization import SequenceNormalizationConfig

from .helpers import accession_hash, canonical_accession


def session_dir(config: DataConfig, source_name: str, dataset_name: str) -> Path:
    return config.sessions_root / source_name / dataset_name


def artifact_from_manifest(
    config: DataConfig,
    manifest: dict[str, object],
    active_session_dir: Path,
    manifest_path: Path,
) -> PreparationSessionArtifact:
    tokenizer_map_path = manifest.get("tokenizer_map_path")
    train_txt_path = manifest.get("train_txt_path")
    if not manifest.get("is_complete"):
        tokenizer_map_path = None
        train_txt_path = str(active_session_dir / "train.txt")

    return PreparationSessionArtifact(
        source_name=str(manifest.get("source_name", "")),
        dataset_name=str(manifest.get("dataset_name", "")),
        storage_mode=str(manifest.get("storage_mode", config.storage_mode)),  # type: ignore[arg-type]
        session_location=str(active_session_dir),
        manifest_path=str(manifest_path),
        train_txt_path=str(train_txt_path),
        tokenizer_map_path=str(tokenizer_map_path) if tokenizer_map_path is not None else None,
        processed_count=int(manifest.get("processed_count", 0)),
        total_count=int(manifest.get("total_count", 0)),
        record_count=int(manifest.get("record_count", 0)),
        dropped_record_count=int(manifest.get("dropped_record_count", 0)),
        is_complete=bool(manifest.get("is_complete", False)),
        current_location=str(manifest.get("current_location")) if manifest.get("current_location") else None,
        snapshot_id=str(manifest.get("snapshot_id")) if manifest.get("snapshot_id") else None,
    )


def request_signature(
    config: DataConfig,
    source_name: str,
    request: FetchRequest,
    sequence_type: str,
    normalization: SequenceNormalizationConfig,
    vocab_size: int | None,
) -> str:
    payload = {
        "source_name": source_name,
        "dataset_name": request.dataset_name,
        "query": request.query,
        "accession_hash": sha256("\n".join(request.accessions).encode("utf-8")).hexdigest(),
        "limit": request.limit,
        "batch_size": request.batch_size,
        "extra_fields": list(request.extra_fields),
        "include_suppressed": request.include_suppressed,
        "sequence_type": sequence_type,
        "normalization": asdict(normalization),
        "vocab_size": vocab_size,
        "storage_mode": config.storage_mode,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def request_payload(
    request: FetchRequest,
    accessions: tuple[str, ...],
    duplicate_accession_count: int,
) -> dict[str, object]:
    return {
        "query": request.query,
        "limit": request.limit,
        "batch_size": request.batch_size,
        "extra_fields": list(request.extra_fields),
        "include_suppressed": request.include_suppressed,
        "accession_hash": accession_hash(accessions),
        "resolved_accession_count": len(accessions),
        "duplicate_accession_count": duplicate_accession_count,
    }


def normalize_requested_accessions(
    accessions: tuple[str, ...],
) -> tuple[tuple[str, ...], dict[str, str], int]:
    ordered_accessions: list[str] = []
    accession_aliases: dict[str, str] = {}
    seen_accessions: set[str] = set()
    duplicate_accession_count = 0

    for requested_accession in accessions:
        accession = canonical_accession(requested_accession)
        if not accession:
            continue
        if accession in seen_accessions:
            duplicate_accession_count += 1
            continue
        seen_accessions.add(accession)
        ordered_accessions.append(accession)
        accession_aliases[accession] = requested_accession

    return tuple(ordered_accessions), accession_aliases, duplicate_accession_count


def read_manifest(manifest_path: Path) -> dict[str, object]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def write_manifest(manifest_path: Path, manifest: dict[str, object]) -> None:
    request = manifest.get("request")
    manifest["accession_hash"] = str(request.get("accession_hash", "")) if isinstance(request, dict) else ""
    manifest["duplicate_accession_count"] = int(request.get("duplicate_accession_count", 0)) if isinstance(request, dict) else 0
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_raw_index(raw_index_path: Path) -> dict[str, dict[str, object]]:
    if not raw_index_path.exists():
        return {}
    payload = json.loads(raw_index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {
        str(accession): value
        for accession, value in payload.items()
        if isinstance(value, dict)
    }


def write_raw_index(raw_index_path: Path, raw_index: dict[str, dict[str, object]]) -> None:
    raw_index_path.write_text(
        json.dumps(raw_index, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def saved_artifact_paths_exist(manifest: dict[str, object]) -> bool:
    train_txt_path = manifest.get("train_txt_path")
    tokenizer_map_path = manifest.get("tokenizer_map_path")
    if train_txt_path is None or tokenizer_map_path is None:
        return False
    if str(manifest.get("storage_mode", "")) != "local":
        return True
    return Path(str(train_txt_path)).exists() and Path(str(tokenizer_map_path)).exists()

"""S3-based distributed 3Di annotation pipeline."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from libs.data.config import DataConfig
from libs.data.training.streaming import (
    S3TextPart,
    build_minio_s3_client,
    downloaded_minio_text_part,
    list_minio_text_parts,
    parse_s3_uri,
)

from .instruction_3di import (
    annotate_instruction_jsonl_3di,
    _ensure_provider_ready,
    _provider_model_name,
)


@dataclass(slots=True, frozen=True)
class S3Instruction3DiPartSummary:
    source_uri: str
    output_uri: str
    source_size_bytes: int | None
    output_size_bytes: int
    total_line_count: int
    written_line_count: int
    new_3di_count: int
    reused_existing_count: int
    cache_hit_count: int
    model_prediction_count: int
    empty_sequence_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class S3Instruction3DiUpdateSummary:
    source: str
    output_prefix_uri: str
    field_name: str
    model_name: str
    manifest_uri: str | None
    part_count: int
    total_line_count: int
    written_line_count: int
    new_3di_count: int
    reused_existing_count: int
    cache_hit_count: int
    model_prediction_count: int
    empty_sequence_count: int
    elapsed_seconds: float
    parts: tuple[S3Instruction3DiPartSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def annotate_s3_instruction_jsonl_3di(
    *,
    provider,
    prefix_uri: str | None = None,
    part_uris: Sequence[str] | None = None,
    output_prefix_uri: str | None = None,
    field_name: str = "3Di",
    sequence_field: str = "output",
    batch_size: int = 2,
    cache_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    keep_downloaded_parts: bool = False,
    s3_client=None,
    config: DataConfig | None = None,
    skip_existing: bool = True,
    overwrite: bool = False,
    allow_in_place: bool = False,
    upload_manifest: bool = True,
    progress_callback: Callable[[str], None] | None = None,
    report_every_seconds: float = 30.0,
) -> S3Instruction3DiUpdateSummary:
    if bool(prefix_uri) == bool(part_uris):
        raise ValueError("Provide exactly one of prefix_uri or part_uris.")

    resolved_output_prefix_uri = output_prefix_uri
    if resolved_output_prefix_uri is None:
        if prefix_uri is None:
            raise ValueError("output_prefix_uri is required when part_uris are provided.")
        resolved_output_prefix_uri = f"{prefix_uri.rstrip('/')}_3di"

    if prefix_uri is not None and _normalize_s3_prefix(prefix_uri) == _normalize_s3_prefix(resolved_output_prefix_uri):
        if not allow_in_place:
            raise ValueError(
                "output_prefix_uri matches prefix_uri. Pass allow_in_place=True if you intentionally want "
                "to overwrite source objects after each part is annotated."
            )

    _ensure_provider_ready(provider)

    client = s3_client or build_minio_s3_client(config)
    parts = list_minio_text_parts(
        prefix_uri=prefix_uri,
        part_uris=part_uris,
        s3_client=client,
        config=config,
        suffixes=(".jsonl",),
    )
    if not parts:
        source = prefix_uri or ", ".join(part_uris or ())
        raise FileNotFoundError(f"No instruction JSONL parts found in {source!r}.")

    output_bucket, output_prefix_key = parse_s3_uri(resolved_output_prefix_uri)
    output_prefix_key = output_prefix_key.strip("/")
    model_name = _provider_model_name(provider)
    start_time = time.time()
    part_summaries: list[S3Instruction3DiPartSummary] = []
    temp_root = Path(cache_dir) if cache_dir is not None else None
    if temp_root is not None:
        temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = _create_temporary_work_dir(prefix="mdc-3di-s3-", root=temp_root)

    try:
        for part in parts:
            if progress_callback is not None:
                progress_callback(f"[3Di/S3] download {part.uri}")

            with downloaded_minio_text_part(
                part,
                s3_client=client,
                config=config,
                cache_dir=cache_dir,
                keep_downloaded_parts=keep_downloaded_parts,
            ) as downloaded_part_path:
                local_output_path = temp_dir / f"{Path(part.key).name}.3di.jsonl"
                local_summary = annotate_instruction_jsonl_3di(
                    downloaded_part_path,
                    local_output_path,
                    provider,
                    field_name=field_name,
                    sequence_field=sequence_field,
                    batch_size=batch_size,
                    cache_path=cache_path,
                    skip_existing=skip_existing,
                    overwrite=True,
                    progress_callback=progress_callback,
                    report_every_seconds=report_every_seconds,
                )

                output_key = _s3_output_key_for_part(
                    part,
                    output_prefix_key=output_prefix_key,
                )
                if not overwrite and _s3_object_exists(client, output_bucket, output_key):
                    raise FileExistsError(f"S3 output object already exists: s3://{output_bucket}/{output_key}")

                _upload_s3_file(client, local_output_path, bucket=output_bucket, key=output_key)
                output_uri = f"s3://{output_bucket}/{output_key}"
                output_size = local_output_path.stat().st_size
                part_summaries.append(
                    S3Instruction3DiPartSummary(
                        source_uri=part.uri,
                        output_uri=output_uri,
                        source_size_bytes=part.size,
                        output_size_bytes=output_size,
                        total_line_count=local_summary.total_line_count,
                        written_line_count=local_summary.written_line_count,
                        new_3di_count=local_summary.new_3di_count,
                        reused_existing_count=local_summary.reused_existing_count,
                        cache_hit_count=local_summary.cache_hit_count,
                        model_prediction_count=local_summary.model_prediction_count,
                        empty_sequence_count=local_summary.empty_sequence_count,
                    )
                )

                if progress_callback is not None:
                    progress_callback(f"[3Di/S3] uploaded {output_uri}")

        totals = _summarize_s3_parts(part_summaries)
        manifest_uri = None
        if upload_manifest:
            manifest_key = f"{output_prefix_key}/manifest.3di.json" if output_prefix_key else "manifest.3di.json"
            manifest = _build_s3_manifest(
                source=prefix_uri or tuple(part_uris or ()),
                output_prefix_uri=resolved_output_prefix_uri,
                field_name=field_name,
                model_name=model_name,
                elapsed_seconds=round(time.time() - start_time, 3),
                parts=part_summaries,
            )
            client.put_object(
                Bucket=output_bucket,
                Key=manifest_key,
                Body=(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
                ContentType="application/json",
            )
            manifest_uri = f"s3://{output_bucket}/{manifest_key}"

        return S3Instruction3DiUpdateSummary(
            source=prefix_uri or ", ".join(part_uris or ()),
            output_prefix_uri=resolved_output_prefix_uri,
            field_name=field_name,
            model_name=model_name,
            manifest_uri=manifest_uri,
            part_count=len(part_summaries),
            total_line_count=totals["total_line_count"],
            written_line_count=totals["written_line_count"],
            new_3di_count=totals["new_3di_count"],
            reused_existing_count=totals["reused_existing_count"],
            cache_hit_count=totals["cache_hit_count"],
            model_prediction_count=totals["model_prediction_count"],
            empty_sequence_count=totals["empty_sequence_count"],
            elapsed_seconds=round(time.time() - start_time, 3),
            parts=tuple(part_summaries),
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# --- S3 helper functions ---


def _normalize_s3_prefix(uri: str) -> str:
    bucket, key = parse_s3_uri(uri)
    return f"s3://{bucket}/{key.strip('/')}"


def _s3_output_key_for_part(part: S3TextPart, *, output_prefix_key: str) -> str:
    file_name = Path(part.key).name
    return f"{output_prefix_key}/{file_name}" if output_prefix_key else file_name


def _s3_object_exists(client, bucket: str, key: str) -> bool:
    head_object = getattr(client, "head_object", None)
    if not callable(head_object):
        return False
    try:
        head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _upload_s3_file(client, path: Path, *, bucket: str, key: str) -> None:
    upload_file = getattr(client, "upload_file", None)
    if callable(upload_file):
        try:
            from boto3.s3.transfer import TransferConfig

            transfer_config = TransferConfig(
                multipart_threshold=64 * 1024 * 1024,
                multipart_chunksize=64 * 1024 * 1024,
                max_concurrency=4,
                use_threads=True,
            )
            upload_file(str(path), bucket, key, Config=transfer_config)
        except TypeError:
            upload_file(str(path), bucket, key)
        return

    client.put_object(Bucket=bucket, Key=key, Body=path.read_bytes())


def _create_temporary_work_dir(*, prefix: str, root: Path | None = None) -> Path:
    base_dir = root or Path(tempfile.gettempdir())
    base_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        candidate = base_dir / f"{prefix}{uuid.uuid4().hex[:12]}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a unique temporary directory under {base_dir}")


def _summarize_s3_parts(parts: Sequence[S3Instruction3DiPartSummary]) -> dict[str, int]:
    fields = (
        "total_line_count",
        "written_line_count",
        "new_3di_count",
        "reused_existing_count",
        "cache_hit_count",
        "model_prediction_count",
        "empty_sequence_count",
    )
    return {field: sum(int(getattr(part, field)) for part in parts) for field in fields}


def _build_s3_manifest(
    *,
    source: str | tuple[str, ...],
    output_prefix_uri: str,
    field_name: str,
    model_name: str,
    elapsed_seconds: float,
    parts: Sequence[S3Instruction3DiPartSummary],
) -> dict[str, object]:
    totals = _summarize_s3_parts(parts)
    return {
        "format": "instruction_jsonl_3di",
        "version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "output_prefix_uri": output_prefix_uri,
        "field_name": field_name,
        "model_name": model_name,
        "elapsed_seconds": elapsed_seconds,
        "totals": totals,
        "parts": [part.to_dict() for part in parts],
    }

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from libs.data.config import DataConfig
from libs.data.training.streaming import (
    S3TextPart,
    build_minio_s3_client,
    downloaded_minio_text_part,
    list_minio_text_parts,
    parse_s3_uri,
)

from .scoring import compact_protein_sequence
from .types import PROSTT5_3DI_TOKENS, VALID_PROTEIN_AMINO_ACIDS, StructurePrediction


DEFAULT_3DI_FIELD = "3Di"
DEFAULT_PROSTT5_MODEL_NAME = "Rostlab/ProstT5"
AA_TO_3DI_PREFIX = "<AA2fold>"
RARE_AMINO_ACID_PATTERN = re.compile(r"[UZOB]")


class Structure3DiBatchProvider(Protocol):
    model_name: str

    def predict_3di_batch(self, sequences: Sequence[str]) -> Sequence[str]:
        ...


@dataclass(slots=True, frozen=True)
class Instruction3DiUpdateSummary:
    input_path: str
    output_path: str
    field_name: str
    model_name: str
    total_line_count: int
    written_line_count: int
    new_3di_count: int
    reused_existing_count: int
    cache_hit_count: int
    model_prediction_count: int
    empty_sequence_count: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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


@dataclass(slots=True)
class _BufferedRecord:
    payload: MutableMapping[str, object] | None = None
    raw_line: str | None = None
    sequence: str | None = None
    needs_prediction: bool = False


class Sequence3DiCache:
    """Small SQLite cache keyed by model + protein sequence hash."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sequence_3di_cache (
                model_name TEXT NOT NULL,
                sequence_sha256 TEXT NOT NULL,
                sequence TEXT NOT NULL,
                structure_3di TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                PRIMARY KEY (model_name, sequence_sha256)
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def get(self, *, model_name: str, sequence: str) -> str | None:
        sequence_hash = _sequence_hash(sequence)
        row = self._connection.execute(
            """
            SELECT structure_3di
            FROM sequence_3di_cache
            WHERE model_name = ? AND sequence_sha256 = ? AND sequence = ?
            """,
            (model_name, sequence_hash, sequence),
        ).fetchone()
        if row is None:
            return None
        normalized = normalize_3di_structure(row[0])
        return normalized or None

    def set_many(self, *, model_name: str, values: Mapping[str, str]) -> None:
        if not values:
            return

        updated_at = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                model_name,
                _sequence_hash(sequence),
                sequence,
                structure_3di,
                updated_at,
            )
            for sequence, structure_3di in values.items()
        ]
        self._connection.executemany(
            """
            INSERT INTO sequence_3di_cache (
                model_name,
                sequence_sha256,
                sequence,
                structure_3di,
                updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(model_name, sequence_sha256) DO UPDATE SET
                sequence = excluded.sequence,
                structure_3di = excluded.structure_3di,
                updated_at_utc = excluded.updated_at_utc
            """,
            rows,
        )
        self._connection.commit()

    def __enter__(self) -> "Sequence3DiCache":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()


class ProstT5Structure3DiProvider:
    """AA-to-3Di provider backed by the optional Rostlab/ProstT5 model."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_PROSTT5_MODEL_NAME,
        device: str | None = None,
        use_half: bool | None = None,
        generation_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_half = use_half
        self.generation_kwargs = dict(generation_kwargs or _default_prostt5_generation_kwargs())
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._resolved_device = None

    @property
    def resolved_device(self) -> str | None:
        return str(self._resolved_device) if self._resolved_device is not None else self.device

    def predict_3di_batch(self, sequences: Sequence[str]) -> Sequence[str]:
        if not sequences:
            return ()

        self._load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._resolved_device is not None

        normalized_sequences = [normalize_prostt5_aa_sequence(sequence) for sequence in sequences]
        prepared_inputs = [
            f"{AA_TO_3DI_PREFIX} {' '.join(sequence)}"
            for sequence in normalized_sequences
        ]
        lengths = [len(sequence) for sequence in normalized_sequences]

        encoded = self._tokenizer(
            prepared_inputs,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        encoded = {key: value.to(self._resolved_device) for key, value in encoded.items()}

        with self._torch.inference_mode():
            generated = self._model.generate(
                encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_length=max(lengths),
                min_length=min(lengths),
                early_stopping=True,
                num_return_sequences=1,
                **self.generation_kwargs,
            )

        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        structures = [normalize_3di_structure(value) for value in decoded]
        for sequence, structure_3di in zip(normalized_sequences, structures, strict=True):
            if not structure_3di:
                raise ValueError("ProstT5 returned an empty 3Di prediction.")
            if len(structure_3di) != len(sequence):
                # Keep the result, but make the mismatch explicit so bad model/runtime settings do not pass silently.
                raise ValueError(
                    "ProstT5 returned a 3Di prediction with unexpected length "
                    f"({len(structure_3di)} for a {len(sequence)} residue sequence)."
                )
        return tuple(structures)

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "ProstT5Structure3DiProvider requires optional dependencies: "
                "transformers and sentencepiece. Install them in the active environment before running 3Di annotation."
            ) from exc

        resolved_device = torch.device(self.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(resolved_device)

        use_half = self.use_half
        if use_half is None:
            use_half = resolved_device.type == "cuda"
        if use_half and resolved_device.type != "cpu":
            model.half()
        else:
            model.float()
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = resolved_device


def annotate_instruction_jsonl_3di(
    input_path: Path | str,
    output_path: Path | str,
    provider,
    *,
    field_name: str = DEFAULT_3DI_FIELD,
    sequence_field: str = "output",
    batch_size: int = 2,
    cache_path: Path | str | None = None,
    skip_existing: bool = True,
    overwrite: bool = False,
    ensure_ascii: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    report_every_seconds: float = 30.0,
) -> Instruction3DiUpdateSummary:
    _ensure_provider_ready(provider)

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    resolved_input_path = Path(input_path)
    resolved_output_path = Path(output_path)
    if not resolved_input_path.is_file():
        raise FileNotFoundError(f"instruction.jsonl was not found: {resolved_input_path}")
    if resolved_input_path.resolve() == resolved_output_path.resolve():
        raise ValueError("input_path and output_path must be different.")
    if resolved_output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {resolved_output_path}")

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path = resolved_output_path.with_name(f"{resolved_output_path.name}.tmp")
    temporary_output_path.unlink(missing_ok=True)

    model_name = _provider_model_name(provider)
    cache = Sequence3DiCache(cache_path) if cache_path is not None else None
    start_time = time.time()
    last_report_time = start_time
    buffer: list[_BufferedRecord] = []
    pending_prediction_count = 0

    total_line_count = 0
    written_line_count = 0
    new_3di_count = 0
    reused_existing_count = 0
    cache_hit_count = 0
    model_prediction_count = 0
    empty_sequence_count = 0
    max_buffer_records = max(batch_size * 4, batch_size)

    def flush_buffer(target_handle) -> None:
        nonlocal buffer
        nonlocal pending_prediction_count
        nonlocal written_line_count
        nonlocal new_3di_count
        nonlocal model_prediction_count

        if not buffer:
            return

        pending_records = [record for record in buffer if record.needs_prediction]
        sequence_to_structure: dict[str, str] = {}
        unique_sequences: list[str] = []
        seen_sequences: set[str] = set()
        for record in pending_records:
            assert record.sequence is not None
            if record.sequence in seen_sequences:
                continue
            seen_sequences.add(record.sequence)
            unique_sequences.append(record.sequence)
        unique_sequences.sort(key=len)

        for sequence_chunk in _chunks(unique_sequences, batch_size):
            predicted_structures = _predict_3di_batch(provider, sequence_chunk)
            if len(predicted_structures) != len(sequence_chunk):
                raise ValueError(
                    "3Di provider returned a different number of predictions than input sequences."
                )
            normalized_predictions: dict[str, str] = {}
            for sequence, predicted_structure in zip(sequence_chunk, predicted_structures, strict=True):
                normalized_structure = normalize_3di_structure(predicted_structure)
                if not normalized_structure:
                    raise ValueError("3Di provider returned an empty or invalid 3Di string.")
                normalized_predictions[sequence] = normalized_structure
                sequence_to_structure[sequence] = normalized_structure
            if cache is not None:
                cache.set_many(model_name=model_name, values=normalized_predictions)
            model_prediction_count += len(sequence_chunk)

        for record in pending_records:
            assert record.payload is not None
            assert record.sequence is not None
            record.payload[field_name] = sequence_to_structure[record.sequence]
            new_3di_count += 1

        for record in buffer:
            if record.raw_line is not None:
                target_handle.write(record.raw_line)
                continue

            assert record.payload is not None
            target_handle.write(json.dumps(record.payload, ensure_ascii=ensure_ascii, separators=(",", ":")))
            target_handle.write("\n")
            written_line_count += 1

        buffer = []
        pending_prediction_count = 0

    try:
        with resolved_input_path.open("r", encoding="utf-8") as source_handle, temporary_output_path.open(
            "w",
            encoding="utf-8",
        ) as target_handle:
            for line_number, raw_line in enumerate(source_handle, start=1):
                if not raw_line.strip():
                    buffer.append(_BufferedRecord(raw_line=raw_line))
                else:
                    total_line_count += 1
                    payload = _parse_json_object_line(raw_line, line_number=line_number, path=resolved_input_path)
                    sequence = extract_instruction_protein_sequence(payload, sequence_field=sequence_field)

                    if skip_existing:
                        existing_3di = usable_instruction_3di(
                            payload,
                            field_name=field_name,
                            sequence=sequence,
                        )
                        if existing_3di:
                            payload[field_name] = existing_3di
                            reused_existing_count += 1
                            buffer.append(_BufferedRecord(payload=payload))
                        elif not sequence:
                            empty_sequence_count += 1
                            buffer.append(_BufferedRecord(payload=payload))
                        else:
                            cached_3di = cache.get(model_name=model_name, sequence=sequence) if cache is not None else None
                            if cached_3di is not None and len(cached_3di) == len(sequence):
                                payload[field_name] = cached_3di
                                cache_hit_count += 1
                                new_3di_count += 1
                                buffer.append(_BufferedRecord(payload=payload))
                            else:
                                buffer.append(
                                    _BufferedRecord(
                                        payload=payload,
                                        sequence=sequence,
                                        needs_prediction=True,
                                    )
                                )
                                pending_prediction_count += 1
                    elif not sequence:
                        empty_sequence_count += 1
                        buffer.append(_BufferedRecord(payload=payload))
                    else:
                        buffer.append(
                            _BufferedRecord(
                                payload=payload,
                                sequence=sequence,
                                needs_prediction=True,
                            )
                        )
                        pending_prediction_count += 1

                if pending_prediction_count >= batch_size or len(buffer) >= max_buffer_records:
                    flush_buffer(target_handle)

                now = time.time()
                if progress_callback is not None and now - last_report_time >= report_every_seconds:
                    elapsed = now - start_time
                    progress_callback(
                        " ".join(
                            (
                                f"[3Di] {resolved_input_path.name}:",
                                f"{written_line_count:,}/{total_line_count:,} records written,",
                                f"{new_3di_count:,} new,",
                                f"{reused_existing_count:,} existing,",
                                f"{cache_hit_count:,} cache hits,",
                                f"{elapsed:.0f}s elapsed",
                            )
                        )
                    )
                    last_report_time = now

            flush_buffer(target_handle)

        temporary_output_path.replace(resolved_output_path)
    finally:
        temporary_output_path.unlink(missing_ok=True)
        if cache is not None:
            cache.close()

    elapsed_seconds = round(time.time() - start_time, 3)
    return Instruction3DiUpdateSummary(
        input_path=str(resolved_input_path),
        output_path=str(resolved_output_path),
        field_name=field_name,
        model_name=model_name,
        total_line_count=total_line_count,
        written_line_count=written_line_count,
        new_3di_count=new_3di_count,
        reused_existing_count=reused_existing_count,
        cache_hit_count=cache_hit_count,
        model_prediction_count=model_prediction_count,
        empty_sequence_count=empty_sequence_count,
        elapsed_seconds=elapsed_seconds,
    )


def annotate_s3_instruction_jsonl_3di(
    *,
    provider,
    prefix_uri: str | None = None,
    part_uris: Sequence[str] | None = None,
    output_prefix_uri: str | None = None,
    field_name: str = DEFAULT_3DI_FIELD,
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


def extract_instruction_protein_sequence(
    payload: Mapping[str, object],
    *,
    sequence_field: str = "output",
) -> str:
    return compact_protein_sequence(str(payload.get(sequence_field) or ""))


def normalize_prostt5_aa_sequence(sequence: str) -> str:
    normalized = RARE_AMINO_ACID_PATTERN.sub("X", compact_protein_sequence(sequence))
    if not normalized:
        raise ValueError("Protein sequence is empty after normalization.")
    return "".join(residue if residue in VALID_PROTEIN_AMINO_ACIDS else "X" for residue in normalized)


def normalize_3di_structure(value: object) -> str:
    normalized = "".join(str(value or "").split()).lower()
    if not normalized:
        return ""
    if any(token not in PROSTT5_3DI_TOKENS for token in normalized):
        return ""
    return normalized


def usable_instruction_3di(
    payload: Mapping[str, object],
    *,
    field_name: str = DEFAULT_3DI_FIELD,
    sequence: str = "",
) -> str:
    structure_3di = normalize_3di_structure(payload.get(field_name))
    if not structure_3di:
        return ""
    if sequence and len(structure_3di) != len(sequence):
        return ""
    return structure_3di


def _predict_3di_batch(provider, sequences: Sequence[str]) -> tuple[str, ...]:
    predict_batch = getattr(provider, "predict_3di_batch", None)
    if callable(predict_batch):
        return tuple(str(value) for value in predict_batch(sequences))

    predict = getattr(provider, "predict", None)
    if not callable(predict):
        raise TypeError("provider must define predict_3di_batch(sequences) or predict(sequence).")

    structures: list[str] = []
    for sequence in sequences:
        prediction = predict(sequence)
        if isinstance(prediction, StructurePrediction):
            if prediction.structure_3di is None:
                raise ValueError("Structure provider returned StructurePrediction without structure_3di.")
            structures.append(prediction.structure_3di)
        else:
            structures.append(str(prediction))
    return tuple(structures)


def _default_prostt5_generation_kwargs() -> dict[str, object]:
    return {
        "do_sample": False,
        "num_beams": 3,
        "repetition_penalty": 1.2,
    }


def _parse_json_object_line(raw_line: str, *, line_number: int, path: Path) -> MutableMapping[str, object]:
    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path} line {line_number}.") from exc
    if not isinstance(payload, MutableMapping):
        raise ValueError(f"{path} line {line_number} must contain a JSON object.")
    return payload


def _ensure_provider_ready(provider) -> None:
    """Eagerly load the model and run a tiny test prediction to fail fast before any data download."""
    load_fn = getattr(provider, "_load", None)
    if callable(load_fn):
        load_fn()
    else:
        _predict_3di_batch(provider, ["ACDEFGHIKLMNPQRSTVWY"])


def _provider_model_name(provider) -> str:
    return str(getattr(provider, "model_name", provider.__class__.__name__))


def _sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _chunks(values: Sequence[str], size: int) -> Sequence[Sequence[str]]:
    return tuple(values[index : index + size] for index in range(0, len(values), size))


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

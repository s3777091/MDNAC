from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from libs.core.pretrain.profiled import MDCProfileSequenceRecord


RESERVED_TRAIN_TOKENS = (
    "<|profile|>",
    "<|sep|>",
    "<|endoftext|>",
    "<|protein|>",
)
DEFAULT_INSTRUCTION_FIELD = "instruction"
DEFAULT_INPUT_FIELD = "input"
DEFAULT_OUTPUT_FIELD = "output"


@dataclass(frozen=True)
class InstructionJsonlAudit:
    paths: tuple[str, ...]
    rows_seen: int
    valid_rows: int
    empty_rows: int
    invalid_json_rows: int
    non_object_rows: int
    missing_instruction_rows: int
    missing_output_rows: int
    sequence_type_counts: dict[str, int]
    field_counts: dict[str, int]
    output_format_counts: dict[str, int]
    preview: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": list(self.paths),
            "rows_seen": self.rows_seen,
            "valid_rows": self.valid_rows,
            "empty_rows": self.empty_rows,
            "invalid_json_rows": self.invalid_json_rows,
            "non_object_rows": self.non_object_rows,
            "missing_instruction_rows": self.missing_instruction_rows,
            "missing_output_rows": self.missing_output_rows,
            "sequence_type_counts": dict(self.sequence_type_counts),
            "field_counts": dict(self.field_counts),
            "output_format_counts": dict(self.output_format_counts),
            "preview": list(self.preview),
        }


def resolve_instruction_paths(paths: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
    if isinstance(paths, (str, Path)):
        raw_paths: Sequence[str | Path] = (paths,)
    else:
        raw_paths = tuple(paths)

    resolved: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.jsonl")))
        else:
            resolved.append(path)

    if not resolved:
        raise ValueError("At least one instruction JSONL path is required.")
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Instruction JSONL file not found: {missing[0]}")
    return tuple(path.resolve() for path in resolved)


def clean_profile_text(value: Any) -> str:
    cleaned = str(value or "")
    for token in RESERVED_TRAIN_TOKENS:
        cleaned = cleaned.replace(token, " ")
    cleaned = (
        cleaned.replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace(";", ",")
        .strip()
    )
    return " ".join(cleaned.split())


def compact_sequence_text(value: Any) -> str:
    return "".join(str(value or "").split()).upper()


def instruction_record_from_payload(
    payload: Mapping[str, Any],
    *,
    default_sequence_type: str = "protein",
    instruction_field: str = DEFAULT_INSTRUCTION_FIELD,
    input_field: str = DEFAULT_INPUT_FIELD,
    output_field: str = DEFAULT_OUTPUT_FIELD,
) -> MDCProfileSequenceRecord:
    instruction = clean_profile_text(payload.get(instruction_field))
    input_text = clean_profile_text(payload.get(input_field))
    sequence = compact_sequence_text(payload.get(output_field))
    if not instruction:
        raise ValueError(f"Instruction row is missing '{instruction_field}'.")
    if not sequence:
        raise ValueError(f"Instruction row is missing '{output_field}'.")

    metadata_payload = payload.get("metadata")
    metadata: dict[str, object] = (
        {str(key): value for key, value in metadata_payload.items()}
        if isinstance(metadata_payload, Mapping)
        else {}
    )
    for key, value in payload.items():
        if key in {instruction_field, input_field, output_field, "metadata"}:
            continue
        metadata.setdefault(str(key), value)

    sequence_type = str(
        payload.get("sequence_type")
        or metadata.get("sequence_type")
        or default_sequence_type
    ).strip().lower()
    profile = instruction if not input_text else f"{instruction}; input {input_text}"
    return MDCProfileSequenceRecord(
        profile=profile,
        sequence=sequence,
        sequence_type=sequence_type,
        metadata=metadata,
    )


def iter_instruction_records(
    paths: str | Path | Sequence[str | Path],
    *,
    default_sequence_type: str = "protein",
    instruction_field: str = DEFAULT_INSTRUCTION_FIELD,
    input_field: str = DEFAULT_INPUT_FIELD,
    output_field: str = DEFAULT_OUTPUT_FIELD,
) -> Iterable[MDCProfileSequenceRecord]:
    for path in resolve_instruction_paths(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}.") from exc
                if not isinstance(payload, Mapping):
                    raise ValueError(f"Instruction JSONL row must be an object at {path}:{line_number}.")
                try:
                    yield instruction_record_from_payload(
                        payload,
                        default_sequence_type=default_sequence_type,
                        instruction_field=instruction_field,
                        input_field=input_field,
                        output_field=output_field,
                    )
                except ValueError as exc:
                    raise ValueError(f"{exc} Source: {path}:{line_number}.") from exc


def instruction_split_key(
    payload: Mapping[str, Any],
    *,
    fallback_key: str,
) -> str:
    accession = str(payload.get("accession") or payload.get("id") or "").strip()
    if accession:
        return f"accession:{accession}"
    selected = {
        key: payload.get(key)
        for key in (
            "instruction",
            "input",
            "output",
            "description",
            "organism",
        )
        if payload.get(key) not in (None, "", [])
    }
    if not selected:
        return fallback_key
    return json.dumps(selected, ensure_ascii=False, sort_keys=True)


def belongs_to_split(
    payload: Mapping[str, Any],
    *,
    split: str,
    train_ratio: float,
    split_seed: int,
    fallback_key: str,
) -> bool:
    if split not in {"train", "val"}:
        raise ValueError("split must be one of: 'train', 'val'.")
    key = instruction_split_key(payload, fallback_key=fallback_key)
    digest = hashlib.sha1(f"{split_seed}:{key}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return bucket < train_ratio if split == "train" else bucket >= train_ratio


def audit_instruction_jsonl(
    paths: str | Path | Sequence[str | Path],
    *,
    default_sequence_type: str = "protein",
    instruction_field: str = DEFAULT_INSTRUCTION_FIELD,
    input_field: str = DEFAULT_INPUT_FIELD,
    output_field: str = DEFAULT_OUTPUT_FIELD,
    preview_rows: int = 3,
) -> InstructionJsonlAudit:
    resolved_paths = resolve_instruction_paths(paths)
    rows_seen = 0
    valid_rows = 0
    empty_rows = 0
    invalid_json_rows = 0
    non_object_rows = 0
    missing_instruction_rows = 0
    missing_output_rows = 0
    sequence_type_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    output_format_counts: Counter[str] = Counter()
    preview: list[dict[str, Any]] = []

    for path in resolved_paths:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    empty_rows += 1
                    continue
                rows_seen += 1
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    invalid_json_rows += 1
                    continue
                if not isinstance(payload, Mapping):
                    non_object_rows += 1
                    continue
                field_counts.update(str(key) for key in payload.keys())

                instruction = clean_profile_text(payload.get(instruction_field))
                sequence = compact_sequence_text(payload.get(output_field))
                if not instruction:
                    missing_instruction_rows += 1
                if not sequence:
                    missing_output_rows += 1
                if not instruction or not sequence:
                    continue

                metadata = payload.get("metadata")
                sequence_type = str(
                    payload.get("sequence_type")
                    or (metadata.get("sequence_type") if isinstance(metadata, Mapping) else None)
                    or default_sequence_type
                ).strip().lower()
                sequence_type_counts[sequence_type] += 1
                output_format_counts[str(payload.get("output_format") or "").strip() or "unspecified"] += 1
                valid_rows += 1
                if len(preview) < preview_rows:
                    preview.append(
                        {
                            "instruction": instruction[:240],
                            "input": clean_profile_text(payload.get(input_field))[:240],
                            "output_prefix": sequence[:80],
                            "output_length": len(sequence),
                            "sequence_type": sequence_type,
                        }
                    )

    return InstructionJsonlAudit(
        paths=tuple(str(path) for path in resolved_paths),
        rows_seen=rows_seen,
        valid_rows=valid_rows,
        empty_rows=empty_rows,
        invalid_json_rows=invalid_json_rows,
        non_object_rows=non_object_rows,
        missing_instruction_rows=missing_instruction_rows,
        missing_output_rows=missing_output_rows,
        sequence_type_counts=dict(sequence_type_counts),
        field_counts=dict(field_counts),
        output_format_counts=dict(output_format_counts),
        preview=tuple(preview),
    )

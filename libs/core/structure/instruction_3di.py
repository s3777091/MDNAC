"""Local file-based 3Di annotation pipeline and normalization utilities."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

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


@dataclass(slots=True)
class _BufferedRecord:
    payload: MutableMapping[str, object] | None = None
    raw_line: str | None = None
    sequence: str | None = None
    needs_prediction: bool = False


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
    from .cache import Sequence3DiCache

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
                if normalized_structure:
                    normalized_predictions[sequence] = normalized_structure
                    sequence_to_structure[sequence] = normalized_structure
            if cache is not None and normalized_predictions:
                cache.set_many(model_name=model_name, values=normalized_predictions)
            model_prediction_count += len(sequence_chunk)

        for record in pending_records:
            assert record.payload is not None
            assert record.sequence is not None
            structure_value = sequence_to_structure.get(record.sequence)
            if structure_value:
                record.payload[field_name] = structure_value
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


# --- Normalization utilities ---


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


# --- Internal helpers ---


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


def _parse_json_object_line(raw_line: str, *, line_number: int, path: Path) -> MutableMapping[str, object]:
    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path} line {line_number}.") from exc
    if not isinstance(payload, MutableMapping):
        raise ValueError(f"{path} line {line_number} must contain a JSON object.")
    return payload


def _ensure_provider_ready(provider) -> None:
    """Eagerly load the model and run a tiny test prediction to fail fast."""
    load_fn = getattr(provider, "_load", None)
    if callable(load_fn):
        load_fn()
    else:
        _predict_3di_batch(provider, ["ACDEFGHIKLMNPQRSTVWY"])


def _provider_model_name(provider) -> str:
    return str(getattr(provider, "model_name", provider.__class__.__name__))


def _chunks(values: Sequence[str], size: int) -> Sequence[Sequence[str]]:
    return tuple(values[index : index + size] for index in range(0, len(values), size))

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .masking import is_standard_amino_acid_text


MASK_TOKEN_RE = re.compile(r"<MASK_(\d+)>")


def validate_span_completion_row(
    row: Mapping[str, Any],
    *,
    original_sequence: str | None = None,
) -> dict[str, Any]:
    instruction = str(row.get("instruction") or "").strip()
    input_text = str(row.get("input") or "").strip()
    output = _compact_sequence(row.get("output"))
    metadata = row.get("metadata")
    if not instruction:
        raise ValueError("row instruction must not be empty.")
    if not input_text:
        raise ValueError("row input must not be empty for span completion.")
    if not output:
        raise ValueError("row output must not be empty.")
    if not is_standard_amino_acid_text(output):
        raise ValueError("row output must contain only standard amino acids.")
    if not isinstance(metadata, Mapping):
        raise ValueError("row metadata must be a mapping.")

    mask_start = int(metadata["mask_start"])
    mask_end = int(metadata["mask_end"])
    mask_length = int(metadata["mask_length"])
    if mask_end - mask_start != mask_length:
        raise ValueError("metadata mask_end - mask_start must equal mask_length.")
    if len(output) != mask_length:
        raise ValueError("len(output) must equal metadata mask_length.")

    mask_tokens = MASK_TOKEN_RE.findall(input_text)
    if len(mask_tokens) != 1:
        raise ValueError("input must contain exactly one <MASK_N> token.")
    if int(mask_tokens[0]) != mask_length:
        raise ValueError("<MASK_N> length must equal len(output).")

    left_flank = _extract_input_field(input_text, "left_flank")
    right_flank = _extract_input_field(input_text, "right_flank")
    partial_sequence = _extract_input_field(input_text, "partial_sequence")
    expected_partial = f"{left_flank}<MASK_{mask_length}>{right_flank}"
    if partial_sequence != expected_partial:
        raise ValueError("partial_sequence must equal left_flank + <MASK_N> + right_flank.")
    if output in f"{instruction}; input {input_text}":
        raise ValueError("output span appears in instruction/input prompt text.")

    if original_sequence is not None:
        normalized_original = _compact_sequence(original_sequence)
        if not 0 <= mask_start < mask_end <= len(normalized_original):
            raise ValueError("mask span is outside the original sequence.")
        if output != normalized_original[mask_start:mask_end]:
            raise ValueError("output does not match original_sequence[mask_start:mask_end].")
        if not normalized_original[:mask_start].endswith(left_flank):
            raise ValueError("left_flank does not match the original sequence.")
        if not normalized_original[mask_end:].startswith(right_flank):
            raise ValueError("right_flank does not match the original sequence.")

    return {
        "mask_start": mask_start,
        "mask_end": mask_end,
        "mask_length": mask_length,
        "left_flank_length": len(left_flank),
        "right_flank_length": len(right_flank),
    }


def validate_jsonl_file(path: Path | str) -> dict[str, Any]:
    resolved_path = Path(path)
    rows_seen = 0
    valid_rows = 0
    mask_length_counts: Counter[int] = Counter()
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            rows_seen += 1
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {resolved_path}:{line_number}.") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row must be an object at {resolved_path}:{line_number}.")
            try:
                validation = validate_span_completion_row(row)
            except ValueError as exc:
                raise ValueError(f"{exc} Source: {resolved_path}:{line_number}.") from exc
            valid_rows += 1
            mask_length_counts[int(validation["mask_length"])] += 1

    return {
        "path": str(resolved_path),
        "rows_seen": rows_seen,
        "valid_rows": valid_rows,
        "mask_length_distribution": dict(sorted(mask_length_counts.items())),
    }


def _extract_input_field(input_text: str, field_name: str) -> str:
    prefix = f"{field_name} "
    for part in input_text.split("; "):
        if part.startswith(prefix):
            return part[len(prefix) :].strip()
    raise ValueError(f"input is missing '{field_name}'.")


def _compact_sequence(value: Any) -> str:
    return "".join(str(value or "").split()).upper()

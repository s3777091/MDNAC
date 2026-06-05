from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from typing import Any


STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
MASK_POLICY_RANDOM_SPAN = "random_span"
MASK_POLICY_N_TERMINAL = "n_terminal_span"
MASK_POLICY_C_TERMINAL = "c_terminal_span"
SUPPORTED_MASK_POLICIES = (
    MASK_POLICY_RANDOM_SPAN,
    MASK_POLICY_N_TERMINAL,
    MASK_POLICY_C_TERMINAL,
)


def choose_random_span(
    sequence: str,
    *,
    min_mask_length: int,
    max_mask_length: int,
    rng: random.Random,
) -> tuple[int, int]:
    normalized_sequence = _compact_sequence(sequence)
    mask_length = _choose_mask_length(
        len(normalized_sequence),
        min_mask_length=min_mask_length,
        max_mask_length=max_mask_length,
        rng=rng,
    )
    mask_start = rng.randint(0, len(normalized_sequence) - mask_length)
    return mask_start, mask_start + mask_length


def choose_n_terminal_span(
    sequence: str,
    *,
    min_mask_length: int,
    max_mask_length: int,
    rng: random.Random,
) -> tuple[int, int]:
    normalized_sequence = _compact_sequence(sequence)
    mask_length = _choose_mask_length(
        len(normalized_sequence),
        min_mask_length=min_mask_length,
        max_mask_length=max_mask_length,
        rng=rng,
    )
    return 0, mask_length


def choose_c_terminal_span(
    sequence: str,
    *,
    min_mask_length: int,
    max_mask_length: int,
    rng: random.Random,
) -> tuple[int, int]:
    normalized_sequence = _compact_sequence(sequence)
    mask_length = _choose_mask_length(
        len(normalized_sequence),
        min_mask_length=min_mask_length,
        max_mask_length=max_mask_length,
        rng=rng,
    )
    return len(normalized_sequence) - mask_length, len(normalized_sequence)


def make_span_completion_example(
    source_row: Mapping[str, Any],
    *,
    source_index: int,
    mask_start: int,
    mask_end: int,
    mask_policy: str,
    left_flank_size: int = 64,
    right_flank_size: int = 64,
    instruction_prefix: str = "task protein span completion",
) -> dict[str, Any]:
    if mask_policy not in SUPPORTED_MASK_POLICIES:
        raise ValueError(f"Unsupported mask_policy: {mask_policy!r}")
    if left_flank_size < 0 or right_flank_size < 0:
        raise ValueError("flank sizes must be non-negative.")

    sequence = _compact_sequence(source_row.get("output"))
    if not sequence:
        raise ValueError("source row output must contain a full protein sequence.")
    if not 0 <= mask_start < mask_end <= len(sequence):
        raise ValueError("mask_start/mask_end must define a non-empty span inside the source sequence.")

    missing_span = sequence[mask_start:mask_end]
    if not is_standard_amino_acid_text(missing_span):
        raise ValueError("masked output span must contain only standard amino acids.")

    mask_length = mask_end - mask_start
    mask_token = f"<MASK_{mask_length}>"
    left_start = max(0, mask_start - left_flank_size)
    right_end = min(len(sequence), mask_end + right_flank_size)
    left_flank = sequence[left_start:mask_start]
    right_flank = sequence[mask_end:right_end]
    partial_sequence = f"{left_flank}{mask_token}{right_flank}"

    instruction = _build_instruction_text(source_row.get("instruction"), instruction_prefix=instruction_prefix)
    input_text = (
        f"mask_policy {mask_policy}; "
        f"mask_start {mask_start}; "
        f"mask_length {mask_length}; "
        f"left_flank {left_flank}; "
        f"right_flank {right_flank}; "
        f"partial_sequence {partial_sequence}"
    )
    prompt_text = f"{instruction}; input {input_text}"
    if missing_span and missing_span in prompt_text:
        raise ValueError("masked output span appears in the prompt/context.")

    metadata = _copy_metadata(source_row)
    metadata.update(
        {
            "source_index": int(source_index),
            "source_length": len(sequence),
            "mask_start": int(mask_start),
            "mask_end": int(mask_end),
            "mask_length": int(mask_length),
            "mask_policy": mask_policy,
        }
    )
    original_output_format = source_row.get("output_format")
    if original_output_format not in (None, ""):
        metadata.setdefault("source_output_format", original_output_format)

    span_row: dict[str, Any] = {
        "instruction": instruction,
        "input": input_text,
        "output": missing_span,
        "metadata": metadata,
    }
    for key, value in source_row.items():
        if key in {"instruction", "input", "output", "metadata", "output_format"}:
            continue
        span_row.setdefault(str(key), copy.deepcopy(value))
    span_row["output_format"] = "protein missing span"
    return span_row


def is_standard_amino_acid_text(value: str) -> bool:
    return bool(value) and all(character in STANDARD_AMINO_ACIDS for character in value)


def _compact_sequence(value: Any) -> str:
    return "".join(str(value or "").split()).upper()


def _choose_mask_length(
    sequence_length: int,
    *,
    min_mask_length: int,
    max_mask_length: int,
    rng: random.Random,
) -> int:
    if min_mask_length <= 0:
        raise ValueError("min_mask_length must be positive.")
    if max_mask_length < min_mask_length:
        raise ValueError("max_mask_length must be greater than or equal to min_mask_length.")
    if sequence_length < min_mask_length:
        raise ValueError("sequence is shorter than min_mask_length.")

    effective_max = min(max_mask_length, sequence_length)
    return rng.randint(min_mask_length, effective_max)


def _build_instruction_text(value: Any, *, instruction_prefix: str) -> str:
    source_instruction = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    if not source_instruction:
        raise ValueError("source row instruction must not be empty.")

    normalized_source = source_instruction.strip()
    normalized_prefix = instruction_prefix.strip()
    if normalized_source.lower().startswith(normalized_prefix.lower()):
        return normalized_source
    return f"{normalized_prefix}; {normalized_source}"


def _copy_metadata(source_row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = source_row.get("metadata")
    if isinstance(metadata, Mapping):
        return {str(key): copy.deepcopy(value) for key, value in metadata.items()}
    return {}

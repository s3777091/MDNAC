from __future__ import annotations

import copy
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .masking import (
    MASK_POLICY_C_TERMINAL,
    MASK_POLICY_N_TERMINAL,
    MASK_POLICY_RANDOM_SPAN,
    SUPPORTED_MASK_POLICIES,
    choose_c_terminal_span,
    choose_n_terminal_span,
    choose_random_span,
    make_span_completion_example,
)
from .validation import validate_span_completion_row


DEFAULT_MASK_POLICIES = (
    MASK_POLICY_RANDOM_SPAN,
    MASK_POLICY_N_TERMINAL,
    MASK_POLICY_C_TERMINAL,
)


def convert_instruction_row_to_span_examples(
    source_row: Mapping[str, Any],
    *,
    source_index: int,
    examples_per_sequence: int = 4,
    min_sequence_length: int = 64,
    max_sequence_length: int = 1024,
    min_mask_length: int = 8,
    max_mask_length: int = 64,
    left_flank_size: int = 64,
    right_flank_size: int = 64,
    mask_policies: Sequence[str] = DEFAULT_MASK_POLICIES,
    rng: random.Random | None = None,
    max_attempts_per_example: int = 32,
) -> list[dict[str, Any]]:
    if examples_per_sequence <= 0:
        raise ValueError("examples_per_sequence must be positive.")
    if min_sequence_length <= 0:
        raise ValueError("min_sequence_length must be positive.")
    if max_sequence_length < min_sequence_length:
        raise ValueError("max_sequence_length must be greater than or equal to min_sequence_length.")
    if not mask_policies:
        raise ValueError("mask_policies must not be empty.")
    unsupported = [policy for policy in mask_policies if policy not in SUPPORTED_MASK_POLICIES]
    if unsupported:
        raise ValueError(f"Unsupported mask policies: {unsupported}")

    sequence = _compact_sequence(source_row.get("output"))
    if len(sequence) < min_sequence_length:
        return []
    if len(sequence) > max_sequence_length:
        return []

    resolved_rng = rng or random.Random()
    generated: list[dict[str, Any]] = []
    used_spans: set[tuple[int, int, str]] = set()

    for example_index in range(examples_per_sequence):
        policy = str(mask_policies[example_index % len(mask_policies)])
        for _ in range(max_attempts_per_example):
            mask_start, mask_end = _choose_span(
                sequence,
                policy=policy,
                min_mask_length=min_mask_length,
                max_mask_length=max_mask_length,
                rng=resolved_rng,
            )
            span_key = (mask_start, mask_end, policy)
            if span_key in used_spans:
                continue
            try:
                row = make_span_completion_example(
                    source_row,
                    source_index=source_index,
                    mask_start=mask_start,
                    mask_end=mask_end,
                    mask_policy=policy,
                    left_flank_size=left_flank_size,
                    right_flank_size=right_flank_size,
                )
                validate_span_completion_row(row, original_sequence=sequence)
            except ValueError:
                continue
            generated.append(row)
            used_spans.add(span_key)
            break

    return generated


def convert_instruction_jsonl_to_span_jsonl(
    source_path: Path | str,
    output_path: Path | str,
    *,
    stats_path: Path | str | None = None,
    examples_per_sequence: int = 4,
    min_sequence_length: int = 64,
    max_sequence_length: int = 1024,
    min_mask_length: int = 8,
    max_mask_length: int = 64,
    left_flank_size: int = 64,
    right_flank_size: int = 64,
    seed: int = 42,
    mask_policies: Sequence[str] = DEFAULT_MASK_POLICIES,
) -> dict[str, Any]:
    resolved_source_path = Path(source_path)
    resolved_output_path = Path(output_path)
    resolved_stats_path = Path(stats_path) if stats_path is not None else resolved_output_path.with_name("stats.json")
    if not resolved_source_path.is_file():
        raise FileNotFoundError(f"Source instruction JSONL not found: {resolved_source_path}")

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_stats_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    source_rows = 0
    accepted_source_rows = 0
    skipped_rows = 0
    generated_span_rows = 0
    invalid_json_rows = 0
    non_object_rows = 0
    skip_reasons: Counter[str] = Counter()
    mask_length_distribution: Counter[int] = Counter()
    example_before: dict[str, Any] | None = None
    example_after: dict[str, Any] | None = None

    with resolved_source_path.open("r", encoding="utf-8") as source_handle, resolved_output_path.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        for line_number, raw_line in enumerate(source_handle, start=1):
            if not raw_line.strip():
                continue
            source_rows += 1
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                invalid_json_rows += 1
                skipped_rows += 1
                skip_reasons["invalid_json"] += 1
                continue
            if not isinstance(payload, Mapping):
                non_object_rows += 1
                skipped_rows += 1
                skip_reasons["non_object"] += 1
                continue

            source_snapshot = copy.deepcopy(payload)
            sequence = _compact_sequence(payload.get("output"))
            if not sequence:
                skipped_rows += 1
                skip_reasons["empty_output"] += 1
                continue
            if len(sequence) < min_sequence_length:
                skipped_rows += 1
                skip_reasons["too_short"] += 1
                continue
            if len(sequence) > max_sequence_length:
                skipped_rows += 1
                skip_reasons["too_long"] += 1
                continue

            examples = convert_instruction_row_to_span_examples(
                payload,
                source_index=source_rows - 1,
                examples_per_sequence=examples_per_sequence,
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                min_mask_length=min_mask_length,
                max_mask_length=max_mask_length,
                left_flank_size=left_flank_size,
                right_flank_size=right_flank_size,
                mask_policies=mask_policies,
                rng=rng,
            )
            if payload != source_snapshot:
                raise AssertionError(f"source row was modified in-place at line {line_number}.")
            if not examples:
                skipped_rows += 1
                skip_reasons["no_valid_span"] += 1
                continue

            accepted_source_rows += 1
            if example_before is None:
                example_before = source_snapshot
                example_after = copy.deepcopy(examples[0])

            for example in examples:
                validation = validate_span_completion_row(example, original_sequence=sequence)
                mask_length_distribution[int(validation["mask_length"])] += 1
                output_handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                generated_span_rows += 1

    stats = {
        "source_path": str(resolved_source_path),
        "output_path": str(resolved_output_path),
        "source_rows": source_rows,
        "accepted_source_rows": accepted_source_rows,
        "skipped_rows": skipped_rows,
        "generated_span_rows": generated_span_rows,
        "invalid_json_rows": invalid_json_rows,
        "non_object_rows": non_object_rows,
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "mask_length_distribution": {
            str(length): count for length, count in sorted(mask_length_distribution.items())
        },
        "parameters": {
            "examples_per_sequence": examples_per_sequence,
            "min_sequence_length": min_sequence_length,
            "max_sequence_length": max_sequence_length,
            "min_mask_length": min_mask_length,
            "max_mask_length": max_mask_length,
            "left_flank_size": left_flank_size,
            "right_flank_size": right_flank_size,
            "seed": seed,
            "mask_policies": list(mask_policies),
        },
        "example_before": example_before,
        "example_after": example_after,
    }
    resolved_stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def _choose_span(
    sequence: str,
    *,
    policy: str,
    min_mask_length: int,
    max_mask_length: int,
    rng: random.Random,
) -> tuple[int, int]:
    if policy == MASK_POLICY_RANDOM_SPAN:
        return choose_random_span(
            sequence,
            min_mask_length=min_mask_length,
            max_mask_length=max_mask_length,
            rng=rng,
        )
    if policy == MASK_POLICY_N_TERMINAL:
        return choose_n_terminal_span(
            sequence,
            min_mask_length=min_mask_length,
            max_mask_length=max_mask_length,
            rng=rng,
        )
    if policy == MASK_POLICY_C_TERMINAL:
        return choose_c_terminal_span(
            sequence,
            min_mask_length=min_mask_length,
            max_mask_length=max_mask_length,
            rng=rng,
        )
    raise ValueError(f"Unsupported mask policy: {policy!r}")


def _compact_sequence(value: Any) -> str:
    return "".join(str(value or "").split()).upper()

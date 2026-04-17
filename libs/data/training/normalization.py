from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from libs.data.entities import SequenceRecord


InvalidBasePolicy = Literal["replace_with_x", "replace_with_n", "drop"]


@dataclass(slots=True, frozen=True)
class SequenceNormalizationConfig:
    sequence_type: Literal["protein"] = "protein"
    min_length: int = 0
    max_length: int | None = None
    invalid_base_policy: InvalidBasePolicy = "replace_with_x"
    max_ambiguous_ratio: float = 1.0
    deduplicate_sequences: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence_type", "protein")
        if self.invalid_base_policy == "replace_with_n":
            object.__setattr__(self, "invalid_base_policy", "replace_with_x")
        if self.min_length < 0:
            raise ValueError("min_length must be >= 0")
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError("max_length must be greater than 0 when provided")
        if self.max_length is not None and self.max_length < self.min_length:
            raise ValueError("max_length must be >= min_length")
        if not 0.0 <= self.max_ambiguous_ratio <= 1.0:
            raise ValueError("max_ambiguous_ratio must be between 0.0 and 1.0")


@dataclass(slots=True, frozen=True)
class NormalizationReport:
    kept_count: int
    dropped_count: int
    dropped_reasons: dict[str, int] = field(default_factory=dict)


def normalize_records(
    records: list[SequenceRecord],
    config: SequenceNormalizationConfig,
) -> tuple[list[SequenceRecord], NormalizationReport]:
    normalized_records: list[SequenceRecord] = []
    dropped_reasons: dict[str, int] = {}
    seen_sequences: set[str] = set()

    for record in records:
        normalized_sequence = _normalize_sequence(record.sequence, config)
        if normalized_sequence is None:
            _bump_reason(dropped_reasons, "invalid_sequence")
            continue

        if len(normalized_sequence) < config.min_length:
            _bump_reason(dropped_reasons, "too_short")
            continue

        if config.max_length is not None and len(normalized_sequence) > config.max_length:
            _bump_reason(dropped_reasons, "too_long")
            continue

        ambiguous_ratio = normalized_sequence.count("X") / len(normalized_sequence)
        if ambiguous_ratio > config.max_ambiguous_ratio:
            _bump_reason(dropped_reasons, "too_ambiguous")
            continue

        if config.deduplicate_sequences and normalized_sequence in seen_sequences:
            _bump_reason(dropped_reasons, "duplicate_sequence")
            continue

        seen_sequences.add(normalized_sequence)
        metadata = dict(record.metadata)
        metadata["sequence_type"] = config.sequence_type

        normalized_records.append(
            SequenceRecord(
                accession=record.accession,
                source_name=record.source_name,
                description=record.description,
                organism=record.organism,
                sequence=normalized_sequence,
                sequence_length=len(normalized_sequence),
                sequence_version=record.sequence_version,
                metadata=metadata,
            )
        )

    report = NormalizationReport(
        kept_count=len(normalized_records),
        dropped_count=sum(dropped_reasons.values()),
        dropped_reasons=dropped_reasons,
    )
    return normalized_records, report


def _normalize_sequence(sequence: str, config: SequenceNormalizationConfig) -> str | None:
    compact = re.sub(r"\s+", "", sequence).upper()
    if not compact:
        return None

    valid_bases = frozenset("ACDEFGHIKLMNPQRSTVWYX")
    unknown_char = "X"

    normalized: list[str] = []
    for base in compact:
        if base in valid_bases:
            normalized.append(base)
            continue

        if config.invalid_base_policy == "drop":
            return None
        normalized.append(unknown_char)

    return "".join(normalized)


def _bump_reason(dropped_reasons: dict[str, int], reason: str) -> None:
    dropped_reasons[reason] = dropped_reasons.get(reason, 0) + 1

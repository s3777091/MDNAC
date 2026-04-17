from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class IncrementalFetchPlan:
    new_accessions: tuple[str, ...]
    updated_accessions: tuple[str, ...]
    unchanged_accessions: tuple[str, ...]
    accessions_to_fetch: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class RebuildResult:
    train_text: str
    record_count: int
    dropped_count: int
    dropped_reasons: dict[str, int]

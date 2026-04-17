from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence


MergeStrategy = Literal["replace", "skip_duplicates", "upsert"]
StorageMode = Literal["local", "minio"]
SequenceType = Literal["protein"]


def _normalize_accessions(accessions: Sequence[str] | None) -> tuple[str, ...]:
    if not accessions:
        return ()
    cleaned = []
    for accession in accessions:
        value = accession.strip()
        if value:
            cleaned.append(value)
    return tuple(cleaned)


def _normalize_fields(fields: Sequence[str] | None) -> tuple[str, ...]:
    if not fields:
        return ()
    cleaned = []
    for field_name in fields:
        value = field_name.strip()
        if value:
            cleaned.append(value)
    return tuple(cleaned)


def _slugify(value: str) -> str:
    stripped = value.strip().lower()
    if not stripped:
        raise ValueError("dataset_name must not be empty")

    parts: list[str] = []
    last_was_separator = False
    for character in stripped:
        if character.isalnum():
            parts.append(character)
            last_was_separator = False
            continue

        if not last_was_separator:
            parts.append("-")
            last_was_separator = True

    slug = "".join(parts).strip("-")
    if not slug:
        raise ValueError("dataset_name must contain at least one alphanumeric character")
    return slug


@dataclass(slots=True, frozen=True)
class FetchRequest:
    dataset_name: str
    query: str | None = None
    accessions: tuple[str, ...] = field(default_factory=tuple)
    limit: int = 100
    batch_size: int | None = None
    extra_fields: tuple[str, ...] = field(default_factory=tuple)
    include_suppressed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_name", _slugify(self.dataset_name))
        object.__setattr__(self, "accessions", _normalize_accessions(self.accessions))
        object.__setattr__(self, "extra_fields", _normalize_fields(self.extra_fields))

        if self.limit < 0:
            raise ValueError("limit must be >= 0; use 0 for no source-level limit")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0 when provided")
        if not self.query and not self.accessions:
            raise ValueError("Either query or accessions must be provided")

    @property
    def effective_limit(self) -> int | None:
        return None if self.limit == 0 else self.limit


@dataclass(slots=True)
class SequenceRecord:
    accession: str
    source_name: str
    description: str
    organism: str
    sequence: str
    sequence_length: int
    sequence_version: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_fasta(self, line_width: int = 80) -> str:
        header = self.accession
        if self.description:
            header = f"{header} {self.description}"

        lines = [f">{header}"]
        for start_index in range(0, len(self.sequence), line_width):
            lines.append(self.sequence[start_index : start_index + line_width])
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accession": self.accession,
            "source_name": self.source_name,
            "description": self.description,
            "organism": self.organism,
            "sequence": self.sequence,
            "sequence_length": self.sequence_length,
            "sequence_version": self.sequence_version,
            "metadata": dict(self.metadata),
        }

    def to_training_line(self) -> str:
        sequence_type = self.metadata.get("sequence_type", "protein").strip().lower()
        if sequence_type and sequence_type != "protein":
            raise ValueError("SequenceRecord only supports protein sequences in training exports.")
        return f"<|protein|>{self.sequence}<|endoftext|>"


@dataclass(slots=True, frozen=True)
class DatasetArtifact:
    source_name: str
    dataset_name: str
    storage_mode: StorageMode
    snapshot_id: str
    current_location: str
    file_locations: dict[str, str]
    record_count: int
    history_location: str | None = None


@dataclass(slots=True, frozen=True)
class TrainingDatasetArtifact:
    source_name: str
    dataset_name: str
    storage_mode: StorageMode
    snapshot_id: str
    current_location: str
    train_txt_path: str
    tokenizer_map_path: str
    record_count: int
    dropped_record_count: int
    sequence_type: SequenceType
    vocab_size: int
    history_location: str | None = None


@dataclass(slots=True, frozen=True)
class PreparationSessionArtifact:
    source_name: str
    dataset_name: str
    storage_mode: StorageMode
    session_location: str
    manifest_path: str
    train_txt_path: str
    tokenizer_map_path: str | None
    processed_count: int
    total_count: int
    record_count: int
    dropped_record_count: int
    is_complete: bool
    current_location: str | None = None
    snapshot_id: str | None = None


@dataclass(slots=True, frozen=True)
class ManagedDataset:
    source_name: str
    dataset_name: str
    storage_mode: StorageMode
    current_location: str
    record_count: int
    updated_at_utc: str
    snapshot_id: str


@dataclass(slots=True, frozen=True)
class DeleteResult:
    source_name: str
    dataset_name: str
    storage_mode: StorageMode
    deleted: bool
    location: str
    permanent: bool


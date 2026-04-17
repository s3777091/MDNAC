from __future__ import annotations

from dataclasses import asdict

from libs.data.entities import FetchRequest, SequenceRecord
from libs.data.training.normalization import SequenceNormalizationConfig, normalize_records

from .helpers import (
    accession_hash,
    accession_key_for_record,
    bump_reason,
    canonical_accession,
    clean_optional_string,
    has_raw_index_entry,
    sequence_hash,
    utc_now_iso,
    version_from_accession_token,
)
from .models import IncrementalFetchPlan, RebuildResult
from .state import saved_artifact_paths_exist


def resolve_sequence_versions(
    source,
    accessions: tuple[str, ...],
    accession_aliases: dict[str, str],
    raw_index: dict[str, dict[str, object]],
    request: FetchRequest,
) -> dict[str, str | None]:
    resolver = getattr(source, "resolve_sequence_versions", None)
    if not callable(resolver):
        return {}

    existing_accessions = tuple(accession for accession in accessions if accession in raw_index)
    if not existing_accessions:
        return {}

    requested_accessions = tuple(accession_aliases[accession] for accession in existing_accessions)
    resolved_versions = resolver(requested_accessions, request=request)
    version_map: dict[str, str | None] = {}
    for accession, version in dict(resolved_versions).items():
        version_map[canonical_accession(accession)] = clean_optional_string(version)
    return version_map


def build_fetch_plan(
    accessions: tuple[str, ...],
    accession_aliases: dict[str, str],
    raw_index: dict[str, dict[str, object]],
    version_map: dict[str, str | None],
) -> IncrementalFetchPlan:
    new_accessions: list[str] = []
    updated_accessions: list[str] = []
    unchanged_accessions: list[str] = []

    for accession in accessions:
        entry = raw_index.get(accession)
        if not has_raw_index_entry(entry):
            new_accessions.append(accession)
            continue

        stored_version = clean_optional_string(entry.get("sequence_version"))
        current_version = version_map.get(accession) or version_from_accession_token(accession_aliases[accession])
        if current_version is not None and stored_version != current_version:
            updated_accessions.append(accession)
            continue

        unchanged_accessions.append(accession)

    return IncrementalFetchPlan(
        new_accessions=tuple(new_accessions),
        updated_accessions=tuple(updated_accessions),
        unchanged_accessions=tuple(unchanged_accessions),
        accessions_to_fetch=tuple([*new_accessions, *updated_accessions]),
    )


def needs_rebuild(
    manifest: dict[str, object],
    raw_index: dict[str, dict[str, object]],
    accessions: tuple[str, ...],
    normalization: SequenceNormalizationConfig,
    sequence_type: str,
    vocab_size: int | None,
    duplicate_accession_count: int,
    fetch_plan: IncrementalFetchPlan,
    storage_mode: str,
) -> bool:
    if not manifest or not raw_index or not manifest.get("is_complete"):
        return True
    if fetch_plan.accessions_to_fetch:
        return True
    if str(manifest.get("storage_mode", "")) != storage_mode:
        return True
    if str(manifest.get("sequence_type", "")) != sequence_type:
        return True
    if manifest.get("vocab_size") != vocab_size:
        return True
    if manifest.get("normalization") != asdict(normalization):
        return True
    if int(manifest.get("duplicate_accession_count", 0) or 0) != duplicate_accession_count:
        return True
    if str(manifest.get("accession_hash", "")) != accession_hash(accessions):
        return True
    if not saved_artifact_paths_exist(manifest):
        return True
    return False


def raw_index_entry(record: SequenceRecord, requested_accession: str) -> dict[str, object]:
    return {
        "accession": accession_key_for_record(record),
        "source_accession": record.accession,
        "requested_accession": requested_accession,
        "source_name": record.source_name,
        "description": record.description,
        "organism": record.organism,
        "sequence": record.sequence,
        "sequence_length": record.sequence_length,
        "sequence_version": record.sequence_version,
        "metadata": dict(record.metadata),
        "raw_sequence_hash": sequence_hash(record.sequence),
        "fetched_at_utc": utc_now_iso(),
        "included_in_current_dataset": False,
        "current_dataset_reason": "",
        "duplicate_of_accession": None,
        "normalized_sequence": "",
        "normalized_sequence_hash": "",
        "training_line": "",
        "rebuilt_at_utc": "",
    }


def record_from_raw_index(entry: dict[str, object], accession: str) -> SequenceRecord:
    sequence = str(entry.get("sequence", ""))
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return SequenceRecord(
        accession=str(entry.get("source_accession") or accession),
        source_name=str(entry.get("source_name", "")),
        description=str(entry.get("description", "")),
        organism=str(entry.get("organism", "")),
        sequence=sequence,
        sequence_length=int(entry.get("sequence_length", len(sequence)) or len(sequence)),
        sequence_version=clean_optional_string(entry.get("sequence_version")),
        metadata={str(key): str(value) for key, value in metadata.items()},
    )


def mark_entry(
    entry: dict[str, object],
    included: bool,
    reason: str,
    rebuilt_at_utc: str,
    normalized_record: SequenceRecord | None = None,
    current_sequence_hash: str | None = None,
    training_line: str | None = None,
    duplicate_of_accession: str | None = None,
) -> None:
    entry["included_in_current_dataset"] = included
    entry["current_dataset_reason"] = reason
    entry["duplicate_of_accession"] = duplicate_of_accession
    entry["rebuilt_at_utc"] = rebuilt_at_utc

    if normalized_record is None:
        entry["normalized_sequence"] = ""
        entry["normalized_sequence_hash"] = ""
        entry["training_line"] = ""
        return

    entry["normalized_sequence"] = normalized_record.sequence
    entry["normalized_sequence_hash"] = current_sequence_hash or ""
    entry["training_line"] = training_line or ""


def rebuild_dataset(
    raw_index: dict[str, dict[str, object]],
    accessions: tuple[str, ...],
    accession_aliases: dict[str, str],
    normalization: SequenceNormalizationConfig,
    duplicate_accession_count: int,
) -> RebuildResult:
    rebuild_timestamp = utc_now_iso()
    requested_accessions = set(accessions)
    for accession, entry in raw_index.items():
        entry["requested_accession"] = clean_optional_string(entry.get("requested_accession")) or accession
        entry["included_in_current_dataset"] = accession in requested_accessions and bool(entry.get("included_in_current_dataset"))
        if accession not in requested_accessions:
            entry["included_in_current_dataset"] = False
            entry["current_dataset_reason"] = "not_requested"
            entry["duplicate_of_accession"] = None
            entry["normalized_sequence"] = ""
            entry["normalized_sequence_hash"] = ""
            entry["training_line"] = ""
            entry["rebuilt_at_utc"] = rebuild_timestamp

    per_record_normalization = SequenceNormalizationConfig(
        sequence_type=normalization.sequence_type,
        min_length=normalization.min_length,
        max_length=normalization.max_length,
        invalid_base_policy=normalization.invalid_base_policy,
        max_ambiguous_ratio=normalization.max_ambiguous_ratio,
        deduplicate_sequences=False,
    )

    kept_lines: list[str] = []
    dropped_reasons: dict[str, int] = {}
    seen_sequence_hashes: dict[str, str] = {}

    for accession in accessions:
        entry = raw_index.get(accession)
        if entry is None:
            bump_reason(dropped_reasons, "missing_index_entry")
            continue

        entry["requested_accession"] = accession_aliases[accession]
        record = record_from_raw_index(entry, accession)
        normalized_records, report = normalize_records([record], per_record_normalization)
        if not normalized_records:
            reason = next(iter(report.dropped_reasons), "filtered_out")
            mark_entry(
                entry=entry,
                included=False,
                reason=reason,
                rebuilt_at_utc=rebuild_timestamp,
            )
            bump_reason(dropped_reasons, reason)
            continue

        normalized_record = normalized_records[0]
        current_sequence_hash = sequence_hash(normalized_record.sequence)
        duplicate_of_accession = seen_sequence_hashes.get(current_sequence_hash)
        if duplicate_of_accession is not None:
            mark_entry(
                entry=entry,
                included=False,
                reason="duplicate_sequence",
                rebuilt_at_utc=rebuild_timestamp,
                normalized_record=normalized_record,
                current_sequence_hash=current_sequence_hash,
                duplicate_of_accession=duplicate_of_accession,
            )
            bump_reason(dropped_reasons, "duplicate_sequence")
            continue

        training_line = normalized_record.to_training_line()
        seen_sequence_hashes[current_sequence_hash] = accession
        kept_lines.append(training_line)
        mark_entry(
            entry=entry,
            included=True,
            reason="kept",
            rebuilt_at_utc=rebuild_timestamp,
            normalized_record=normalized_record,
            current_sequence_hash=current_sequence_hash,
            training_line=training_line,
        )

    if duplicate_accession_count:
        dropped_reasons["duplicate_accession"] = dropped_reasons.get("duplicate_accession", 0) + duplicate_accession_count

    train_text = "\n".join(kept_lines)
    if train_text:
        train_text += "\n"

    return RebuildResult(
        train_text=train_text,
        record_count=len(kept_lines),
        dropped_count=sum(dropped_reasons.values()),
        dropped_reasons=dropped_reasons,
    )

from __future__ import annotations

import os
import gzip
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

from libs.core.pretrain.profiled import (
    MDCProfileCompilerConfig,
    _infer_profile_labels,
    build_profile_text_from_sequence_metadata,
)
from libs.data.training.kmer import _normalize_sequence as normalize_sequence
from libs.data.training.tokenizer import SequenceTokenizer
from libs.data.utilities.parsers import ParsedFastaEntry, iter_fasta_entries
from libs.data.utilities.refseq_history import (
    bootstrap_refseq_history,
    load_refseq_history,
    mark_archive_compiled,
    record_build_snapshot,
    resolve_refseq_history_path,
    resolve_refseq_history_root,
    save_refseq_history,
)
from libs.data.utilities.storage import render_tokenizer_map_payload


REFSEQ_PROFILE_METADATA_FIELDS = (
    "dataset_group",
    "keywords",
    "gene",
    "gene_synonym",
    "product",
    "note",
    "coded_by",
    "dbsource",
    "chromosome",
    "plasmid",
    "organelle",
    "segment",
    "host",
)

REFSEQ_INSTRUCTION_METADATA_FIELDS = (
    "keywords",
    "gene",
    "gene_synonym",
    "product",
    "note",
    "host",
    "plasmid",
    "organelle",
)

REFSEQ_PROFILE_LABEL_SIGNAL_FIELDS = (
    "fasta_header",
    "description",
    "keywords",
    "function",
    "pathway",
    "note",
    "product",
    "gene",
    "gene_synonym",
    "strain",
)

PROTEIN_START_TOKEN = "<|protein|>"
TRAIN_END_TOKEN = "<|endoftext|>"
TRAIN_TEXT_ARTIFACT_NAME = "train.txt"
TOKENIZER_MAP_ARTIFACT_NAME = "tokenizer_map.json"
INSTRUCTION_ARTIFACT_NAME = "instruction.jsonl"
OUTPUT_ARTIFACT_ALIASES = {
    "train": TRAIN_TEXT_ARTIFACT_NAME,
    TRAIN_TEXT_ARTIFACT_NAME: TRAIN_TEXT_ARTIFACT_NAME,
    "tokenizer_map": TOKENIZER_MAP_ARTIFACT_NAME,
    "tokenizer": TOKENIZER_MAP_ARTIFACT_NAME,
    TOKENIZER_MAP_ARTIFACT_NAME: TOKENIZER_MAP_ARTIFACT_NAME,
    "instruction": INSTRUCTION_ARTIFACT_NAME,
    INSTRUCTION_ARTIFACT_NAME: INSTRUCTION_ARTIFACT_NAME,
}
PARALLEL_MIN_RECORDS = 25_000
PARALLEL_TARGET_CHUNKS_PER_WORKER = 2
PARALLEL_MIN_CHUNK_SIZE = 2_000


@dataclass(slots=True, frozen=True)
class RefseqArchiveBundle:
    key: str
    group_name: str
    gpff_path: Path | None
    faa_path: Path | None


@dataclass(slots=True)
class RefseqInputFileSummary:
    kind: str
    path: str
    record_count: int = 0
    truncated: bool = False
    dropped_incomplete_records: int = 0


@dataclass(slots=True, frozen=True)
class RefseqProcessedArchiveState:
    kind: str
    path: str
    relative_path: str
    size_bytes: int
    modified_time_ns: int
    record_count: int = 0
    truncated: bool = False
    dropped_incomplete_records: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "modified_time_ns": self.modified_time_ns,
            "record_count": self.record_count,
            "truncated": self.truncated,
            "dropped_incomplete_records": self.dropped_incomplete_records,
        }


@dataclass(slots=True, frozen=True)
class RefseqProcessedBundleState:
    bundle_key: str
    group_name: str
    archives: tuple[RefseqProcessedArchiveState, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_key": self.bundle_key,
            "group_name": self.group_name,
            "archives": [archive.to_dict() for archive in self.archives],
        }


@dataclass(slots=True, frozen=True)
class RefseqProteinSourceRecord:
    accession: str
    accession_version: str
    description: str
    organism: str
    sequence: str
    metadata: dict[str, str]

    @property
    def sort_key(self) -> tuple[str, str]:
        versioned = self.accession_version or self.accession
        return (_canonical_accession(versioned), versioned)


@dataclass(slots=True, frozen=True)
class RefseqCandidateRecord:
    accession: str
    accession_version: str
    description: str
    organism: str
    sequence: str
    metadata: dict[str, str]
    origin: str

    @property
    def version_sort_key(self) -> tuple[int, str]:
        version = _accession_version_number(self.accession_version)
        return (version, self.accession_version or self.accession)

    @property
    def preference_key(self) -> tuple[int, tuple[int, str], int, int]:
        origin_rank = {"paired": 3, "gpff_only": 2, "faa_only": 1}.get(self.origin, 0)
        description_length = len(self.description)
        metadata_length = sum(len(key) + len(value) for key, value in self.metadata.items())
        return (origin_rank, self.version_sort_key, description_length, metadata_length)


@dataclass(slots=True, frozen=True)
class RefseqCompiledRecord:
    accession: str
    accession_version: str
    description: str
    organism: str
    sequence: str
    metadata: dict[str, str]
    profile: str
    derived_labels: tuple[str, ...]
    derived_keywords: tuple[str, ...]
    label_source: str
    origin: str
    content_hash: str

    @property
    def sequence_hash(self) -> str:
        return hashlib.sha256(self.sequence.encode("utf-8")).hexdigest()

    @property
    def sequence_train_line(self) -> str:
        return f"{PROTEIN_START_TOKEN}{self.sequence}{TRAIN_END_TOKEN}"


@dataclass(slots=True)
class RefseqLocalBuildSummary:
    input_root: str
    output_dir: str
    history_path: str
    train_text_path: str
    tokenizer_map_path: str
    instruction_path: str
    summary_path: str
    source_record_count: int
    record_count: int
    instruction_record_count: int
    instruction_condition_count: int
    skipped_instruction_condition_count: int
    duplicate_accession_count: int
    duplicate_sequence_count: int
    paired_record_count: int
    gpff_only_record_count: int
    faa_only_record_count: int
    skipped_empty_sequence_count: int
    sequence_mismatch_count: int
    truncated_input_count: int
    new_source_record_count: int
    updated_source_record_count: int
    unchanged_source_record_count: int
    removed_source_record_count: int
    processed_archive_count: int
    deleted_archive_count: int
    reused_existing_artifacts: bool
    max_records: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "input_root": self.input_root,
            "output_dir": self.output_dir,
            "history_path": self.history_path,
            "train_text_path": self.train_text_path,
            "tokenizer_map_path": self.tokenizer_map_path,
            "instruction_path": self.instruction_path,
            "summary_path": self.summary_path,
            "source_record_count": self.source_record_count,
            "record_count": self.record_count,
            "instruction_record_count": self.instruction_record_count,
            "instruction_condition_count": self.instruction_condition_count,
            "skipped_instruction_condition_count": self.skipped_instruction_condition_count,
            "duplicate_accession_count": self.duplicate_accession_count,
            "duplicate_sequence_count": self.duplicate_sequence_count,
            "paired_record_count": self.paired_record_count,
            "gpff_only_record_count": self.gpff_only_record_count,
            "faa_only_record_count": self.faa_only_record_count,
            "skipped_empty_sequence_count": self.skipped_empty_sequence_count,
            "sequence_mismatch_count": self.sequence_mismatch_count,
            "truncated_input_count": self.truncated_input_count,
            "new_source_record_count": self.new_source_record_count,
            "updated_source_record_count": self.updated_source_record_count,
            "unchanged_source_record_count": self.unchanged_source_record_count,
            "removed_source_record_count": self.removed_source_record_count,
            "processed_archive_count": self.processed_archive_count,
            "deleted_archive_count": self.deleted_archive_count,
            "reused_existing_artifacts": self.reused_existing_artifacts,
            "max_records": self.max_records,
        }


def build_local_refseq_profile_text_artifacts(
    input_root: Path | str,
    output_dir: Path | str,
    *,
    source_name: str = "refseq",
    vocab_size: int | None = None,
    instruction_min_proteins: int = 10,
    kmer_size: int = 3,
    profile_vocab_size: int = 256,
    profile_sample_char_limit: int = 2_000_000,
    max_records: int | None = None,
    profile_config: MDCProfileCompilerConfig | None = None,
    workers: int = 1,
    skip_artifacts: str | tuple[str, ...] | list[str] | set[str] | None = None,
) -> RefseqLocalBuildSummary:
    del kmer_size, profile_sample_char_limit

    if instruction_min_proteins <= 0:
        raise ValueError("instruction_min_proteins must be greater than 0.")
    effective_workers = _resolve_worker_count(workers)
    skipped_artifact_names = _normalize_output_artifact_names(skip_artifacts)

    requested_input_root = Path(input_root)
    resolved_output_dir = Path(output_dir)
    if not requested_input_root.exists():
        raise FileNotFoundError(f"Input directory was not found: {requested_input_root}")

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_input_root = _resolve_scoped_input_root(requested_input_root, resolved_output_dir)
    history_root = resolve_refseq_history_root(resolved_input_root)
    history_path = resolve_refseq_history_path(resolved_input_root)
    train_text_path = resolved_output_dir / TRAIN_TEXT_ARTIFACT_NAME
    tokenizer_map_path = resolved_output_dir / TOKENIZER_MAP_ARTIFACT_NAME
    instruction_path = resolved_output_dir / INSTRUCTION_ARTIFACT_NAME
    legacy_source_index_path = resolved_output_dir / "source_index.json"
    summary_path = resolved_output_dir / "summary.json"
    write_train_text = TRAIN_TEXT_ARTIFACT_NAME not in skipped_artifact_names
    write_tokenizer_map = TOKENIZER_MAP_ARTIFACT_NAME not in skipped_artifact_names
    write_instruction_jsonl = INSTRUCTION_ARTIFACT_NAME not in skipped_artifact_names
    if write_tokenizer_map and not write_train_text and not train_text_path.exists():
        raise FileNotFoundError(
            "Cannot build tokenizer_map.json while train.txt is skipped because the existing train.txt file "
            f"was not found: {train_text_path}"
        )

    effective_vocab_size = profile_vocab_size if vocab_size is None else vocab_size
    effective_profile_config = profile_config or MDCProfileCompilerConfig(
        metadata_fields=REFSEQ_PROFILE_METADATA_FIELDS,
        label_signal_fields=REFSEQ_PROFILE_LABEL_SIGNAL_FIELDS,
    )
    instruction_profile_config = replace(
        effective_profile_config,
        metadata_fields=(
            effective_profile_config.metadata_fields
            if profile_config is not None
            else REFSEQ_INSTRUCTION_METADATA_FIELDS
        ),
        include_task=False,
        include_label_source=False,
        include_sequence_type=False,
        include_source_name=False,
    )
    history = load_refseq_history(history_path, input_root=history_root)
    if history_root.exists():
        bootstrap_refseq_history(history_root, history)

    previous_summary_payload = _load_summary_payload(summary_path)
    previous_bundle_states = _load_processed_bundle_states(previous_summary_payload)
    bundle_state_root = _resolve_bundle_state_root(
        requested_input_root=requested_input_root,
        previous_summary_payload=previous_summary_payload,
    )
    bundles = _discover_archive_bundles(
        resolved_input_root,
        bundle_key_root=bundle_state_root,
    )
    bundles = _reuse_previous_bundle_keys(
        bundles,
        previous_bundle_states=previous_bundle_states,
    )
    previous_records = _load_previous_compiled_records(
        instruction_path=instruction_path,
        source_name=source_name,
        profile_config=effective_profile_config,
    )
    if not bundles and not previous_records:
        raise ValueError(
            f"No .gpff.gz or .faa.gz files were found under {resolved_input_root}, "
            "and no reusable compiled artifacts were found."
        )

    previous_records_support_bundle_state = _records_have_bundle_keys(previous_records)
    if previous_records:
        bundles_to_process = [
            bundle
            for bundle in bundles
            if _bundle_requires_processing_from_summary(
                bundle,
                input_root=bundle_state_root,
                previous_bundle_states=previous_bundle_states,
            )
        ]
    else:
        bundles_to_process = bundles

    current_bundle_keys = {bundle.key for bundle in bundles}
    removed_bundle_keys = (
        _removed_bundle_keys_within_scope(
            previous_bundle_states,
            current_bundle_keys=current_bundle_keys,
            scope_root=resolved_input_root,
        )
        if write_instruction_jsonl
        else set()
    )
    if previous_records and not previous_records_support_bundle_state and (bundles_to_process or removed_bundle_keys):
        bundles_to_process = bundles
        removed_bundle_keys = set()

    candidate_by_accession: dict[str, RefseqCandidateRecord] = {}
    processed_bundle_summaries: dict[str, list[RefseqInputFileSummary]] = {}

    duplicate_accession_count = 0
    skipped_empty_sequence_count = 0
    sequence_mismatch_count = 0
    truncated_input_count = 0
    stop_requested = False
    fully_processed_bundles: list[RefseqArchiveBundle] = []

    for bundle in bundles_to_process:
        bundle_records, bundle_summaries = _iter_bundle_records(
            bundle,
            source_name=source_name,
        )
        bundle_completed = True
        try:
            for bundle_summary in bundle_summaries:
                if bundle_summary.truncated:
                    truncated_input_count += 1

            while True:
                next_item = next(bundle_records, None)
                if next_item is None:
                    break
                merged_record, origin = next_item
                normalized_sequence = normalize_sequence(merged_record.sequence, sequence_type="protein")
                if not normalized_sequence:
                    skipped_empty_sequence_count += 1
                    continue
                if merged_record.metadata.get("sequence_mismatch") == "true":
                    sequence_mismatch_count += 1

                candidate = RefseqCandidateRecord(
                    accession=merged_record.accession,
                    accession_version=merged_record.accession_version,
                    description=merged_record.description,
                    organism=merged_record.organism,
                    sequence=normalized_sequence,
                    metadata=dict(merged_record.metadata),
                    origin=origin,
                )
                existing_candidate = candidate_by_accession.get(candidate.accession)
                if existing_candidate is not None:
                    duplicate_accession_count += 1
                    if existing_candidate.preference_key >= candidate.preference_key:
                        continue
                candidate_by_accession[candidate.accession] = candidate

                if max_records is not None and len(candidate_by_accession) >= max_records:
                    stop_requested = True
                    bundle_completed = next(bundle_records, None) is None
                    break
        finally:
            close_bundle_records = getattr(bundle_records, "close", None)
            if callable(close_bundle_records):
                close_bundle_records()

        if bundle_completed:
            fully_processed_bundles.append(bundle)
            processed_bundle_summaries[bundle.key] = bundle_summaries

        if stop_requested:
            break

    processed_archive_paths = _bundle_archive_paths(fully_processed_bundles)

    pending_records = _compile_refseq_records(
        candidate_by_accession,
        source_name=source_name,
        profile_config=effective_profile_config,
        workers=effective_workers,
    )

    previous_source_hashes = {record.accession: record.content_hash for record in previous_records}
    pending_bundle_keys = {bundle.key for bundle in bundles_to_process}
    if not previous_records:
        carried_forward_records: list[RefseqCompiledRecord] = []
    elif not bundles:
        carried_forward_records = list(previous_records)
    elif not pending_bundle_keys and not removed_bundle_keys:
        carried_forward_records = list(previous_records)
    elif previous_records_support_bundle_state:
        carried_forward_records = _filter_previous_records_for_pending_bundles(
            previous_records,
            pending_bundle_keys=pending_bundle_keys,
            removed_bundle_keys=removed_bundle_keys,
        )
    else:
        carried_forward_records = []
    compiled_record_by_accession = {record.accession: record for record in carried_forward_records}
    for record in pending_records:
        compiled_record_by_accession[record.accession] = record
    compiled_records = [compiled_record_by_accession[accession] for accession in sorted(compiled_record_by_accession)]
    if not compiled_records:
        raise ValueError("No training records were produced from the provided RefSeq archive directory.")

    current_source_hashes = {record.accession: record.content_hash for record in compiled_records}
    new_source_record_count, updated_source_record_count, unchanged_source_record_count, removed_source_record_count = (
        _diff_source_hashes(previous_source_hashes, current_source_hashes)
    )

    kept_records: list[RefseqCompiledRecord] = []
    duplicate_sequence_count = 0
    seen_sequence_hashes: dict[str, str] = {}
    for record in compiled_records:
        duplicate_of_accession = seen_sequence_hashes.get(record.sequence_hash)
        if duplicate_of_accession is not None:
            duplicate_sequence_count += 1
            continue

        kept_records.append(record)
        seen_sequence_hashes[record.sequence_hash] = record.accession
        if max_records is not None and len(kept_records) >= max_records:
            break

    if not kept_records:
        raise ValueError("No training records were produced from the provided RefSeq archive directory.")

    builder_metadata = {
        "type": "local_refseq_sequence_only",
        "source_name": source_name,
        "tokenizer_type": "bpe",
        "vocab_size_requested": effective_vocab_size,
        "instruction_min_proteins": instruction_min_proteins,
        "max_records": max_records,
    }
    instruction_record_count, instruction_condition_count, skipped_instruction_condition_count = (
        _count_instruction_metadata(
            kept_records,
            instruction_min_proteins=instruction_min_proteins,
        )
    )
    paired_record_count = sum(1 for record in compiled_records if record.origin == "paired")
    gpff_only_record_count = sum(1 for record in compiled_records if record.origin == "gpff_only")
    faa_only_record_count = sum(1 for record in compiled_records if record.origin == "faa_only")
    sequence_mismatch_count = sum(
        1 for record in compiled_records if record.metadata.get("sequence_mismatch") == "true"
    )
    previous_record_by_accession = {record.accession: record for record in previous_records}
    train_requires_rewrite = removed_source_record_count > 0 or any(
        previous_record.sequence_hash != compiled_record_by_accession[accession].sequence_hash
        for accession, previous_record in previous_record_by_accession.items()
        if accession in compiled_record_by_accession
    )
    instruction_requires_rewrite = updated_source_record_count > 0 or removed_source_record_count > 0

    artifact_writes_performed = False
    if write_train_text:
        train_records_to_append = _collect_train_records_to_append(previous_records, kept_records)
        if not previous_records or not train_text_path.exists() or train_requires_rewrite:
            _rewrite_train_text_artifact(train_text_path, kept_records)
            artifact_writes_performed = True
        elif train_records_to_append:
            _append_train_text_artifact(train_text_path, train_records_to_append)
            artifact_writes_performed = True

    if write_tokenizer_map:
        tokenizer_train_text = _render_train_text(kept_records)
        tokenizer_map_text = _render_tokenizer_map_text(
            tokenizer_train_text,
            source_name=source_name,
            vocab_size=effective_vocab_size,
            builder_metadata=builder_metadata,
        )
        if _write_text_if_changed(tokenizer_map_path, tokenizer_map_text):
            artifact_writes_performed = True

    if write_instruction_jsonl:
        instruction_records_to_append = _collect_instruction_records_to_append(previous_records, compiled_records)
        if not previous_records or not instruction_path.exists() or instruction_requires_rewrite:
            _rewrite_instruction_jsonl_artifact(
                instruction_path,
                compiled_records,
                source_name=source_name,
                profile_config=instruction_profile_config,
                workers=effective_workers,
            )
            artifact_writes_performed = True
        elif instruction_records_to_append:
            _append_instruction_jsonl_artifact(
                instruction_path,
                instruction_records_to_append,
                source_name=source_name,
                profile_config=instruction_profile_config,
                workers=effective_workers,
            )
            artifact_writes_performed = True

    reused_existing_artifacts = not bundles_to_process and not artifact_writes_performed

    for archive_path in processed_archive_paths:
        mark_archive_compiled(
            history,
            history_root,
            archive_path,
            output_dir=resolved_output_dir,
        )

    legacy_source_index_path.unlink(missing_ok=True)

    current_bundle_states = dict(previous_bundle_states)
    if write_instruction_jsonl:
        if bundles:
            for bundle_key in removed_bundle_keys:
                current_bundle_states.pop(bundle_key, None)
        for bundle in fully_processed_bundles:
            current_bundle_states[bundle.key] = _build_processed_bundle_state(
                bundle,
                input_root=bundle_state_root,
                bundle_summaries=processed_bundle_summaries.get(bundle.key, []),
            )

    input_summaries = _flatten_processed_bundle_states(current_bundle_states)

    summary = RefseqLocalBuildSummary(
        input_root=str(resolved_input_root),
        output_dir=str(resolved_output_dir),
        history_path=str(history_path),
        train_text_path=str(train_text_path),
        tokenizer_map_path=str(tokenizer_map_path),
        instruction_path=str(instruction_path),
        summary_path=str(summary_path),
        source_record_count=len(compiled_records),
        record_count=len(kept_records),
        instruction_record_count=instruction_record_count,
        instruction_condition_count=instruction_condition_count,
        skipped_instruction_condition_count=skipped_instruction_condition_count,
        duplicate_accession_count=duplicate_accession_count,
        duplicate_sequence_count=duplicate_sequence_count,
        paired_record_count=paired_record_count,
        gpff_only_record_count=gpff_only_record_count,
        faa_only_record_count=faa_only_record_count,
        skipped_empty_sequence_count=skipped_empty_sequence_count,
        sequence_mismatch_count=sequence_mismatch_count,
        truncated_input_count=truncated_input_count,
        new_source_record_count=new_source_record_count,
        updated_source_record_count=updated_source_record_count,
        unchanged_source_record_count=unchanged_source_record_count,
        removed_source_record_count=removed_source_record_count,
        processed_archive_count=len(processed_archive_paths),
        deleted_archive_count=0,
        reused_existing_artifacts=reused_existing_artifacts,
        max_records=max_records,
    )
    summary_payload = {
        **summary.to_dict(),
        "builder": builder_metadata,
        "requested_input_root": str(requested_input_root),
        "processed_bundles": [
            current_bundle_states[bundle_key].to_dict()
            for bundle_key in sorted(current_bundle_states)
        ],
        "input_files": input_summaries,
        "skip_artifacts": sorted(skipped_artifact_names),
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    record_build_snapshot(
        history,
        output_dir=resolved_output_dir,
        summary=summary_payload,
    )
    save_refseq_history(history_path, history)
    return summary


def _load_previous_compiled_records(
    *,
    instruction_path: Path,
    source_name: str,
    profile_config: MDCProfileCompilerConfig,
) -> list[RefseqCompiledRecord]:
    return _load_compiled_records_jsonl(
        instruction_path,
        source_name=source_name,
        profile_config=profile_config,
    )


def _resolve_scoped_input_root(input_root: Path, output_dir: Path) -> Path:
    scope_candidate = input_root / output_dir.name
    if output_dir.name and scope_candidate.is_dir():
        return scope_candidate
    return input_root


def _resolve_bundle_state_root(
    *,
    requested_input_root: Path,
    previous_summary_payload: dict[str, object],
) -> Path:
    previous_requested_value = _clean_field_text(str(previous_summary_payload.get("requested_input_root", "")))
    if not previous_requested_value:
        return requested_input_root

    previous_requested_input_root = Path(previous_requested_value)
    try:
        requested_input_root.relative_to(previous_requested_input_root)
    except ValueError:
        return requested_input_root
    return previous_requested_input_root


def _load_summary_payload(summary_path: Path) -> dict[str, object]:
    if not summary_path.exists():
        return {}

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_processed_bundle_states(
    summary_payload: dict[str, object],
) -> dict[str, RefseqProcessedBundleState]:
    processed_bundles = summary_payload.get("processed_bundles", [])
    if not isinstance(processed_bundles, list):
        return {}

    loaded: dict[str, RefseqProcessedBundleState] = {}
    for payload in processed_bundles:
        state = _processed_bundle_state_from_payload(payload)
        if state is None:
            continue
        loaded[state.bundle_key] = state
    return loaded


def _processed_bundle_state_from_payload(payload: object) -> RefseqProcessedBundleState | None:
    if not isinstance(payload, dict):
        return None

    bundle_key = _clean_field_text(str(payload.get("bundle_key", "")))
    group_name = _clean_field_text(str(payload.get("group_name", "")))
    raw_archives = payload.get("archives", [])
    if not bundle_key or not isinstance(raw_archives, list):
        return None

    archives: list[RefseqProcessedArchiveState] = []
    for raw_archive in raw_archives:
        if not isinstance(raw_archive, dict):
            continue
        kind = _clean_field_text(str(raw_archive.get("kind", "")))
        path = _clean_field_text(str(raw_archive.get("path", "")))
        relative_path = _clean_field_text(str(raw_archive.get("relative_path", "")))
        size_bytes = _coerce_optional_int(raw_archive.get("size_bytes"))
        modified_time_ns = _coerce_optional_int(raw_archive.get("modified_time_ns"))
        if not kind or not path or not relative_path or size_bytes is None or modified_time_ns is None:
            continue
        archives.append(
            RefseqProcessedArchiveState(
                kind=kind,
                path=path,
                relative_path=relative_path,
                size_bytes=size_bytes,
                modified_time_ns=modified_time_ns,
                record_count=_coerce_optional_int(raw_archive.get("record_count")) or 0,
                truncated=bool(raw_archive.get("truncated")),
                dropped_incomplete_records=_coerce_optional_int(
                    raw_archive.get("dropped_incomplete_records")
                )
                or 0,
            )
        )

    if not archives:
        return None

    return RefseqProcessedBundleState(
        bundle_key=bundle_key,
        group_name=group_name,
        archives=tuple(sorted(archives, key=lambda archive: (archive.kind, archive.relative_path))),
    )


def _load_compiled_records_jsonl(
    path: Path,
    *,
    source_name: str,
    profile_config: MDCProfileCompilerConfig,
) -> list[RefseqCompiledRecord]:
    if not path.exists():
        return []

    compiled_record_by_accession: dict[str, RefseqCompiledRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                continue

            accession_version = _clean_field_text(
                str(payload.get("accession_version", payload.get("accession", "")))
            )
            accession = _canonical_accession(accession_version)
            if not accession:
                continue

            sequence_value = payload.get("sequence", payload.get("output", ""))
            sequence = normalize_sequence(str(sequence_value), sequence_type="protein")
            if not sequence:
                continue

            raw_metadata = payload.get("metadata", {})
            metadata: dict[str, str] = {}
            if isinstance(raw_metadata, dict):
                for key, value in raw_metadata.items():
                    normalized_key = _clean_field_text(str(key))
                    normalized_value = _clean_field_text(str(value))
                    if normalized_key and normalized_value:
                        metadata[normalized_key] = normalized_value

            description = _clean_field_text(str(payload.get("description", "")))
            organism = _clean_field_text(str(payload.get("organism", "")))
            derived_labels = _tuple_of_strings(payload.get("derived_labels", ()))
            derived_keywords = _tuple_of_strings(payload.get("derived_keywords", ()))
            label_source = _clean_field_text(str(payload.get("label_source", "")))
            origin = _clean_field_text(str(payload.get("origin", ""))) or _infer_record_origin(metadata)
            profile = build_profile_text_from_sequence_metadata(
                source_name=source_name,
                sequence_type="protein",
                description=description,
                organism=organism,
                metadata=metadata,
                config=profile_config,
            )
            compiled_record_by_accession[accession] = RefseqCompiledRecord(
                accession=accession,
                accession_version=accession_version or accession,
                description=description,
                organism=organism,
                sequence=sequence,
                metadata=metadata,
                profile=profile,
                derived_labels=derived_labels,
                derived_keywords=derived_keywords,
                label_source=label_source,
                origin=origin,
                content_hash=_content_hash_for_record(
                    accession=accession,
                    accession_version=accession_version or accession,
                    description=description,
                    organism=organism,
                    sequence=sequence,
                    metadata=metadata,
                ),
            )

    return [compiled_record_by_accession[accession] for accession in sorted(compiled_record_by_accession)]


def _records_have_bundle_keys(records: list[RefseqCompiledRecord]) -> bool:
    return all(_clean_field_text(record.metadata.get("dataset_bundle", "")) for record in records)


def _normalize_output_artifact_names(
    skip_artifacts: str | tuple[str, ...] | list[str] | set[str] | None,
) -> set[str]:
    if skip_artifacts is None:
        return set()

    raw_values = [skip_artifacts] if isinstance(skip_artifacts, str) else list(skip_artifacts)
    normalized: set[str] = set()
    for raw_value in raw_values:
        for token in str(raw_value).split(","):
            cleaned = _clean_field_text(token).lower()
            if not cleaned:
                continue
            artifact_name = OUTPUT_ARTIFACT_ALIASES.get(cleaned)
            if artifact_name is None:
                supported = ", ".join(sorted(OUTPUT_ARTIFACT_ALIASES))
                raise ValueError(f"Unsupported skip artifact '{token}'. Supported values: {supported}")
            normalized.add(artifact_name)
    return normalized


def _render_train_text(records: list[RefseqCompiledRecord]) -> str:
    return "\n".join(record.sequence_train_line for record in records) + "\n"


def _render_tokenizer_map_text(
    train_text: str,
    *,
    source_name: str,
    vocab_size: int,
    builder_metadata: dict[str, object],
) -> str:
    tokenizer = SequenceTokenizer.from_text(
        train_text,
        sequence_type="protein",
        vocab_size=vocab_size,
    )
    tokenizer_map_payload = json.loads(
        render_tokenizer_map_payload(
            source_name=source_name,
            record_count=_count_nonempty_lines(train_text),
            tokenizer=tokenizer,
        )
    )
    tokenizer_map_payload["builder"] = {
        **builder_metadata,
        "vocab_size_actual": tokenizer.vocab_size,
    }
    return json.dumps(tokenizer_map_payload, ensure_ascii=False, indent=2) + "\n"


def _write_text_if_changed(path: Path, text: str) -> bool:
    encoded_text = text.encode("utf-8")
    if _path_matches_bytes(path, encoded_text):
        return False
    path.write_bytes(encoded_text)
    return True


def _path_matches_bytes(path: Path, payload: bytes, *, chunk_size: int = 1_048_576) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size != len(payload):
        return False

    offset = 0
    with path.open("rb") as handle:
        while offset < len(payload):
            expected_chunk = payload[offset : offset + chunk_size]
            if handle.read(len(expected_chunk)) != expected_chunk:
                return False
            offset += len(expected_chunk)
    return True


def _collect_train_records_to_append(
    previous_records: list[RefseqCompiledRecord],
    current_kept_records: list[RefseqCompiledRecord],
) -> list[RefseqCompiledRecord]:
    seen_sequence_hashes = {record.sequence_hash for record in _dedupe_records_by_sequence(previous_records)}
    records_to_append: list[RefseqCompiledRecord] = []
    for record in current_kept_records:
        if record.sequence_hash in seen_sequence_hashes:
            continue
        seen_sequence_hashes.add(record.sequence_hash)
        records_to_append.append(record)
    return records_to_append


def _collect_instruction_records_to_append(
    previous_records: list[RefseqCompiledRecord],
    current_records: list[RefseqCompiledRecord],
) -> list[RefseqCompiledRecord]:
    previous_accessions = {record.accession for record in previous_records}
    return [record for record in current_records if record.accession not in previous_accessions]


def _rewrite_train_text_artifact(path: Path, records: list[RefseqCompiledRecord]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.sequence_train_line)
            handle.write("\n")
    temp_path.replace(path)


def _append_train_text_artifact(path: Path, records: list[RefseqCompiledRecord]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.sequence_train_line)
            handle.write("\n")


def _rewrite_instruction_jsonl_artifact(
    path: Path,
    records: list[RefseqCompiledRecord],
    *,
    source_name: str,
    profile_config: MDCProfileCompilerConfig,
    workers: int,
) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for chunk_text in _iter_instruction_jsonl_chunks(
            records,
            source_name=source_name,
            profile_config=profile_config,
            workers=workers,
        ):
            handle.write(chunk_text)
    temp_path.replace(path)


def _append_instruction_jsonl_artifact(
    path: Path,
    records: list[RefseqCompiledRecord],
    *,
    source_name: str,
    profile_config: MDCProfileCompilerConfig,
    workers: int,
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for chunk_text in _iter_instruction_jsonl_chunks(
            records,
            source_name=source_name,
            profile_config=profile_config,
            workers=workers,
        ):
            handle.write(chunk_text)


def _count_nonempty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def _bundle_requires_processing_from_summary(
    bundle: RefseqArchiveBundle,
    *,
    input_root: Path,
    previous_bundle_states: dict[str, RefseqProcessedBundleState],
) -> bool:
    previous_state = previous_bundle_states.get(bundle.key)
    if previous_state is None:
        return True

    current_archives = tuple(
        _build_processed_archive_state(
            path=archive_path,
            kind=kind,
            input_root=input_root,
        )
        for kind, archive_path in (
            ("gpff", bundle.gpff_path),
            ("faa", bundle.faa_path),
        )
        if archive_path is not None
    )
    if len(current_archives) != len(previous_state.archives):
        return True

    previous_archives = {
        archive.kind: archive
        for archive in previous_state.archives
    }
    for archive in current_archives:
        previous_archive = previous_archives.get(archive.kind)
        if previous_archive is None:
            return True
        if previous_archive.relative_path != archive.relative_path:
            return True
        if previous_archive.size_bytes != archive.size_bytes:
            return True
        if previous_archive.modified_time_ns != archive.modified_time_ns:
            return True
    return False


def _removed_bundle_keys_within_scope(
    previous_bundle_states: dict[str, RefseqProcessedBundleState],
    *,
    current_bundle_keys: set[str],
    scope_root: Path,
) -> set[str]:
    return {
        bundle_key
        for bundle_key, state in previous_bundle_states.items()
        if bundle_key not in current_bundle_keys and _bundle_state_is_within_scope(state, scope_root=scope_root)
    }


def _bundle_state_is_within_scope(
    state: RefseqProcessedBundleState,
    *,
    scope_root: Path,
) -> bool:
    return any(_path_is_within_root(Path(archive.path), scope_root) for archive in state.archives)


def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _bundle_archive_paths(bundles: list[RefseqArchiveBundle]) -> list[Path]:
    archive_paths: list[Path] = []
    for bundle in bundles:
        if bundle.gpff_path is not None:
            archive_paths.append(bundle.gpff_path)
        if bundle.faa_path is not None:
            archive_paths.append(bundle.faa_path)
    return archive_paths


def _build_processed_bundle_state(
    bundle: RefseqArchiveBundle,
    *,
    input_root: Path,
    bundle_summaries: list[RefseqInputFileSummary],
) -> RefseqProcessedBundleState:
    summaries_by_kind = {summary.kind: summary for summary in bundle_summaries}
    archives = tuple(
        _build_processed_archive_state(
            path=archive_path,
            kind=kind,
            input_root=input_root,
            summary=summaries_by_kind.get(kind),
        )
        for kind, archive_path in (
            ("gpff", bundle.gpff_path),
            ("faa", bundle.faa_path),
        )
        if archive_path is not None
    )
    return RefseqProcessedBundleState(
        bundle_key=bundle.key,
        group_name=bundle.group_name,
        archives=archives,
    )


def _build_processed_archive_state(
    *,
    path: Path,
    kind: str,
    input_root: Path,
    summary: RefseqInputFileSummary | None = None,
) -> RefseqProcessedArchiveState:
    stat = path.stat()
    return RefseqProcessedArchiveState(
        kind=kind,
        path=str(path),
        relative_path=path.relative_to(input_root).as_posix(),
        size_bytes=stat.st_size,
        modified_time_ns=stat.st_mtime_ns,
        record_count=summary.record_count if summary is not None else 0,
        truncated=summary.truncated if summary is not None else False,
        dropped_incomplete_records=summary.dropped_incomplete_records if summary is not None else 0,
    )


def _flatten_processed_bundle_states(
    processed_bundle_states: dict[str, RefseqProcessedBundleState],
) -> list[dict[str, object]]:
    flattened: list[dict[str, object]] = []
    for bundle_key in sorted(processed_bundle_states):
        state = processed_bundle_states[bundle_key]
        for archive in state.archives:
            flattened.append(
                {
                    "bundle_key": state.bundle_key,
                    "group_name": state.group_name,
                    "kind": archive.kind,
                    "path": archive.path,
                    "relative_path": archive.relative_path,
                    "size_bytes": archive.size_bytes,
                    "modified_time_ns": archive.modified_time_ns,
                    "record_count": archive.record_count,
                    "truncated": archive.truncated,
                    "dropped_incomplete_records": archive.dropped_incomplete_records,
                }
            )
    return flattened


def _filter_previous_records_for_pending_bundles(
    previous_records: list[RefseqCompiledRecord],
    *,
    pending_bundle_keys: set[str],
    removed_bundle_keys: set[str],
) -> list[RefseqCompiledRecord]:
    return [
        record
        for record in previous_records
        if _clean_field_text(record.metadata.get("dataset_bundle", "")) not in pending_bundle_keys
        and _clean_field_text(record.metadata.get("dataset_bundle", "")) not in removed_bundle_keys
    ]


def _count_instruction_metadata(
    records: list[RefseqCompiledRecord],
    *,
    instruction_min_proteins: int,
) -> tuple[int, int, int]:
    keyword_label_counts: dict[str, int] = {}
    for record in records:
        if record.label_source != "keyword rules":
            continue
        for label in record.derived_labels:
            keyword_label_counts[label] = keyword_label_counts.get(label, 0) + 1

    emitted_condition_count = sum(
        1
        for label_count in keyword_label_counts.values()
        if label_count >= instruction_min_proteins
    )
    skipped_condition_count = sum(
        1
        for label_count in keyword_label_counts.values()
        if label_count < instruction_min_proteins
    )
    return len(records), emitted_condition_count, skipped_condition_count


def _compile_refseq_records(
    candidate_by_accession: dict[str, RefseqCandidateRecord],
    *,
    source_name: str,
    profile_config: MDCProfileCompilerConfig,
    workers: int = 1,
) -> list[RefseqCompiledRecord]:
    sorted_candidates = [candidate_by_accession[accession] for accession in sorted(candidate_by_accession)]
    if not _should_use_process_pool(record_count=len(sorted_candidates), workers=workers):
        return [
            _compile_refseq_candidate_record(
                candidate,
                source_name=source_name,
                profile_config=profile_config,
            )
            for candidate in sorted_candidates
        ]

    work_items = [
        (tuple(chunk_candidates), source_name, profile_config)
        for chunk_candidates in _chunk_for_parallelism(sorted_candidates, workers=workers)
    ]
    compiled: list[RefseqCompiledRecord] = []
    try:
        with _executor_class_for_parallelism()(max_workers=workers) as executor:
            for chunk_records in executor.map(_compile_refseq_record_chunk, work_items):
                compiled.extend(chunk_records)
    except (OSError, PermissionError, MemoryError):
        return [
            _compile_refseq_candidate_record(
                candidate,
                source_name=source_name,
                profile_config=profile_config,
            )
            for candidate in sorted_candidates
        ]
    return compiled


def _iter_instruction_jsonl_chunks(
    records: list[RefseqCompiledRecord],
    *,
    source_name: str,
    profile_config: MDCProfileCompilerConfig,
    workers: int = 1,
) -> Iterator[str]:
    if not _should_use_process_pool(record_count=len(records), workers=workers):
        instruction_lines = [
            _build_instruction_jsonl_line(
                record,
                source_name=source_name,
                profile_config=profile_config,
            )
            for record in records
        ]
        if instruction_lines:
            yield "\n".join(instruction_lines) + "\n"
        return

    work_items = [
        (tuple(chunk_records), source_name, profile_config)
        for chunk_records in _chunk_for_parallelism(records, workers=workers)
    ]
    try:
        with _executor_class_for_parallelism()(max_workers=workers) as executor:
            for chunk_text, _ in executor.map(_build_instruction_jsonl_chunk, work_items):
                if chunk_text:
                    yield chunk_text
    except (OSError, PermissionError, MemoryError):
        instruction_lines = [
            _build_instruction_jsonl_line(
                record,
                source_name=source_name,
                profile_config=profile_config,
            )
            for record in records
        ]
        if instruction_lines:
            yield "\n".join(instruction_lines) + "\n"


def _resolve_worker_count(workers: int) -> int:
    if workers < 0:
        raise ValueError("workers must be greater than or equal to 0.")
    if workers == 0:
        return max(os.cpu_count() or 1, 1)
    return workers


def _should_use_process_pool(*, record_count: int, workers: int) -> bool:
    return workers > 1 and record_count >= PARALLEL_MIN_RECORDS


def _executor_class_for_parallelism() -> type[ProcessPoolExecutor] | type[ThreadPoolExecutor]:
    # Windows uses spawn semantics, which forces huge record chunks through pickle and can blow up RAM.
    # Threads keep the parallel path available without serializing every candidate record.
    if os.name == "nt":
        return ThreadPoolExecutor
    return ProcessPoolExecutor


def _chunk_for_parallelism(records: list, *, workers: int) -> Iterator[list]:
    chunk_size = max(
        PARALLEL_MIN_CHUNK_SIZE,
        (len(records) + (workers * PARALLEL_TARGET_CHUNKS_PER_WORKER) - 1)
        // (workers * PARALLEL_TARGET_CHUNKS_PER_WORKER),
    )
    for start in range(0, len(records), chunk_size):
        yield records[start : start + chunk_size]


def _compile_refseq_candidate_record(
    candidate: RefseqCandidateRecord,
    *,
    source_name: str,
    profile_config: MDCProfileCompilerConfig,
) -> RefseqCompiledRecord:
    profile = build_profile_text_from_sequence_metadata(
        source_name=source_name,
        sequence_type="protein",
        description=candidate.description,
        organism=candidate.organism,
        metadata=candidate.metadata,
        config=profile_config,
    )
    derived_labels, derived_keywords, label_source = _infer_profile_labels(
        description=candidate.description,
        sequence_type="protein",
        metadata=candidate.metadata,
        config=profile_config,
    )
    content_hash = _content_hash_for_record(
        accession=candidate.accession,
        accession_version=candidate.accession_version,
        description=candidate.description,
        organism=candidate.organism,
        sequence=candidate.sequence,
        metadata=candidate.metadata,
    )
    return RefseqCompiledRecord(
        accession=candidate.accession,
        accession_version=candidate.accession_version,
        description=candidate.description,
        organism=candidate.organism,
        sequence=candidate.sequence,
        metadata=dict(candidate.metadata),
        profile=profile,
        derived_labels=derived_labels,
        derived_keywords=derived_keywords,
        label_source=label_source,
        origin=candidate.origin,
        content_hash=content_hash,
    )


def _compile_refseq_record_chunk(
    args: tuple[tuple[RefseqCandidateRecord, ...], str, MDCProfileCompilerConfig],
) -> list[RefseqCompiledRecord]:
    candidates, source_name, profile_config = args
    return [
        _compile_refseq_candidate_record(
            candidate,
            source_name=source_name,
            profile_config=profile_config,
        )
        for candidate in candidates
    ]


def _build_instruction_jsonl_line(
    record: RefseqCompiledRecord,
    *,
    source_name: str,
    profile_config: MDCProfileCompilerConfig,
) -> str:
    instruction_profile = build_profile_text_from_sequence_metadata(
        source_name=source_name,
        sequence_type="protein",
        description=record.description,
        organism=record.organism,
        metadata=record.metadata,
        config=profile_config,
    )
    payload = {
        "instruction": instruction_profile,
        "input": "",
        "output": record.sequence,
        "accession": record.accession_version or record.accession,
        "description": record.description,
        "organism": record.organism,
        "metadata": dict(record.metadata),
        "derived_labels": list(record.derived_labels),
        "derived_keywords": list(record.derived_keywords),
        "label_source": record.label_source,
        "origin": record.origin,
        "output_format": "single protein sequence",
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_instruction_jsonl_chunk(
    args: tuple[tuple[RefseqCompiledRecord, ...], str, MDCProfileCompilerConfig],
) -> tuple[str, int]:
    records, source_name, profile_config = args
    if not records:
        return "", 0

    lines = [
        _build_instruction_jsonl_line(
            record,
            source_name=source_name,
            profile_config=profile_config,
        )
        for record in records
    ]
    return "\n".join(lines) + "\n", len(lines)


def _dedupe_records_by_sequence(records: list[RefseqCompiledRecord]) -> list[RefseqCompiledRecord]:
    ordered: list[RefseqCompiledRecord] = []
    seen: set[str] = set()
    for record in records:
        if record.sequence_hash in seen:
            continue
        ordered.append(record)
        seen.add(record.sequence_hash)
    return ordered


def _content_hash_for_record(
    *,
    accession: str,
    accession_version: str,
    description: str,
    organism: str,
    sequence: str,
    metadata: dict[str, str],
) -> str:
    return _hash_payload(
        {
            "accession": accession,
            "accession_version": accession_version,
            "description": description,
            "organism": organism,
            "sequence": sequence,
            "metadata": metadata,
        }
    )


def _infer_record_origin(metadata: dict[str, str]) -> str:
    if "fasta_header" not in metadata:
        return "gpff_only"

    if any(
        field in metadata
        for field in (
            "keywords",
            "dbsource",
            "taxonomy",
            "product",
            "note",
            "coded_by",
            "gene",
            "gene_synonym",
            "chromosome",
            "plasmid",
            "organelle",
            "segment",
            "host",
        )
    ):
        return "paired"
    return "faa_only"


def _diff_source_hashes(
    previous_hashes: dict[str, str],
    current_hashes: dict[str, str],
) -> tuple[int, int, int, int]:
    new_count = 0
    updated_count = 0
    unchanged_count = 0

    for accession, content_hash in current_hashes.items():
        previous_hash = previous_hashes.get(accession)
        if previous_hash is None:
            new_count += 1
            continue
        if previous_hash != content_hash:
            updated_count += 1
            continue
        unchanged_count += 1

    removed_count = len(set(previous_hashes) - set(current_hashes))
    return new_count, updated_count, unchanged_count, removed_count


def _hash_payload(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _discover_archive_bundles(
    input_root: Path,
    *,
    bundle_key_root: Path | None = None,
) -> list[RefseqArchiveBundle]:
    effective_bundle_key_root = bundle_key_root or input_root
    gpff_by_key: dict[str, Path] = {}
    faa_by_key: dict[str, Path] = {}
    for path in input_root.rglob("*.gpff.gz"):
        gpff_by_key[_bundle_key(effective_bundle_key_root, path, ".gpff.gz")] = path
    for path in input_root.rglob("*.faa.gz"):
        faa_by_key[_bundle_key(effective_bundle_key_root, path, ".faa.gz")] = path

    bundles: list[RefseqArchiveBundle] = []
    for key in sorted(set(gpff_by_key) | set(faa_by_key)):
        relative = Path(key)
        group_name = relative.parent.name if relative.parent != Path(".") else effective_bundle_key_root.name
        bundles.append(
            RefseqArchiveBundle(
                key=key,
                group_name=group_name,
                gpff_path=gpff_by_key.get(key),
                faa_path=faa_by_key.get(key),
            )
        )
    return bundles


def _reuse_previous_bundle_keys(
    bundles: list[RefseqArchiveBundle],
    *,
    previous_bundle_states: dict[str, RefseqProcessedBundleState],
) -> list[RefseqArchiveBundle]:
    if not previous_bundle_states:
        return bundles

    previous_bundle_key_by_archive_path: dict[str, str] = {}
    for state in previous_bundle_states.values():
        for archive in state.archives:
            previous_bundle_key_by_archive_path[str(Path(archive.path))] = state.bundle_key

    normalized_bundles: list[RefseqArchiveBundle] = []
    for bundle in bundles:
        matching_bundle_keys = {
            previous_bundle_key_by_archive_path[str(archive_path)]
            for archive_path in (bundle.gpff_path, bundle.faa_path)
            if archive_path is not None and str(archive_path) in previous_bundle_key_by_archive_path
        }
        if len(matching_bundle_keys) == 1:
            normalized_bundles.append(
                RefseqArchiveBundle(
                    key=next(iter(matching_bundle_keys)),
                    group_name=bundle.group_name,
                    gpff_path=bundle.gpff_path,
                    faa_path=bundle.faa_path,
                )
            )
            continue
        normalized_bundles.append(bundle)
    return normalized_bundles


def _bundle_key(input_root: Path, path: Path, suffix: str) -> str:
    relative = path.relative_to(input_root).as_posix()
    return relative[: -len(suffix)]


def _iter_bundle_records(
    bundle: RefseqArchiveBundle,
    *,
    source_name: str,
) -> tuple[Iterator[tuple[RefseqProteinSourceRecord, str]], list[RefseqInputFileSummary]]:
    gpff_summary = (
        RefseqInputFileSummary(kind="gpff", path=str(bundle.gpff_path))
        if bundle.gpff_path is not None
        else None
    )
    faa_summary = (
        RefseqInputFileSummary(kind="faa", path=str(bundle.faa_path))
        if bundle.faa_path is not None
        else None
    )
    summaries = [summary for summary in (gpff_summary, faa_summary) if summary is not None]

    gpff_iter = (
        _iter_gpff_records(
            bundle.gpff_path,
            bundle.group_name,
            bundle.key,
            source_name=source_name,
            summary=gpff_summary,
        )
        if bundle.gpff_path is not None and gpff_summary is not None
        else iter(())
    )
    faa_iter = (
        _iter_faa_records(
            bundle.faa_path,
            bundle.group_name,
            bundle.key,
            source_name=source_name,
            summary=faa_summary,
        )
        if bundle.faa_path is not None and faa_summary is not None
        else iter(())
    )

    def generator() -> Iterator[tuple[RefseqProteinSourceRecord, str]]:
        nonlocal gpff_iter, faa_iter
        try:
            current_gpff = next(gpff_iter, None)
            current_faa = next(faa_iter, None)

            while current_gpff is not None or current_faa is not None:
                if current_gpff is None:
                    yield current_faa, "faa_only"
                    current_faa = next(faa_iter, None)
                    continue
                if current_faa is None:
                    yield current_gpff, "gpff_only"
                    current_gpff = next(gpff_iter, None)
                    continue

                if current_gpff.sort_key == current_faa.sort_key:
                    yield _merge_source_records(current_gpff, current_faa), "paired"
                    current_gpff = next(gpff_iter, None)
                    current_faa = next(faa_iter, None)
                    continue

                if current_gpff.sort_key < current_faa.sort_key:
                    yield current_gpff, "gpff_only"
                    current_gpff = next(gpff_iter, None)
                    continue

                yield current_faa, "faa_only"
                current_faa = next(faa_iter, None)
        finally:
            close_gpff_iter = getattr(gpff_iter, "close", None)
            if callable(close_gpff_iter):
                close_gpff_iter()
            close_faa_iter = getattr(faa_iter, "close", None)
            if callable(close_faa_iter):
                close_faa_iter()

    return generator(), summaries


def _iter_gpff_records(
    path: Path,
    group_name: str,
    bundle_key: str,
    *,
    source_name: str,
    summary: RefseqInputFileSummary,
) -> Iterator[RefseqProteinSourceRecord]:
    for block_lines in _iter_gpff_blocks(path, summary):
        record = _parse_gpff_block(
            block_lines,
            group_name=group_name,
            bundle_key=bundle_key,
            source_name=source_name,
        )
        if record is None:
            continue
        summary.record_count += 1
        yield record


def _iter_gpff_blocks(path: Path, summary: RefseqInputFileSummary) -> Iterator[list[str]]:
    block_lines: list[str] = []
    for line in _iter_safe_gzip_lines(path, summary):
        block_lines.append(line.rstrip("\r\n"))
        if line.strip() == "//":
            yield block_lines
            block_lines = []
    if block_lines:
        summary.dropped_incomplete_records += 1


def _iter_faa_records(
    path: Path,
    group_name: str,
    bundle_key: str,
    *,
    source_name: str,
    summary: RefseqInputFileSummary,
) -> Iterator[RefseqProteinSourceRecord]:
    for entry in iter_fasta_entries(_iter_safe_gzip_lines(path, summary)):
        summary.record_count += 1
        yield _build_faa_record(
            entry,
            group_name=group_name,
            bundle_key=bundle_key,
            source_name=source_name,
        )


def _iter_safe_gzip_lines(path: Path, summary: RefseqInputFileSummary) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        while True:
            try:
                line = handle.readline()
            except EOFError:
                summary.truncated = True
                break
            if line == "":
                break
            yield line


def _parse_gpff_block(
    block_lines: list[str],
    *,
    group_name: str,
    bundle_key: str,
    source_name: str,
) -> RefseqProteinSourceRecord | None:
    top_level_fields, organism, taxonomy_lines, feature_lines, sequence = _split_gpff_sections(block_lines)
    accession = _first_token(_first_field_value(top_level_fields, "ACCESSION"))
    accession_version = _first_token(_first_field_value(top_level_fields, "VERSION")) or accession
    if not accession_version:
        return None

    qualifiers = _parse_feature_qualifiers(feature_lines)
    description = _clean_field_text(_join_field_values(top_level_fields, "DEFINITION"))
    keywords = _clean_field_text(_join_field_values(top_level_fields, "KEYWORDS"))
    dbsource = _clean_field_text(_join_field_values(top_level_fields, "DBSOURCE"))
    source_label = _clean_field_text(_join_field_values(top_level_fields, "SOURCE"))
    resolved_organism = _clean_field_text(organism or source_label)
    taxonomy = _clean_field_text(" ".join(taxonomy_lines))

    metadata: dict[str, str] = {
        "dataset_group": group_name,
        "dataset_bundle": bundle_key,
        "source_name": source_name,
    }
    _merge_metadata_if_present(metadata, "description", description)
    _merge_metadata_if_present(metadata, "keywords", _strip_terminal_period(keywords))
    _merge_metadata_if_present(metadata, "dbsource", dbsource)
    _merge_metadata_if_present(metadata, "scientific_name", resolved_organism)
    _merge_metadata_if_present(metadata, "taxonomy", taxonomy)
    _merge_feature_metadata(metadata, qualifiers)

    return RefseqProteinSourceRecord(
        accession=_canonical_accession(accession_version),
        accession_version=accession_version,
        description=description or metadata.get("product", ""),
        organism=resolved_organism,
        sequence=sequence,
        metadata=metadata,
    )


def _split_gpff_sections(
    block_lines: list[str],
) -> tuple[dict[str, list[str]], str, list[str], list[str], str]:
    top_level_fields: dict[str, list[str]] = {}
    organism = ""
    taxonomy_lines: list[str] = []
    feature_lines: list[str] = []
    sequence_chunks: list[str] = []

    current_field: str | None = None
    in_features = False
    in_origin = False

    for raw_line in block_lines:
        line = raw_line.rstrip("\r\n")
        if not line or line == "//":
            continue

        if in_origin:
            letters_only = re.sub(r"[^A-Za-z]", "", line)
            if letters_only:
                sequence_chunks.append(letters_only.upper())
            continue

        if line.startswith("ORIGIN"):
            in_origin = True
            in_features = False
            current_field = None
            continue

        if line.startswith("FEATURES"):
            in_features = True
            current_field = None
            continue

        if in_features:
            feature_lines.append(line)
            continue

        if line.startswith("  ORGANISM"):
            organism = line[12:].strip()
            current_field = "ORGANISM"
            continue

        field_name = line[:12].strip() if len(line) >= 12 else line.strip()
        if field_name and not line.startswith(" "):
            value = line[12:].strip() if len(line) > 12 else ""
            top_level_fields.setdefault(field_name, []).append(value)
            current_field = field_name
            continue

        if current_field == "ORGANISM":
            continuation = line[12:].strip() if len(line) > 12 else line.strip()
            if continuation:
                taxonomy_lines.append(continuation)
            continue

        if current_field is not None:
            continuation = line[12:].strip() if len(line) > 12 else line.strip()
            if continuation:
                top_level_fields.setdefault(current_field, []).append(continuation)

    return top_level_fields, organism, taxonomy_lines, feature_lines, "".join(sequence_chunks)


def _parse_feature_qualifiers(feature_lines: list[str]) -> dict[str, tuple[str, ...]]:
    tracked_features = {"source", "protein", "cds"}
    tracked_qualifiers = {
        "gene",
        "gene_synonym",
        "product",
        "note",
        "coded_by",
        "chromosome",
        "plasmid",
        "organelle",
        "segment",
        "host",
    }
    collected: dict[str, list[str]] = {}
    current_feature: str | None = None
    current_qualifier: str | None = None

    for line in feature_lines:
        if len(line) >= 21 and line.startswith("     ") and line[5:21].strip():
            current_feature = line[5:21].strip().lower()
            current_qualifier = None
            continue

        if current_feature not in tracked_features:
            continue

        if line.startswith("                     /"):
            qualifier_name, qualifier_value = _parse_qualifier_payload(line[21:].strip())
            normalized_name = qualifier_name.lower()
            if normalized_name not in tracked_qualifiers:
                current_qualifier = None
                continue

            collected.setdefault(normalized_name, [])
            if qualifier_value:
                collected[normalized_name].append(qualifier_value)
            current_qualifier = normalized_name
            continue

        if line.startswith("                     ") and current_qualifier is not None:
            continuation = line[21:].strip().strip('"')
            if not continuation or current_qualifier not in collected or not collected[current_qualifier]:
                continue
            collected[current_qualifier][-1] = f"{collected[current_qualifier][-1]} {continuation}".strip()

    return {
        name: tuple(_dedupe_preserve_order(values))
        for name, values in collected.items()
        if values
    }


def _parse_qualifier_payload(payload: str) -> tuple[str, str]:
    normalized = payload.lstrip("/")
    if "=" not in normalized:
        return normalized.strip(), "true"
    key, value = normalized.split("=", 1)
    return key.strip(), value.strip().strip('"')


def _merge_feature_metadata(metadata: dict[str, str], qualifiers: dict[str, tuple[str, ...]]) -> None:
    for field_name in (
        "gene",
        "gene_synonym",
        "product",
        "note",
        "coded_by",
        "chromosome",
        "plasmid",
        "organelle",
        "segment",
        "host",
    ):
        values = qualifiers.get(field_name, ())
        cleaned = _clean_field_text(", ".join(value for value in values if value.strip()))
        if cleaned:
            metadata[field_name] = cleaned


def _build_faa_record(
    entry: ParsedFastaEntry,
    *,
    group_name: str,
    bundle_key: str,
    source_name: str,
) -> RefseqProteinSourceRecord:
    description, organism = _split_fasta_header(entry.header)
    accession_version = entry.accession
    metadata: dict[str, str] = {
        "dataset_group": group_name,
        "dataset_bundle": bundle_key,
        "fasta_header": entry.header,
        "source_name": source_name,
    }
    _merge_metadata_if_present(metadata, "description", description)
    _merge_metadata_if_present(metadata, "scientific_name", organism)
    return RefseqProteinSourceRecord(
        accession=_canonical_accession(accession_version),
        accession_version=accession_version,
        description=description,
        organism=organism,
        sequence=entry.sequence,
        metadata=metadata,
    )


def _split_fasta_header(header: str) -> tuple[str, str]:
    parts = header.split(maxsplit=1)
    description = parts[1].strip() if len(parts) > 1 else ""
    organism = ""
    bracket_match = re.search(r"\[([^\[\]]+)\]\s*$", description)
    if bracket_match is not None:
        organism = bracket_match.group(1).strip()
        description = description[: bracket_match.start()].strip()
    return description, organism


def _merge_source_records(
    gpff_record: RefseqProteinSourceRecord,
    faa_record: RefseqProteinSourceRecord,
) -> RefseqProteinSourceRecord:
    merged_metadata = dict(gpff_record.metadata)
    for key, value in faa_record.metadata.items():
        if value and not merged_metadata.get(key):
            merged_metadata[key] = value

    gpff_sequence = normalize_sequence(gpff_record.sequence, sequence_type="protein")
    faa_sequence = normalize_sequence(faa_record.sequence, sequence_type="protein")
    if gpff_sequence and faa_sequence and gpff_sequence != faa_sequence:
        merged_metadata["sequence_mismatch"] = "true"

    merged_description = gpff_record.description or faa_record.description
    merged_organism = gpff_record.organism or faa_record.organism
    merged_sequence = faa_record.sequence or gpff_record.sequence
    merged_accession_version = faa_record.accession_version or gpff_record.accession_version

    return RefseqProteinSourceRecord(
        accession=_canonical_accession(merged_accession_version),
        accession_version=merged_accession_version,
        description=merged_description,
        organism=merged_organism,
        sequence=merged_sequence,
        metadata=merged_metadata,
    )


def _first_field_value(fields: dict[str, list[str]], field_name: str) -> str:
    values = fields.get(field_name, [])
    return values[0] if values else ""


def _join_field_values(fields: dict[str, list[str]], field_name: str) -> str:
    return " ".join(value for value in fields.get(field_name, []) if value)


def _first_token(value: str) -> str:
    return value.split()[0].strip() if value.strip() else ""


def _canonical_accession(value: str) -> str:
    token = value.strip()
    if not token:
        return ""
    prefix, separator, suffix = token.rpartition(".")
    if separator and suffix.isdigit():
        return prefix
    return token


def _accession_version_number(value: str) -> int:
    token = value.strip()
    if not token:
        return -1
    prefix, separator, suffix = token.rpartition(".")
    if separator and suffix.isdigit() and prefix:
        return int(suffix)
    return -1


def _coerce_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_field_text(value: str) -> str:
    cleaned = value.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
    return " ".join(cleaned.split())


def _clean_optional_text(value: object) -> str | None:
    cleaned = _clean_field_text(str(value))
    return cleaned or None


def _strip_terminal_period(value: str) -> str:
    stripped = value.strip()
    return stripped[:-1].strip() if stripped.endswith(".") else stripped


def _merge_metadata_if_present(metadata: dict[str, str], key: str, value: str) -> None:
    cleaned = _clean_field_text(value)
    if cleaned:
        metadata[key] = cleaned


def _tuple_of_strings(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(_dedupe_preserve_order(str(value) for value in values))


def _dedupe_preserve_order(values) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_field_text(value)
        if not cleaned or cleaned in seen:
            continue
        ordered.append(cleaned)
        seen.add(cleaned)
    return ordered

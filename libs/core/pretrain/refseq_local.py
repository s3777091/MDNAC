from __future__ import annotations

import os
import gzip
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from libs.core.pretrain.profiled import (
    MDCProfileCompilerConfig,
    _infer_profile_labels,
    build_profile_text_from_sequence_metadata,
)
from libs.data.training.kmer import _normalize_sequence as normalize_sequence
from libs.data.training.tokenizer import SequenceTokenizer, SequenceTokenizerTextTrainingStats
from libs.data.utilities.parsers import ParsedFastaEntry, iter_fasta_entries
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
TokenizerProgressCallback = Callable[[dict[str, object]], None]


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
    train_text_path: str
    tokenizer_map_path: str
    instruction_path: str
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
    max_records: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "input_root": self.input_root,
            "output_dir": self.output_dir,
            "train_text_path": self.train_text_path,
            "tokenizer_map_path": self.tokenizer_map_path,
            "instruction_path": self.instruction_path,
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
            "max_records": self.max_records,
        }


@dataclass(slots=True)
class RefseqTokenizerMapBuildSummary:
    output_dir: str
    train_text_path: str
    tokenizer_map_path: str
    record_count: int
    tokenizer_train_record_count: int
    vocab_size_requested: int

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "train_text_path": self.train_text_path,
            "tokenizer_map_path": self.tokenizer_map_path,
            "record_count": self.record_count,
            "tokenizer_train_record_count": self.tokenizer_train_record_count,
            "vocab_size_requested": self.vocab_size_requested,
        }


@dataclass(slots=True)
class RefseqLocalArtifactDedupeSummary:
    output_dir: str
    train_text_path: str
    instruction_path: str
    original_train_line_count: int
    deduped_train_line_count: int
    removed_train_duplicates: int
    original_instruction_line_count: int
    deduped_instruction_line_count: int
    removed_instruction_duplicates: int
    train_text_changed: bool
    instruction_changed: bool
    dry_run: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "train_text_path": self.train_text_path,
            "instruction_path": self.instruction_path,
            "original_train_line_count": self.original_train_line_count,
            "deduped_train_line_count": self.deduped_train_line_count,
            "removed_train_duplicates": self.removed_train_duplicates,
            "original_instruction_line_count": self.original_instruction_line_count,
            "deduped_instruction_line_count": self.deduped_instruction_line_count,
            "removed_instruction_duplicates": self.removed_instruction_duplicates,
            "train_text_changed": self.train_text_changed,
            "instruction_changed": self.instruction_changed,
            "dry_run": self.dry_run,
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
    tokenizer_train_line_limit: int | None = None,
    tokenizer_resume: bool = False,
    tokenizer_workers: int = 1,
    tokenizer_progress_callback: TokenizerProgressCallback | None = None,
) -> RefseqLocalBuildSummary:
    del kmer_size, profile_sample_char_limit

    if instruction_min_proteins <= 0:
        raise ValueError("instruction_min_proteins must be greater than 0.")
    _validate_optional_positive_int(
        tokenizer_train_line_limit,
        name="tokenizer_train_line_limit",
    )
    effective_workers = _resolve_worker_count(workers)
    skipped_artifact_names = _normalize_output_artifact_names(skip_artifacts)

    requested_input_root = Path(input_root)
    resolved_output_dir = Path(output_dir)
    if not requested_input_root.exists():
        raise FileNotFoundError(f"Input directory was not found: {requested_input_root}")

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_input_root = _resolve_scoped_input_root(requested_input_root, resolved_output_dir)
    train_text_path = resolved_output_dir / TRAIN_TEXT_ARTIFACT_NAME
    tokenizer_map_path = resolved_output_dir / TOKENIZER_MAP_ARTIFACT_NAME
    instruction_path = resolved_output_dir / INSTRUCTION_ARTIFACT_NAME
    legacy_source_index_path = resolved_output_dir / "source_index.json"
    obsolete_summary_path = resolved_output_dir / "summary.json"
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
    bundles = _discover_archive_bundles(resolved_input_root)
    if not bundles:
        raise ValueError(f"No .gpff.gz or .faa.gz files were found under {resolved_input_root}.")

    candidate_by_accession: dict[str, RefseqCandidateRecord] = {}
    duplicate_accession_count = 0
    skipped_empty_sequence_count = 0
    truncated_input_count = 0
    stop_requested = False

    for bundle in bundles:
        bundle_records, bundle_summaries = _iter_bundle_records(
            bundle,
            source_name=source_name,
        )
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
                    break
        finally:
            close_bundle_records = getattr(bundle_records, "close", None)
            if callable(close_bundle_records):
                close_bundle_records()

        if stop_requested:
            break

    compiled_records = _compile_refseq_records(
        candidate_by_accession,
        source_name=source_name,
        profile_config=effective_profile_config,
        workers=effective_workers,
    )
    if not compiled_records:
        raise ValueError("No training records were produced from the provided RefSeq archive directory.")

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
            compiled_records,
            instruction_min_proteins=instruction_min_proteins,
        )
    )
    paired_record_count = sum(1 for record in compiled_records if record.origin == "paired")
    gpff_only_record_count = sum(1 for record in compiled_records if record.origin == "gpff_only")
    faa_only_record_count = sum(1 for record in compiled_records if record.origin == "faa_only")
    sequence_mismatch_count = sum(
        1 for record in compiled_records if record.metadata.get("sequence_mismatch") == "true"
    )

    if write_train_text:
        _append_train_text_artifact(train_text_path, kept_records)

    if write_tokenizer_map:
        tokenizer_map_text, tokenizer_training_stats = _render_tokenizer_map_text_from_train_path(
            train_text_path,
            source_name=source_name,
            vocab_size=effective_vocab_size,
            builder_metadata=builder_metadata,
            tokenizer_train_line_limit=tokenizer_train_line_limit,
            tokenizer_resume=tokenizer_resume,
            tokenizer_workers=tokenizer_workers,
            cache_dir=resolved_output_dir,
            progress_callback=tokenizer_progress_callback,
        )
        if tokenizer_training_stats.record_count <= 0:
            raise ValueError(f"Cannot build tokenizer_map.json from an empty train.txt file: {train_text_path}")
        _write_text_if_changed(tokenizer_map_path, tokenizer_map_text)

    if write_instruction_jsonl:
        _append_instruction_jsonl_artifact(
            instruction_path,
            compiled_records,
            source_name=source_name,
            profile_config=instruction_profile_config,
            workers=effective_workers,
        )

    legacy_source_index_path.unlink(missing_ok=True)
    obsolete_summary_path.unlink(missing_ok=True)

    summary = RefseqLocalBuildSummary(
        input_root=str(resolved_input_root),
        output_dir=str(resolved_output_dir),
        train_text_path=str(train_text_path),
        tokenizer_map_path=str(tokenizer_map_path),
        instruction_path=str(instruction_path),
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
        max_records=max_records,
    )
    return summary


def rebuild_local_refseq_tokenizer_map_from_train_text(
    output_dir: Path | str,
    *,
    source_name: str = "refseq",
    vocab_size: int | None = None,
    profile_vocab_size: int = 256,
    tokenizer_train_line_limit: int | None = None,
    tokenizer_resume: bool = False,
    tokenizer_workers: int = 1,
    tokenizer_progress_callback: TokenizerProgressCallback | None = None,
) -> RefseqTokenizerMapBuildSummary:
    _validate_optional_positive_int(
        tokenizer_train_line_limit,
        name="tokenizer_train_line_limit",
    )
    resolved_output_dir = Path(output_dir)
    train_text_path = resolved_output_dir / TRAIN_TEXT_ARTIFACT_NAME
    tokenizer_map_path = resolved_output_dir / TOKENIZER_MAP_ARTIFACT_NAME
    if not train_text_path.exists():
        raise FileNotFoundError(f"train.txt was not found: {train_text_path}")

    effective_vocab_size = profile_vocab_size if vocab_size is None else vocab_size
    tokenizer_map_text, tokenizer_training_stats = _render_tokenizer_map_text_from_train_path(
        train_text_path,
        source_name=source_name,
        vocab_size=effective_vocab_size,
        builder_metadata={
            "type": "local_refseq_sequence_only_from_train_txt",
            "source_name": source_name,
            "tokenizer_type": "bpe",
            "vocab_size_requested": effective_vocab_size,
            "rebuilt_from_existing_train_text": True,
        },
        tokenizer_train_line_limit=tokenizer_train_line_limit,
        tokenizer_resume=tokenizer_resume,
        tokenizer_workers=tokenizer_workers,
        cache_dir=resolved_output_dir,
        progress_callback=tokenizer_progress_callback,
    )
    if tokenizer_training_stats.record_count <= 0:
        raise ValueError(f"Cannot build tokenizer_map.json from an empty train.txt file: {train_text_path}")
    if tokenizer_training_stats.tokenizer_train_record_count <= 0:
        raise ValueError(f"Cannot build tokenizer_map.json from an empty tokenizer training sample: {train_text_path}")
    _write_text_if_changed(tokenizer_map_path, tokenizer_map_text)
    return RefseqTokenizerMapBuildSummary(
        output_dir=str(resolved_output_dir),
        train_text_path=str(train_text_path),
        tokenizer_map_path=str(tokenizer_map_path),
        record_count=tokenizer_training_stats.record_count,
        tokenizer_train_record_count=tokenizer_training_stats.tokenizer_train_record_count,
        vocab_size_requested=effective_vocab_size,
    )


def dedupe_local_refseq_sequence_only_artifacts(
    output_dir: Path | str,
    *,
    dry_run: bool = False,
) -> RefseqLocalArtifactDedupeSummary:
    resolved_output_dir = Path(output_dir)
    train_text_path = resolved_output_dir / TRAIN_TEXT_ARTIFACT_NAME
    instruction_path = resolved_output_dir / INSTRUCTION_ARTIFACT_NAME

    if not train_text_path.exists():
        raise FileNotFoundError(f"train.txt was not found: {train_text_path}")
    if not instruction_path.exists():
        raise FileNotFoundError(f"instruction.jsonl was not found: {instruction_path}")

    original_train_line_count, deduped_train_line_count, train_text_changed = _dedupe_text_file_in_place(
        train_text_path,
        dry_run=dry_run,
    )
    (
        original_instruction_line_count,
        deduped_instruction_line_count,
        instruction_changed,
    ) = _dedupe_text_file_in_place(
        instruction_path,
        dry_run=dry_run,
    )
    summary = RefseqLocalArtifactDedupeSummary(
        output_dir=str(resolved_output_dir),
        train_text_path=str(train_text_path),
        instruction_path=str(instruction_path),
        original_train_line_count=original_train_line_count,
        deduped_train_line_count=deduped_train_line_count,
        removed_train_duplicates=original_train_line_count - deduped_train_line_count,
        original_instruction_line_count=original_instruction_line_count,
        deduped_instruction_line_count=deduped_instruction_line_count,
        removed_instruction_duplicates=original_instruction_line_count - deduped_instruction_line_count,
        train_text_changed=train_text_changed,
        instruction_changed=instruction_changed,
        dry_run=dry_run,
    )
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


def _validate_optional_positive_int(value: int | None, *, name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be greater than 0 when provided.")


def _dedupe_text_file_in_place(path: Path, *, dry_run: bool = False) -> tuple[int, int, bool]:
    temp_path = path.with_name(f"{path.name}.dedupe.tmp")
    temp_path.unlink(missing_ok=True)

    seen: set[bytes] = set()
    original_nonempty_line_count = 0
    deduped_nonempty_line_count = 0
    with path.open("rb") as source_handle, temp_path.open("wb") as target_handle:
        for raw_line in source_handle:
            line = raw_line.strip()
            if not line:
                continue
            original_nonempty_line_count += 1
            if line in seen:
                continue
            seen.add(line)
            deduped_nonempty_line_count += 1
            target_handle.write(line)
            target_handle.write(b"\n")

    changed = not _paths_have_same_content(path, temp_path)
    if dry_run or not changed:
        temp_path.unlink(missing_ok=True)
    else:
        temp_path.replace(path)

    return original_nonempty_line_count, deduped_nonempty_line_count, changed


def _paths_have_same_content(left_path: Path, right_path: Path, *, chunk_size: int = 1_048_576) -> bool:
    if not left_path.exists() or not right_path.exists():
        return False
    if left_path.stat().st_size != right_path.stat().st_size:
        return False

    with left_path.open("rb") as left_handle, right_path.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(chunk_size)
            right_chunk = right_handle.read(chunk_size)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _render_train_text(records: list[RefseqCompiledRecord]) -> str:
    return "\n".join(record.sequence_train_line for record in records) + "\n"


def _render_tokenizer_map_text_from_train_path(
    train_text_path: Path,
    *,
    source_name: str,
    vocab_size: int,
    builder_metadata: dict[str, object],
    tokenizer_train_line_limit: int | None,
    tokenizer_resume: bool,
    tokenizer_workers: int,
    cache_dir: Path,
    progress_callback: TokenizerProgressCallback | None,
) -> tuple[str, SequenceTokenizerTextTrainingStats]:
    tokenizer = SequenceTokenizer.from_sequence_type("protein")
    training_stats = tokenizer.train_from_text_file(
        train_text_path,
        vocab_size=vocab_size,
        line_limit=tokenizer_train_line_limit,
        cache_dir=cache_dir,
        progress_callback=progress_callback,
        resume=tokenizer_resume,
        worker_count=tokenizer_workers,
    )
    tokenizer_map_payload = json.loads(
        render_tokenizer_map_payload(
            source_name=source_name,
            record_count=training_stats.record_count,
            tokenizer=tokenizer,
        )
    )
    tokenizer_map_payload["builder"] = {
        **builder_metadata,
        "tokenizer_train_record_count": training_stats.tokenizer_train_record_count,
        "tokenizer_train_line_limit": tokenizer_train_line_limit,
        "tokenizer_resume": tokenizer_resume,
        "tokenizer_workers": tokenizer_workers,
        "vocab_size_actual": tokenizer.vocab_size,
    }
    return json.dumps(tokenizer_map_payload, ensure_ascii=False, indent=2) + "\n", training_stats


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


def _append_train_text_artifact(path: Path, records: list[RefseqCompiledRecord]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.sequence_train_line)
            handle.write("\n")


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

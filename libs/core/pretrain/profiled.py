from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from libs.core.fusion import FusedVocabularyLayout, ProfileSequenceBatchBuilder, ProfileSequenceFusionConfig
from libs.core.interfaces import CausalLMBatch, FusedProfileSequenceBatch
from libs.data.entities import PreparationSessionArtifact
from libs.data.training import KmerTokenizer, ProfileBPETokenizer, ProfileSequencePair
from libs.data.training.raw_pipeline.defaults import DEFAULT_KEYWORD_RULES
from libs.data.training.raw_pipeline.helpers import match_rules
from libs.data.training.raw_pipeline.models import ProfileKeywordRule


MDC_PROFILE_SEQUENCE_FORMAT = "mdc_profile_sequence_v1"
PROFILE_START_TOKEN = "<|profile|>"
TRAIN_SEPARATOR_TOKEN = "<|sep|>"
TRAIN_END_TOKEN = "<|endoftext|>"
PROTEIN_SEQUENCE_START_TOKEN = "<|protein|>"
SUPPORTED_SEQUENCE_TYPES = ("protein",)
RESERVED_TRAIN_TOKENS = (
    PROFILE_START_TOKEN,
    TRAIN_SEPARATOR_TOKEN,
    TRAIN_END_TOKEN,
    PROTEIN_SEQUENCE_START_TOKEN,
)


@dataclass(slots=True, frozen=True)
class MDCProfileSequenceRecord:
    profile: str
    sequence: str
    sequence_type: str = "protein"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_sequence_type = _normalize_sequence_type(self.sequence_type)
        _validate_field_text("profile", self.profile)
        _validate_field_text("sequence", self.sequence)
        object.__setattr__(self, "sequence_type", normalized_sequence_type)

    def to_train_line(self) -> str:
        return build_profile_sequence_train_line(
            profile=self.profile,
            sequence=self.sequence,
            sequence_type=self.sequence_type,
        )


@dataclass(slots=True, frozen=True)
class MDCProfileSequenceTextArtifact:
    output_dir: str
    train_text_path: str
    tokenizer_map_path: str
    record_count: int
    sequence_type: str
    kmer_size: int
    profile_vocab_size: int
    sequence_vocab_size: int
    format_version: str = MDC_PROFILE_SEQUENCE_FORMAT


@dataclass(slots=True)
class MDCEncodedProfileSequenceExample:
    profile_input_ids: torch.Tensor
    sequence_input_ids: torch.Tensor
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class MDCProfileCompilerConfig:
    task_name: str = "conditional sequence generation"
    metadata_fields: tuple[str, ...] = (
        "biomol",
        "moltype",
        "genome",
        "completeness",
        "topology",
        "tech",
        "assemblyacc",
        "biosample",
        "strain",
        "scientific_name",
    )
    label_signal_fields: tuple[str, ...] = (
        "fasta_header",
        "description",
        "function",
        "pathway",
        "note",
        "product",
        "strain",
    )
    keyword_rules: tuple[ProfileKeywordRule, ...] = DEFAULT_KEYWORD_RULES
    include_task: bool = True
    include_labels: bool = True
    include_label_source: bool = True
    include_matched_keywords: bool = True
    include_description: bool = True
    include_organism: bool = True
    include_sequence_type: bool = True
    include_source_name: bool = True
    generic_label_source_name: str = "structural metadata"


@dataclass(slots=True)
class MDCProfileSequencePretrainArtifacts:
    train_text_path: Path
    tokenizer_map_path: Path
    train_text: str
    examples: tuple[MDCProfileSequenceRecord, ...]
    profile_tokenizer: ProfileBPETokenizer
    sequence_tokenizer: KmerTokenizer
    layout: FusedVocabularyLayout
    format_version: str = MDC_PROFILE_SEQUENCE_FORMAT

    @classmethod
    def from_files(
        cls,
        train_text_path: Path | str,
        tokenizer_map_path: Path | str,
    ) -> "MDCProfileSequencePretrainArtifacts":
        resolved_train_text_path = Path(train_text_path)
        resolved_tokenizer_map_path = Path(tokenizer_map_path)
        train_text = resolved_train_text_path.read_text(encoding="utf-8")
        tokenizer_map = json.loads(resolved_tokenizer_map_path.read_text(encoding="utf-8"))

        if tokenizer_map.get("format") != MDC_PROFILE_SEQUENCE_FORMAT:
            raise ValueError(
                "tokenizer_map.json does not match the MDC profile-aware train.txt format. "
                f"Expected format '{MDC_PROFILE_SEQUENCE_FORMAT}'."
            )

        profile_tokenizer = _load_profile_tokenizer_from_payload(tokenizer_map["profile_tokenizer"])
        sequence_tokenizer = _load_kmer_tokenizer_from_payload(tokenizer_map["sequence_tokenizer"])
        layout = _load_layout_from_payload(tokenizer_map["layout"])
        examples = tuple(
            parse_profile_sequence_train_line(line)
            for line in train_text.splitlines()
            if line.strip()
        )

        expected_count = int(tokenizer_map.get("record_count", len(examples)))
        if expected_count != len(examples):
            raise ValueError(
                f"tokenizer_map.json record_count ({expected_count}) does not match train.txt line count "
                f"({len(examples)})."
            )

        dataset_sequence_type = str(tokenizer_map.get("sequence_type", sequence_tokenizer.sequence_type))
        if dataset_sequence_type != sequence_tokenizer.sequence_type:
            raise ValueError(
                "tokenizer_map.json sequence_type does not match the embedded sequence tokenizer configuration."
            )

        for example in examples:
            if example.sequence_type != dataset_sequence_type:
                raise ValueError(
                    "train.txt contains mixed or unexpected sequence types for the configured tokenizer."
                )

        return cls(
            train_text_path=resolved_train_text_path,
            tokenizer_map_path=resolved_tokenizer_map_path,
            train_text=train_text,
            examples=examples,
            profile_tokenizer=profile_tokenizer,
            sequence_tokenizer=sequence_tokenizer,
            layout=layout,
            format_version=str(tokenizer_map.get("format", MDC_PROFILE_SEQUENCE_FORMAT)),
        )

    @classmethod
    def from_directory(cls, directory: Path | str) -> "MDCProfileSequencePretrainArtifacts":
        resolved_directory = Path(directory)
        return cls.from_files(
            train_text_path=resolved_directory / "train.txt",
            tokenizer_map_path=resolved_directory / "tokenizer_map.json",
        )

    @property
    def record_count(self) -> int:
        return len(self.examples)

    @property
    def sequence_type(self) -> str:
        return self.sequence_tokenizer.sequence_type

    @property
    def profile_vocab_size(self) -> int:
        return self.profile_tokenizer.vocab_size

    @property
    def sequence_vocab_size(self) -> int:
        return self.sequence_tokenizer.vocab_size

    @property
    def kmer_size(self) -> int:
        return self.sequence_tokenizer.kmer_size

    def encode_record(self, record: MDCProfileSequenceRecord) -> MDCEncodedProfileSequenceExample:
        if record.sequence_type != self.sequence_type:
            raise ValueError(
                f"Record sequence_type '{record.sequence_type}' does not match artifact sequence_type "
                f"'{self.sequence_type}'."
            )

        metadata = {
            "profile": record.profile,
            "sequence": record.sequence,
            "sequence_type": record.sequence_type,
            **dict(record.metadata),
        }

        return MDCEncodedProfileSequenceExample(
            profile_input_ids=torch.tensor(
                self.profile_tokenizer.encode(record.profile, add_bos=True, add_eos=True),
                dtype=torch.long,
            ),
            sequence_input_ids=torch.tensor(
                self.sequence_tokenizer.encode(record.sequence, add_bos=True, add_eos=True),
                dtype=torch.long,
            ),
            metadata=metadata,
        )

    def build_raw_tensor_payload(
        self,
        encoded_examples: Sequence[MDCEncodedProfileSequenceExample],
    ) -> dict[str, object]:
        if not encoded_examples:
            raise ValueError("encoded_examples must not be empty.")

        profile_input_ids, profile_attention_mask = _pad_token_tensors(
            [example.profile_input_ids for example in encoded_examples],
            pad_token_id=self.profile_tokenizer.pad_token_id,
        )
        sequence_input_ids, sequence_attention_mask = _pad_token_tensors(
            [example.sequence_input_ids for example in encoded_examples],
            pad_token_id=self.sequence_tokenizer.pad_token_id,
        )

        result: dict[str, object] = {
            "profile_input_ids": profile_input_ids,
            "profile_attention_mask": profile_attention_mask,
            "sequence_input_ids": sequence_input_ids,
            "sequence_attention_mask": sequence_attention_mask,
            "metadata": [dict(example.metadata) for example in encoded_examples],
            "config": {
                "profile_vocab_size": self.layout.profile_vocab_size,
                "sequence_vocab_size": self.layout.sequence_vocab_size,
            },
        }

        return result

    def build_fused_batch(
        self,
        encoded_examples: Sequence[MDCEncodedProfileSequenceExample],
        *,
        fusion_config: ProfileSequenceFusionConfig | None = None,
    ) -> FusedProfileSequenceBatch:
        payload = self.build_raw_tensor_payload(encoded_examples)
        builder = ProfileSequenceBatchBuilder(layout=self.layout, config=fusion_config)
        return builder.build_from_raw_tensor_payload(payload)

    def build_causal_lm_batch(
        self,
        encoded_examples: Sequence[MDCEncodedProfileSequenceExample],
        *,
        fusion_config: ProfileSequenceFusionConfig | None = None,
        train_on_prompt: bool = False,
        include_separator_in_loss: bool = False,
    ) -> CausalLMBatch:
        fused_batch = self.build_fused_batch(encoded_examples, fusion_config=fusion_config)
        return fused_batch.to_causal_lm_batch(
            train_on_prompt=train_on_prompt,
            include_separator_in_loss=include_separator_in_loss,
        )

    def decode_profile(self, token_ids: Sequence[int] | torch.Tensor, skip_special: bool = False) -> str:
        normalized_token_ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        return self.profile_tokenizer.decode(normalized_token_ids, skip_special=skip_special)

    def decode_sequence(self, token_ids: Sequence[int] | torch.Tensor, skip_special: bool = True) -> str:
        normalized_token_ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        return self.sequence_tokenizer.decode(normalized_token_ids, skip_special=skip_special)


class MDCProfileSequencePretrainDataset(Dataset[MDCEncodedProfileSequenceExample]):
    def __init__(self, artifacts: MDCProfileSequencePretrainArtifacts) -> None:
        self.artifacts = artifacts
        self._encoded_examples = [artifacts.encode_record(record) for record in artifacts.examples]

    @classmethod
    def from_artifacts(
        cls,
        artifacts: MDCProfileSequencePretrainArtifacts,
    ) -> "MDCProfileSequencePretrainDataset":
        return cls(artifacts=artifacts)

    @classmethod
    def from_files(
        cls,
        train_text_path: Path | str,
        tokenizer_map_path: Path | str,
    ) -> "MDCProfileSequencePretrainDataset":
        artifacts = MDCProfileSequencePretrainArtifacts.from_files(train_text_path, tokenizer_map_path)
        return cls.from_artifacts(artifacts)

    @classmethod
    def from_directory(
        cls,
        directory: Path | str,
    ) -> "MDCProfileSequencePretrainDataset":
        artifacts = MDCProfileSequencePretrainArtifacts.from_directory(directory)
        return cls.from_artifacts(artifacts)

    def __len__(self) -> int:
        return len(self._encoded_examples)

    def __getitem__(self, index: int) -> MDCEncodedProfileSequenceExample:
        return self._encoded_examples[index]


class MDCProfileSequenceBatchCollator:
    def __init__(
        self,
        artifacts: MDCProfileSequencePretrainArtifacts,
        *,
        fusion_config: ProfileSequenceFusionConfig | None = None,
        train_on_prompt: bool = False,
        include_separator_in_loss: bool = False,
    ) -> None:
        self.artifacts = artifacts
        self.fusion_config = fusion_config
        self.train_on_prompt = train_on_prompt
        self.include_separator_in_loss = include_separator_in_loss

    def __call__(self, batch: Sequence[MDCEncodedProfileSequenceExample]) -> CausalLMBatch:
        return self.artifacts.build_causal_lm_batch(
            batch,
            fusion_config=self.fusion_config,
            train_on_prompt=self.train_on_prompt,
            include_separator_in_loss=self.include_separator_in_loss,
        )


def save_mdc_profile_sequence_pretrain_artifacts(
    records: Sequence[MDCProfileSequenceRecord | ProfileSequencePair],
    output_dir: Path | str,
    *,
    sequence_type: str = "protein",
    kmer_size: int = 3,
    profile_vocab_size: int = 256,
) -> MDCProfileSequenceTextArtifact:
    resolved_records = _coerce_records(records, default_sequence_type=sequence_type)
    if not resolved_records:
        raise ValueError("records must not be empty.")

    resolved_sequence_type = _normalize_sequence_type(sequence_type)
    for record in resolved_records:
        if record.sequence_type != resolved_sequence_type:
            raise ValueError(
                "All records must share the same sequence_type when using the MDC profile-aware text format."
            )

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    train_text = "\n".join(record.to_train_line() for record in resolved_records) + "\n"
    profile_corpus = "\n".join(record.profile for record in resolved_records) + "\n"

    profile_tokenizer = ProfileBPETokenizer.from_text(profile_corpus, vocab_size=profile_vocab_size)
    sequence_tokenizer = KmerTokenizer.from_sequences(
        (record.sequence for record in resolved_records),
        kmer_size=kmer_size,
        sequence_type=resolved_sequence_type,
    )

    layout = FusedVocabularyLayout(
        profile_vocab_size=profile_tokenizer.vocab_size,
        sequence_vocab_size=sequence_tokenizer.vocab_size,
    )

    train_text_path = resolved_output_dir / "train.txt"
    tokenizer_map_path = resolved_output_dir / "tokenizer_map.json"

    train_text_path.write_text(train_text, encoding="utf-8")
    tokenizer_map_path.write_text(
        json.dumps(
            _build_tokenizer_map_payload(
                record_count=len(resolved_records),
                sequence_type=resolved_sequence_type,
                profile_tokenizer=profile_tokenizer,
                sequence_tokenizer=sequence_tokenizer,
                layout=layout,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return MDCProfileSequenceTextArtifact(
        output_dir=str(resolved_output_dir),
        train_text_path=str(train_text_path),
        tokenizer_map_path=str(tokenizer_map_path),
        record_count=len(resolved_records),
        sequence_type=resolved_sequence_type,
        kmer_size=sequence_tokenizer.kmer_size,
        profile_vocab_size=profile_tokenizer.vocab_size,
        sequence_vocab_size=sequence_tokenizer.vocab_size,
    )


def build_profile_sequence_train_line(
    *,
    profile: str,
    sequence: str,
    sequence_type: str = "protein",
) -> str:
    normalized_sequence_type = _normalize_sequence_type(sequence_type)
    _validate_field_text("profile", profile)
    _validate_field_text("sequence", sequence)
    sequence_start_token = _sequence_start_token(normalized_sequence_type)
    return f"{PROFILE_START_TOKEN}{profile}{TRAIN_SEPARATOR_TOKEN}{sequence_start_token}{sequence}{TRAIN_END_TOKEN}"


def parse_profile_sequence_train_line(line: str) -> MDCProfileSequenceRecord:
    stripped_line = line.rstrip("\r\n")
    if not stripped_line.startswith(PROFILE_START_TOKEN):
        raise ValueError("train.txt line must start with <|profile|>.")
    if not stripped_line.endswith(TRAIN_END_TOKEN):
        raise ValueError("train.txt line must end with <|endoftext|>.")

    body = stripped_line[len(PROFILE_START_TOKEN) : -len(TRAIN_END_TOKEN)]
    if TRAIN_SEPARATOR_TOKEN not in body:
        raise ValueError("train.txt line is missing <|sep|>.")

    profile, sequence_payload = body.split(TRAIN_SEPARATOR_TOKEN, 1)
    if not sequence_payload.startswith(PROTEIN_SEQUENCE_START_TOKEN):
        raise ValueError("train.txt line is missing a supported sequence prefix token.")
    sequence = sequence_payload[len(PROTEIN_SEQUENCE_START_TOKEN):]

    return MDCProfileSequenceRecord(
        profile=profile,
        sequence=sequence,
        sequence_type="protein",
    )


def create_mdc_profile_sequence_pretrain_dataloader(
    dataset: MDCProfileSequencePretrainDataset,
    *,
    batch_size: int = 4,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    fusion_config: ProfileSequenceFusionConfig | None = None,
    train_on_prompt: bool = False,
    include_separator_in_loss: bool = False,
) -> DataLoader[CausalLMBatch]:
    collator = MDCProfileSequenceBatchCollator(
        dataset.artifacts,
        fusion_config=fusion_config,
        train_on_prompt=train_on_prompt,
        include_separator_in_loss=include_separator_in_loss,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )


def build_profile_text_from_sequence_metadata(
    *,
    source_name: str,
    sequence_type: str,
    description: str = "",
    organism: str = "",
    metadata: Mapping[str, object] | None = None,
    config: MDCProfileCompilerConfig | None = None,
) -> str:
    resolved_config = config or MDCProfileCompilerConfig()
    resolved_sequence_type = _normalize_sequence_type(sequence_type)
    resolved_metadata = metadata or {}
    matched_labels, matched_keywords, label_source = _infer_profile_labels(
        description=description,
        sequence_type=resolved_sequence_type,
        metadata=resolved_metadata,
        config=resolved_config,
    )

    parts: list[str] = []
    if resolved_config.include_task:
        parts.append(f"task {_sanitize_profile_component(resolved_config.task_name)}")
    if resolved_config.include_labels:
        parts.append(f"labels {_sanitize_profile_component(', '.join(matched_labels))}")
    if resolved_config.include_label_source:
        parts.append(f"label source {_sanitize_profile_component(label_source)}")
    if resolved_config.include_matched_keywords and matched_keywords:
        parts.append(f"keywords {_sanitize_profile_component(', '.join(matched_keywords))}")
    if resolved_config.include_description and description:
        parts.append(f"description {_sanitize_profile_component(description)}")
    if resolved_config.include_organism and organism:
        parts.append(f"organism {_sanitize_profile_component(organism)}")
    if resolved_config.include_sequence_type:
        parts.append(f"sequence type {resolved_sequence_type}")
    if resolved_config.include_source_name and source_name:
        parts.append(f"source {_sanitize_profile_component(source_name)}")

    for field_name in resolved_config.metadata_fields:
        raw_value = resolved_metadata.get(field_name)
        cleaned_value = _sanitize_profile_component(str(raw_value)) if raw_value is not None else ""
        if cleaned_value:
            parts.append(f"{field_name} {cleaned_value}")

    if not parts:
        parts.append(f"sequence type {resolved_sequence_type}")
        if source_name:
            parts.append(f"source {_sanitize_profile_component(source_name)}")

    return "; ".join(parts)


def load_mdc_profile_sequence_records_from_instruction_jsonl(
    instruction_jsonl_path: Path | str,
    *,
    default_sequence_type: str = "protein",
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
) -> tuple[MDCProfileSequenceRecord, ...]:
    resolved_path = Path(instruction_jsonl_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"instruction.jsonl was not found: {resolved_path}")

    records: list[MDCProfileSequenceRecord] = []
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue

            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in instruction.jsonl line {line_number}.") from exc

            if not isinstance(payload, Mapping):
                raise ValueError(f"instruction.jsonl line {line_number} must contain a JSON object.")

            instruction = _sanitize_profile_component(str(payload.get(instruction_field) or ""))
            input_text = _sanitize_profile_component(str(payload.get(input_field) or ""))
            sequence = _compact_sequence_text(str(payload.get(output_field) or ""))
            if not instruction:
                raise ValueError(f"instruction.jsonl line {line_number} is missing '{instruction_field}'.")
            if not sequence:
                raise ValueError(f"instruction.jsonl line {line_number} is missing '{output_field}'.")

            metadata_payload = payload.get("metadata")
            metadata: dict[str, object] = (
                {str(key): value for key, value in metadata_payload.items()}
                if isinstance(metadata_payload, Mapping)
                else {}
            )
            for key, value in payload.items():
                if key in {instruction_field, input_field, output_field, "metadata"}:
                    continue
                metadata.setdefault(str(key), value)

            sequence_type = _normalize_sequence_type(
                str(
                    payload.get("sequence_type")
                    or metadata.get("sequence_type")
                    or default_sequence_type
                )
            )
            profile = instruction if not input_text else f"{instruction}; input {input_text}"
            records.append(
                MDCProfileSequenceRecord(
                    profile=profile,
                    sequence=sequence,
                    sequence_type=sequence_type,
                    metadata=metadata,
                )
            )

    if not records:
        raise ValueError(f"instruction.jsonl does not contain any non-empty records: {resolved_path}")
    return tuple(records)


def save_mdc_profile_sequence_pretrain_from_instruction_jsonl(
    instruction_jsonl_path: Path | str,
    output_dir: Path | str,
    *,
    default_sequence_type: str = "protein",
    kmer_size: int = 3,
    profile_vocab_size: int = 256,
) -> MDCProfileSequenceTextArtifact:
    records = load_mdc_profile_sequence_records_from_instruction_jsonl(
        instruction_jsonl_path,
        default_sequence_type=default_sequence_type,
    )
    sequence_types = {record.sequence_type for record in records}
    if len(sequence_types) != 1:
        raise ValueError(
            "The current MDC profile-aware text format requires a single sequence_type across instruction records."
        )

    return save_mdc_profile_sequence_pretrain_artifacts(
        records,
        output_dir,
        sequence_type=next(iter(sequence_types)),
        kmer_size=kmer_size,
        profile_vocab_size=profile_vocab_size,
    )


def load_mdc_profile_sequence_records_from_session_artifact(
    artifact: PreparationSessionArtifact,
    *,
    default_sequence_type: str = "protein",
    profile_config: MDCProfileCompilerConfig | None = None,
) -> tuple[MDCProfileSequenceRecord, ...]:
    session_dir = Path(artifact.session_location)
    raw_index_path = session_dir / "raw_index.json"
    accessions_path = session_dir / "accessions.txt"

    if not raw_index_path.exists():
        raise ValueError(f"raw_index.json was not found for session '{artifact.dataset_name}'.")

    payload = json.loads(raw_index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"raw_index.json for session '{artifact.dataset_name}' must contain an object mapping.")

    raw_index = {str(accession): entry for accession, entry in payload.items() if isinstance(entry, dict)}
    ordered_accessions = _ordered_session_accessions(raw_index, accessions_path)

    records: list[MDCProfileSequenceRecord] = []
    for accession in ordered_accessions:
        entry = raw_index.get(accession)
        if entry is None or not bool(entry.get("included_in_current_dataset")):
            continue

        metadata = entry.get("metadata", {})
        resolved_metadata = metadata if isinstance(metadata, Mapping) else {}
        sequence_type = str(resolved_metadata.get("sequence_type", default_sequence_type))
        normalized_sequence_type = _normalize_sequence_type(sequence_type)
        normalized_sequence = str(entry.get("normalized_sequence") or entry.get("sequence") or "").strip()
        if not normalized_sequence:
            continue

        source_name = str(entry.get("source_name") or artifact.source_name)
        description = str(entry.get("description") or "")
        organism = str(entry.get("organism") or "")
        profile = build_profile_text_from_sequence_metadata(
            source_name=source_name,
            sequence_type=normalized_sequence_type,
            description=description,
            organism=organism,
            metadata=resolved_metadata,
            config=profile_config,
        )
        matched_labels, matched_keywords, label_source = _infer_profile_labels(
            description=description,
            sequence_type=normalized_sequence_type,
            metadata=resolved_metadata,
            config=profile_config or MDCProfileCompilerConfig(),
        )

        record_metadata = {
            "accession": accession,
            "source_name": source_name,
            "description": description,
            "organism": organism,
            "derived_labels": list(matched_labels),
            "derived_keywords": list(matched_keywords),
            "derived_label_source": label_source,
            **{str(key): value for key, value in resolved_metadata.items()},
        }
        records.append(
            MDCProfileSequenceRecord(
                profile=profile,
                sequence=normalized_sequence,
                sequence_type=normalized_sequence_type,
                metadata=record_metadata,
            )
        )

    return tuple(records)


def save_mdc_profile_sequence_pretrain_from_preparation_sessions(
    artifacts: Sequence[PreparationSessionArtifact],
    output_dir: Path | str,
    *,
    default_sequence_type: str = "protein",
    kmer_size: int = 3,
    profile_vocab_size: int = 256,
    profile_config: MDCProfileCompilerConfig | None = None,
) -> MDCProfileSequenceTextArtifact:
    if not artifacts:
        raise ValueError("artifacts must not be empty.")

    records: list[MDCProfileSequenceRecord] = []
    for artifact in artifacts:
        records.extend(
            load_mdc_profile_sequence_records_from_session_artifact(
                artifact,
                default_sequence_type=default_sequence_type,
                profile_config=profile_config,
            )
        )

    if not records:
        raise ValueError("No included records were found in the provided preparation sessions.")

    sequence_types = {record.sequence_type for record in records}
    if len(sequence_types) != 1:
        raise ValueError(
            "The current MDC profile-aware text format requires a single sequence_type across all sessions."
        )

    return save_mdc_profile_sequence_pretrain_artifacts(
        records,
        output_dir,
        sequence_type=next(iter(sequence_types)),
        kmer_size=kmer_size,
        profile_vocab_size=profile_vocab_size,
    )


def _coerce_records(
    records: Sequence[MDCProfileSequenceRecord | ProfileSequencePair],
    *,
    default_sequence_type: str,
) -> tuple[MDCProfileSequenceRecord, ...]:
    resolved_records: list[MDCProfileSequenceRecord] = []
    normalized_default_sequence_type = _normalize_sequence_type(default_sequence_type)
    for record in records:
        if isinstance(record, MDCProfileSequenceRecord):
            resolved_records.append(record)
            continue

        metadata = dict(record.metadata)
        resolved_records.append(
            MDCProfileSequenceRecord(
                profile=record.profile,
                sequence=record.sequence,
                sequence_type=str(metadata.get("sequence_type", normalized_default_sequence_type)),
                metadata=metadata,
            )
        )
    return tuple(resolved_records)


def _ordered_session_accessions(raw_index: Mapping[str, dict[str, object]], accessions_path: Path) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    if accessions_path.exists():
        for raw_line in accessions_path.read_text(encoding="utf-8").splitlines():
            accession = _canonical_accession(raw_line)
            if accession and accession in raw_index and accession not in seen:
                ordered.append(accession)
                seen.add(accession)

    for accession in raw_index:
        if accession not in seen:
            ordered.append(accession)
            seen.add(accession)
    return ordered


def _derive_generic_labels_from_metadata(
    *,
    sequence_type: str,
    metadata: Mapping[str, object],
) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()

    def add_label(label: str) -> None:
        cleaned_label = _sanitize_profile_component(label)
        if cleaned_label and cleaned_label not in seen:
            labels.append(cleaned_label)
            seen.add(cleaned_label)

    biomol = _normalize_signal_value(metadata.get("biomol"))
    moltype = _normalize_signal_value(metadata.get("moltype"))
    genome = _normalize_signal_value(metadata.get("genome"))
    completeness = _normalize_signal_value(metadata.get("completeness"))
    topology = _normalize_signal_value(metadata.get("topology"))

    structural_signal = " ".join(value for value in (biomol, moltype, genome, completeness, topology) if value)

    if "plasmid" in genome:
        add_label("plasmid sequence")
    if "chromosome" in genome:
        add_label("chromosome sequence")
    if "mitochond" in genome:
        add_label("mitochondrial sequence")
    if "chloroplast" in genome or "plastid" in genome:
        add_label("plastid sequence")
    if "segment" in genome:
        add_label("segmented genome sequence")
    if "genome" in genome or "genomic" in genome or "genomic" in biomol or "genomic" in moltype:
        add_label("genomic sequence")

    if "complete" in completeness:
        add_label("complete sequence")
    if "partial" in completeness:
        add_label("partial sequence")
    if "draft" in completeness:
        add_label("draft sequence")

    if "circular" in topology:
        add_label("circular molecule")
    if "linear" in topology:
        add_label("linear molecule")

    if "mrna" in structural_signal:
        add_label("mRNA sequence")
    if "rrna" in structural_signal:
        add_label("rRNA sequence")
    if "trna" in structural_signal:
        add_label("tRNA sequence")
    if "cdna" in structural_signal:
        add_label("cDNA sequence")
    if "genomic dna" in structural_signal:
        add_label("genomic DNA sequence")
    elif "dna" in structural_signal:
        add_label("DNA sequence")
    if "rna" in structural_signal and not any(label.endswith("RNA sequence") for label in labels):
        add_label("RNA sequence")

    add_label("DNA sequence" if _normalize_sequence_type(sequence_type) == "dna" else "RNA sequence" if _normalize_sequence_type(sequence_type) == "rna" else "protein sequence")

    return tuple(labels)


def _infer_profile_labels(
    *,
    description: str,
    sequence_type: str,
    metadata: Mapping[str, object],
    config: MDCProfileCompilerConfig,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    signal_parts: list[str] = []
    if description:
        signal_parts.append(description)

    for field_name in config.label_signal_fields:
        raw_value = metadata.get(field_name)
        if raw_value is None:
            continue
        cleaned_value = _sanitize_profile_component(str(raw_value))
        if cleaned_value:
            signal_parts.append(cleaned_value)

    signal_text = " ".join(signal_parts).strip()
    if not signal_text:
        generic_labels = _derive_generic_labels_from_metadata(
            sequence_type=sequence_type,
            metadata=metadata,
        )
        return generic_labels, (), config.generic_label_source_name

    matches = match_rules(signal_text, config.keyword_rules)
    labels: list[str] = []
    keywords: list[str] = []
    seen_labels: set[str] = set()
    seen_keywords: set[str] = set()

    for rule, matched_rule_keywords in matches:
        if rule.label not in seen_labels:
            labels.append(rule.label)
            seen_labels.add(rule.label)
        for keyword in matched_rule_keywords:
            if keyword not in seen_keywords:
                keywords.append(keyword)
                seen_keywords.add(keyword)

    if labels:
        return tuple(labels), tuple(keywords), "keyword rules"

    generic_labels = _derive_generic_labels_from_metadata(
        sequence_type=sequence_type,
        metadata=metadata,
    )
    return generic_labels, (), config.generic_label_source_name


def _build_tokenizer_map_payload(
    *,
    record_count: int,
    sequence_type: str,
    profile_tokenizer: ProfileBPETokenizer,
    sequence_tokenizer: KmerTokenizer,
    layout: FusedVocabularyLayout,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "format": MDC_PROFILE_SEQUENCE_FORMAT,
        "record_count": record_count,
        "sequence_type": sequence_type,
        "text_format": {
            "profile_start_token": PROFILE_START_TOKEN,
            "separator_token": TRAIN_SEPARATOR_TOKEN,
            "train_end_token": TRAIN_END_TOKEN,
            "sequence_start_token": _sequence_start_token(sequence_type),
        },
        "profile_tokenizer": {
            "type": "bpe",
            "tokenizer": json.loads(profile_tokenizer.to_json()),
        },
        "sequence_tokenizer": {
            "type": "kmer",
            "tokenizer": json.loads(sequence_tokenizer.to_json()),
        },
        "layout": {
            "profile_vocab_size": layout.profile_vocab_size,
            "sequence_vocab_size": layout.sequence_vocab_size,
            "pad_token_id": layout.pad_token_id,
            "bos_token_id": layout.bos_token_id,
            "eos_token_id": layout.eos_token_id,
            "sep_token_id": layout.sep_token_id,
        },
    }
    return payload


def _load_profile_tokenizer_from_payload(payload: Mapping[str, object]) -> ProfileBPETokenizer:
    tokenizer_payload = payload.get("tokenizer", payload)
    str_to_int = {str(token): int(token_id) for token, token_id in tokenizer_payload["str_to_int"].items()}
    int_to_str = {int(token_id): str(token) for token_id, token in tokenizer_payload["int_to_str"].items()}
    special_tokens = tuple(tokenizer_payload.get("special_tokens", ("<|pad|>", "<|bos|>", "<|eos|>")))
    base_charset = tuple(tokenizer_payload.get("base_charset", ()))

    bpe_merges: dict[tuple[int, int], int] = {}
    merge_ranks: dict[tuple[int, int], int] = {}
    for rank, merge in enumerate(tokenizer_payload.get("bpe_merges", [])):
        pair = tuple(int(item) for item in merge["pair"])
        new_id = int(merge["new_id"])
        bpe_merges[pair] = new_id
        merge_ranks[pair] = rank

    return ProfileBPETokenizer(
        str_to_int=str_to_int,
        int_to_str=int_to_str,
        special_tokens=special_tokens,
        bpe_merges=bpe_merges,
        merge_ranks=merge_ranks,
        base_charset=base_charset,
    )


def _load_kmer_tokenizer_from_payload(payload: Mapping[str, object]) -> KmerTokenizer:
    tokenizer_payload = payload.get("tokenizer", payload)
    str_to_int = {str(token): int(token_id) for token, token_id in tokenizer_payload["str_to_int"].items()}
    int_to_str = {int(token_id): str(token) for token_id, token in tokenizer_payload["int_to_str"].items()}
    return KmerTokenizer(
        kmer_size=int(tokenizer_payload["kmer_size"]),
        stride=int(tokenizer_payload.get("stride", 1)),
        sequence_type=str(tokenizer_payload.get("sequence_type", "protein")),
        str_to_int=str_to_int,
        int_to_str=int_to_str,
        special_tokens=tuple(tokenizer_payload.get("special_tokens", ("<|pad|>", "<|bos|>", "<|eos|>"))),
    )


def _load_layout_from_payload(payload: Mapping[str, object]) -> FusedVocabularyLayout:
    return FusedVocabularyLayout(
        profile_vocab_size=int(payload["profile_vocab_size"]),
        sequence_vocab_size=int(payload["sequence_vocab_size"]),
        pad_token_id=int(payload.get("pad_token_id", 0)),
        bos_token_id=int(payload.get("bos_token_id", 1)),
        eos_token_id=int(payload.get("eos_token_id", 2)),
        sep_token_id=int(payload.get("sep_token_id", 3)),
    )


def _pad_token_tensors(
    sequences: Sequence[torch.Tensor],
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_length = max(int(sequence.size(0)) for sequence in sequences)
    input_ids = torch.full((len(sequences), target_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), target_length), dtype=torch.long)

    for row_index, sequence in enumerate(sequences):
        sequence_length = int(sequence.size(0))
        input_ids[row_index, :sequence_length] = sequence
        attention_mask[row_index, :sequence_length] = 1

    return input_ids, attention_mask


def _normalize_sequence_type(sequence_type: str) -> str:
    normalized = sequence_type.strip().lower()
    if normalized not in SUPPORTED_SEQUENCE_TYPES:
        raise ValueError(f"Unsupported sequence_type '{sequence_type}'. Expected one of {SUPPORTED_SEQUENCE_TYPES}.")
    return normalized


def _canonical_accession(accession: str | None) -> str:
    if accession is None:
        return ""
    value = accession.strip()
    if not value:
        return ""
    prefix, separator, suffix = value.rpartition(".")
    if separator and suffix.isdigit() and prefix:
        return prefix
    return value


def _normalize_signal_value(value: object | None) -> str:
    if value is None:
        return ""
    return _sanitize_profile_component(str(value)).lower()


def _sequence_start_token(sequence_type: str) -> str:
    _normalize_sequence_type(sequence_type)
    return PROTEIN_SEQUENCE_START_TOKEN


def _validate_field_text(field_name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must not be empty.")
    for token in RESERVED_TRAIN_TOKENS:
        if token in value:
            raise ValueError(f"{field_name} must not contain reserved token '{token}'.")


def _sanitize_profile_component(value: str) -> str:
    cleaned = (
        value.replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace(";", ",")
        .strip()
    )
    for token in RESERVED_TRAIN_TOKENS:
        cleaned = cleaned.replace(token, " ")
    return " ".join(cleaned.split())


def _compact_sequence_text(value: str) -> str:
    return "".join(str(value).split()).upper()

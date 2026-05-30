from __future__ import annotations

from typing import Iterable

from libs.data.backends import DatasetManager
from libs.data.config import DataConfig
from libs.data.contracts import SequenceSource
from libs.data.entities import (
    DeleteResult,
    FetchRequest,
    ManagedDataset,
    MergeStrategy,
    PreparationSessionArtifact,
    SequenceRecord,
    TrainingDatasetArtifact,
)
from libs.data.training.normalization import SequenceNormalizationConfig, normalize_records
from libs.data.training.preparation import ResumableTrainingDataPreparer
from libs.data.training.tokenizer import DEFAULT_VOCAB_SIZES, SequenceTokenizer
from libs.data.utilities import SourceConfigurationError
from libs.data.utilities.exceptions import DataNotFoundError


class MicrobialDataHub:
    def __init__(self, sources: Iterable[SequenceSource], dataset_manager: DatasetManager, config: DataConfig) -> None:
        self._sources = {source.name: source for source in sources}
        self._dataset_manager = dataset_manager
        self._config = config
        self._preparer = ResumableTrainingDataPreparer(dataset_manager=dataset_manager, config=config)

    def collect(
        self,
        source_name: str,
        request: FetchRequest,
        sequence_type: str = "protein",
        merge_strategy: MergeStrategy = "replace",
        normalization: SequenceNormalizationConfig | None = None,
    ) -> TrainingDatasetArtifact:
        resolved_sequence_type = _normalize_sequence_type(sequence_type)
        effective_request = self._with_default_batch_size(request)
        source = self._resolve_source(source_name)

        fetched_records = source.fetch(effective_request)
        return self._normalize_save_and_build_artifact(
            source_name=source_name,
            request=effective_request,
            records=fetched_records,
            sequence_type=resolved_sequence_type,
            merge_strategy=merge_strategy,
            normalization=normalization,
        )

    def add_records(
        self,
        source_name: str,
        request: FetchRequest,
        records: list[SequenceRecord],
        sequence_type: str = "protein",
        merge_strategy: MergeStrategy = "replace",
        normalization: SequenceNormalizationConfig | None = None,
    ) -> TrainingDatasetArtifact:
        resolved_sequence_type = _normalize_sequence_type(sequence_type)
        effective_request = self._with_default_batch_size(request)
        return self._normalize_save_and_build_artifact(
            source_name=source_name,
            request=effective_request,
            records=records,
            sequence_type=resolved_sequence_type,
            merge_strategy=merge_strategy,
            normalization=normalization,
        )

    def list_datasets(self, source_name: str | None = None) -> list[ManagedDataset]:
        return self._dataset_manager.list_datasets(source_name=source_name)

    def delete_dataset(self, source_name: str, dataset_name: str, permanent: bool = False) -> DeleteResult:
        return self._dataset_manager.delete_dataset(
            source_name=source_name,
            dataset_name=dataset_name,
            permanent=permanent,
        )

    def prepare_training_data(
        self,
        source_name: str,
        request: FetchRequest,
        sequence_type: str = "protein",
        normalization: SequenceNormalizationConfig | None = None,
        vocab_size: int | None = None,
        restart: bool = False,
    ) -> PreparationSessionArtifact:
        resolved_sequence_type = _normalize_sequence_type(sequence_type)
        effective_request = self._with_default_batch_size(request)
        source = self._resolve_source(source_name)

        normalization_config = normalization or SequenceNormalizationConfig(sequence_type=resolved_sequence_type)
        return self._preparer.prepare(
            source_name=source_name,
            source=source,
            request=effective_request,
            sequence_type=resolved_sequence_type,
            normalization=normalization_config,
            vocab_size=vocab_size or DEFAULT_VOCAB_SIZES["protein"],
            restart=restart,
        )

    def _resolve_source(self, source_name: str) -> SequenceSource:
        source = self._sources.get(source_name)
        if source is None:
            available_sources = ", ".join(sorted(self._sources))
            raise SourceConfigurationError(
                f"Unknown source '{source_name}'. Available sources: {available_sources}"
            )
        return source

    def _normalize_save_and_build_artifact(
        self,
        source_name: str,
        request: FetchRequest,
        records: list[SequenceRecord],
        sequence_type: str,
        merge_strategy: MergeStrategy,
        normalization: SequenceNormalizationConfig | None,
    ) -> TrainingDatasetArtifact:
        normalization_config = normalization or SequenceNormalizationConfig(sequence_type=sequence_type)
        normalized_records, report = normalize_records(records, normalization_config)
        if not normalized_records:
            raise DataNotFoundError(
                f"All records were filtered out while preparing training data for '{request.dataset_name}'"
            )

        dataset_artifact = self._dataset_manager.save_records(
            source_name=source_name,
            request=request,
            records=normalized_records,
            merge_strategy=merge_strategy,
        )
        tokenizer = SequenceTokenizer.from_records(normalized_records)
        return TrainingDatasetArtifact(
            source_name=dataset_artifact.source_name,
            dataset_name=dataset_artifact.dataset_name,
            storage_mode=dataset_artifact.storage_mode,
            snapshot_id=dataset_artifact.snapshot_id,
            current_location=dataset_artifact.current_location,
            train_txt_path=dataset_artifact.file_locations["train.txt"],
            tokenizer_map_path=dataset_artifact.file_locations["tokenizer_map.json"],
            record_count=dataset_artifact.record_count,
            dropped_record_count=report.dropped_count,
            sequence_type=tokenizer.sequence_type,
            vocab_size=tokenizer.vocab_size,
            history_location=dataset_artifact.history_location,
        )

    def _with_default_batch_size(self, request: FetchRequest) -> FetchRequest:
        if request.batch_size is not None:
            return request
        return FetchRequest(
            dataset_name=request.dataset_name,
            query=request.query,
            accessions=request.accessions,
            limit=request.limit,
            batch_size=self._config.default_batch_size,
            extra_fields=request.extra_fields,
            include_suppressed=request.include_suppressed,
        )


def _normalize_sequence_type(sequence_type: str) -> str:
    normalized = sequence_type.strip().lower() or "protein"
    if normalized != "protein":
        raise ValueError("MicrobialDataHub now supports protein-only training workflows.")
    return normalized

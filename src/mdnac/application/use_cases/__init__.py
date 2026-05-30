"""Application use cases - orchestrated domain operations.

Each use case encapsulates a single business operation that was previously
embedded directly in MicrobialDataHub.
"""

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
from libs.data.utilities.exceptions import DataNotFoundError, SourceConfigurationError


def _normalize_sequence_type(sequence_type: str) -> str:
    normalized = sequence_type.strip().lower() or "protein"
    if normalized != "protein":
        raise ValueError("Only protein training workflows are supported.")
    return normalized


def _resolve_source(sources: dict[str, SequenceSource], source_name: str) -> SequenceSource:
    source = sources.get(source_name)
    if source is None:
        available = ", ".join(sorted(sources))
        raise SourceConfigurationError(
            f"Unknown source '{source_name}'. Available sources: {available}"
        )
    return source


def _build_training_artifact(
    dataset_artifact,
    report,
    tokenizer: SequenceTokenizer,
) -> TrainingDatasetArtifact:
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


class CollectDatasetUseCase:
    """Fetch sequences from a source, normalize, and persist as training data."""

    def __init__(self, sources: dict[str, SequenceSource], dataset_manager: DatasetManager) -> None:
        self._sources = sources
        self._manager = dataset_manager

    def execute(
        self,
        source_name: str,
        request: FetchRequest,
        sequence_type: str = "protein",
        merge_strategy: MergeStrategy = "replace",
        normalization: SequenceNormalizationConfig | None = None,
    ) -> TrainingDatasetArtifact:
        resolved_type = _normalize_sequence_type(sequence_type)
        source = _resolve_source(self._sources, source_name)
        records = source.fetch(request)

        config = normalization or SequenceNormalizationConfig(sequence_type=resolved_type)
        normalized, report = normalize_records(records, config)
        if not normalized:
            raise DataNotFoundError(
                f"All records filtered out for '{request.dataset_name}'"
            )

        artifact = self._manager.save_records(
            source_name=source_name,
            request=request,
            records=normalized,
            merge_strategy=merge_strategy,
        )
        tokenizer = SequenceTokenizer.from_records(normalized)
        return _build_training_artifact(artifact, report, tokenizer)


class AddRecordsUseCase:
    """Import pre-fetched records, normalize, and persist as training data."""

    def __init__(self, dataset_manager: DatasetManager) -> None:
        self._manager = dataset_manager

    def execute(
        self,
        source_name: str,
        request: FetchRequest,
        records: list[SequenceRecord],
        sequence_type: str = "protein",
        merge_strategy: MergeStrategy = "replace",
        normalization: SequenceNormalizationConfig | None = None,
    ) -> TrainingDatasetArtifact:
        resolved_type = _normalize_sequence_type(sequence_type)
        config = normalization or SequenceNormalizationConfig(sequence_type=resolved_type)
        normalized, report = normalize_records(records, config)
        if not normalized:
            raise DataNotFoundError(
                f"All records filtered out for '{request.dataset_name}'"
            )

        artifact = self._manager.save_records(
            source_name=source_name,
            request=request,
            records=normalized,
            merge_strategy=merge_strategy,
        )
        tokenizer = SequenceTokenizer.from_records(normalized)
        return _build_training_artifact(artifact, report, tokenizer)


class PrepareTrainingDataUseCase:
    """Resumable multi-batch training data preparation from a source."""

    def __init__(
        self,
        sources: dict[str, SequenceSource],
        dataset_manager: DatasetManager,
        config: DataConfig,
    ) -> None:
        self._sources = sources
        self._preparer = ResumableTrainingDataPreparer(dataset_manager=dataset_manager, config=config)

    def execute(
        self,
        source_name: str,
        request: FetchRequest,
        sequence_type: str = "protein",
        normalization: SequenceNormalizationConfig | None = None,
        vocab_size: int | None = None,
        restart: bool = False,
    ) -> PreparationSessionArtifact:
        resolved_type = _normalize_sequence_type(sequence_type)
        source = _resolve_source(self._sources, source_name)
        config = normalization or SequenceNormalizationConfig(sequence_type=resolved_type)
        return self._preparer.prepare(
            source_name=source_name,
            source=source,
            request=request,
            sequence_type=resolved_type,
            normalization=config,
            vocab_size=vocab_size or DEFAULT_VOCAB_SIZES["protein"],
            restart=restart,
        )


class ListDatasetsUseCase:
    """List managed datasets from the catalog."""

    def __init__(self, dataset_manager: DatasetManager) -> None:
        self._manager = dataset_manager

    def execute(self, source_name: str | None = None) -> list[ManagedDataset]:
        return self._manager.list_datasets(source_name=source_name)


class DeleteDatasetUseCase:
    """Delete or soft-delete a managed dataset."""

    def __init__(self, dataset_manager: DatasetManager) -> None:
        self._manager = dataset_manager

    def execute(self, source_name: str, dataset_name: str, permanent: bool = False) -> DeleteResult:
        return self._manager.delete_dataset(
            source_name=source_name,
            dataset_name=dataset_name,
            permanent=permanent,
        )


__all__ = [
    "AddRecordsUseCase",
    "CollectDatasetUseCase",
    "DeleteDatasetUseCase",
    "ListDatasetsUseCase",
    "PrepareTrainingDataUseCase",
]

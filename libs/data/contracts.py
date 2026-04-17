from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

from libs.data.entities import DatasetArtifact, DeleteResult, FetchRequest, ManagedDataset, MergeStrategy, SequenceRecord


class HttpTransport(ABC):
    @abstractmethod
    def get_text(
        self,
        url: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> str:
        """Return response text for a GET request."""


class SequenceSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable source identifier."""

    @abstractmethod
    def fetch(self, request: FetchRequest) -> list[SequenceRecord]:
        """Load sequence records for a request."""


class DatasetRepository(ABC):
    @abstractmethod
    def save_dataset(
        self,
        source_name: str,
        request: FetchRequest,
        records: Sequence[SequenceRecord],
        merge_strategy: MergeStrategy = "upsert",
    ) -> DatasetArtifact:
        """Persist dataset artifacts and return their locations."""

    @abstractmethod
    def save_prebuilt_dataset(
        self,
        source_name: str,
        dataset_name: str,
        train_text: str,
        tokenizer_map_text: str,
        record_count: int,
    ) -> DatasetArtifact:
        """Persist already-prepared training artifacts and return their locations."""

    @abstractmethod
    def list_datasets(self, source_name: str | None = None) -> list[ManagedDataset]:
        """List managed datasets."""

    @abstractmethod
    def delete_dataset(self, source_name: str, dataset_name: str, permanent: bool = False) -> DeleteResult:
        """Delete or soft-delete a managed dataset."""

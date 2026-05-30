from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from libs.data.backends.catalog import CatalogRepository
from libs.data.backends.object_store import S3ObjectStore
from libs.data.config import DataConfig
from libs.data.contracts import DatasetRepository
from libs.data.entities import DatasetArtifact, DeleteResult, FetchRequest, ManagedDataset, SequenceRecord
from libs.data.utilities.exceptions import DataNotFoundError, DatasetNotFoundError
from libs.data.utilities.storage import (
    build_prebuilt_dataset_bundle,
    build_dataset_bundle,
    utc_snapshot_id,
)


class MinioDatasetRepository(DatasetRepository):
    def __init__(self, config: DataConfig, s3_client=None) -> None:
        self._config = config
        self._store = S3ObjectStore(minio_config=config.minio, s3_client=s3_client)
        catalog_key = f"{config.minio.root_prefix}/catalog/datasets.csv"
        self._catalog = CatalogRepository(object_store=self._store, catalog_key=catalog_key)

    def save_dataset(
        self,
        source_name: str,
        request: FetchRequest,
        records: Sequence[SequenceRecord],
        merge_strategy: str = "upsert",
    ) -> DatasetArtifact:
        if not records:
            raise DataNotFoundError(f"No records available to save for source '{source_name}'")

        snapshot_id = utc_snapshot_id()
        current_prefix = self._current_prefix(source_name, request.dataset_name)
        history_location: str | None = None

        if self._store.prefix_exists(current_prefix):
            history_prefix = self._history_prefix(source_name, request.dataset_name, snapshot_id)
            self._store.copy_prefix(current_prefix, history_prefix)
            history_location = self._store.uri(history_prefix)

        bundle = build_dataset_bundle(
            source_name=source_name,
            request=request,
            records=records,
            storage_mode="minio",
            snapshot_id=snapshot_id,
            merge_strategy=merge_strategy,
        )

        file_locations: dict[str, str] = {}
        for file_name, content in bundle.items():
            key = f"{current_prefix}/{file_name}"
            self._store.put_text(key, content)
            file_locations[file_name] = self._store.uri(key)

        artifact = DatasetArtifact(
            source_name=source_name,
            dataset_name=request.dataset_name,
            storage_mode="minio",
            snapshot_id=snapshot_id,
            current_location=self._store.uri(current_prefix),
            file_locations=file_locations,
            record_count=len(records),
            history_location=history_location,
        )
        self._catalog.upsert(artifact)
        return artifact

    def save_prebuilt_dataset(
        self,
        source_name: str,
        dataset_name: str,
        train_text: str,
        tokenizer_map_text: str,
        record_count: int,
    ) -> DatasetArtifact:
        if not train_text.strip():
            raise DataNotFoundError(f"No training text available to save for source '{source_name}'")

        snapshot_id = utc_snapshot_id()
        current_prefix = self._current_prefix(source_name, dataset_name)
        history_location: str | None = None

        if self._store.prefix_exists(current_prefix):
            history_prefix = self._history_prefix(source_name, dataset_name, snapshot_id)
            self._store.copy_prefix(current_prefix, history_prefix)
            history_location = self._store.uri(history_prefix)

        bundle = build_prebuilt_dataset_bundle(train_text=train_text, tokenizer_map_text=tokenizer_map_text)
        file_locations: dict[str, str] = {}
        for file_name, content in bundle.items():
            key = f"{current_prefix}/{file_name}"
            self._store.put_text(key, content)
            file_locations[file_name] = self._store.uri(key)

        artifact = DatasetArtifact(
            source_name=source_name,
            dataset_name=dataset_name,
            storage_mode="minio",
            snapshot_id=snapshot_id,
            current_location=self._store.uri(current_prefix),
            file_locations=file_locations,
            record_count=record_count,
            history_location=history_location,
        )
        self._catalog.upsert(artifact)
        return artifact

    def list_datasets(self, source_name: str | None = None) -> list[ManagedDataset]:
        return self._catalog.list_datasets(source_name=source_name)

    def delete_dataset(self, source_name: str, dataset_name: str, permanent: bool = False) -> DeleteResult:
        dataset_prefix = self._dataset_prefix(source_name, dataset_name)
        if not self._store.prefix_exists(dataset_prefix):
            raise DatasetNotFoundError(f"Dataset '{source_name}/{dataset_name}' does not exist")

        location = self._store.uri(dataset_prefix)
        if permanent:
            self._store.delete_prefix(dataset_prefix)
        else:
            deleted_at = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            trash_prefix = self._trash_prefix(source_name, dataset_name, deleted_at)
            self._store.copy_prefix(dataset_prefix, trash_prefix)
            self._store.delete_prefix(dataset_prefix)
            location = self._store.uri(trash_prefix)

        self._catalog.remove(source_name, dataset_name)
        return DeleteResult(
            source_name=source_name,
            dataset_name=dataset_name,
            storage_mode="minio",
            deleted=True,
            location=location,
            permanent=permanent,
        )

    def _dataset_prefix(self, source_name: str, dataset_name: str) -> str:
        return f"{self._config.minio.root_prefix}/datasets/{source_name}/{dataset_name}"

    def _current_prefix(self, source_name: str, dataset_name: str) -> str:
        return f"{self._dataset_prefix(source_name, dataset_name)}/current"

    def _history_prefix(self, source_name: str, dataset_name: str, snapshot_id: str) -> str:
        return f"{self._dataset_prefix(source_name, dataset_name)}/history/{snapshot_id}"

    def _trash_prefix(self, source_name: str, dataset_name: str, deleted_at: str) -> str:
        return f"{self._config.minio.root_prefix}/trash/{source_name}/{dataset_name}/{deleted_at}"

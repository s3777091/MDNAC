from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from libs.data.backends.catalog import CatalogRepository
from libs.data.backends.object_store import LocalObjectStore
from libs.data.config import DataConfig
from libs.data.contracts import DatasetRepository
from libs.data.entities import DatasetArtifact, DeleteResult, FetchRequest, ManagedDataset, SequenceRecord
from libs.data.utilities.exceptions import DataNotFoundError, DatasetNotFoundError
from libs.data.utilities.storage import (
    build_prebuilt_dataset_bundle,
    build_dataset_bundle,
    utc_snapshot_id,
)


class LocalDatasetRepository(DatasetRepository):
    def __init__(self, config: DataConfig) -> None:
        self._config = config
        self._ensure_roots()
        self._store = LocalObjectStore(root=config.datasets_root)
        # Catalog lives under catalog_root; use a separate store rooted there
        self._catalog_store = LocalObjectStore(root=config.catalog_root)
        self._catalog = CatalogRepository(object_store=self._catalog_store, catalog_key="datasets.csv")

    def save_dataset(
        self,
        source_name: str,
        request: FetchRequest,
        records: Sequence[SequenceRecord],
        merge_strategy: str = "upsert",
    ) -> DatasetArtifact:
        if not records:
            raise DataNotFoundError(f"No records available to save for source '{source_name}'")

        self._ensure_roots()

        snapshot_id = utc_snapshot_id()
        dataset_root = self._dataset_root(source_name, request.dataset_name)
        current_dir = dataset_root / "current"
        history_location: str | None = None

        if current_dir.exists():
            history_dir = dataset_root / "history" / snapshot_id
            history_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(current_dir, history_dir)
            history_location = str(history_dir)

        current_dir.mkdir(parents=True, exist_ok=True)
        bundle = build_dataset_bundle(
            source_name=source_name,
            request=request,
            records=records,
            storage_mode="local",
            snapshot_id=snapshot_id,
            merge_strategy=merge_strategy,
        )

        file_locations: dict[str, str] = {}
        for file_name, content in bundle.items():
            file_path = current_dir / file_name
            file_path.write_text(content, encoding="utf-8")
            file_locations[file_name] = str(file_path)

        artifact = DatasetArtifact(
            source_name=source_name,
            dataset_name=request.dataset_name,
            storage_mode="local",
            snapshot_id=snapshot_id,
            current_location=str(current_dir),
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

        self._ensure_roots()
        snapshot_id = utc_snapshot_id()
        dataset_root = self._dataset_root(source_name, dataset_name)
        current_dir = dataset_root / "current"
        history_location: str | None = None

        if current_dir.exists():
            history_dir = dataset_root / "history" / snapshot_id
            history_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(current_dir, history_dir)
            history_location = str(history_dir)

        current_dir.mkdir(parents=True, exist_ok=True)
        bundle = build_prebuilt_dataset_bundle(train_text=train_text, tokenizer_map_text=tokenizer_map_text)

        file_locations: dict[str, str] = {}
        for file_name, content in bundle.items():
            file_path = current_dir / file_name
            file_path.write_text(content, encoding="utf-8")
            file_locations[file_name] = str(file_path)

        artifact = DatasetArtifact(
            source_name=source_name,
            dataset_name=dataset_name,
            storage_mode="local",
            snapshot_id=snapshot_id,
            current_location=str(current_dir),
            file_locations=file_locations,
            record_count=record_count,
            history_location=history_location,
        )
        self._catalog.upsert(artifact)
        return artifact

    def list_datasets(self, source_name: str | None = None) -> list[ManagedDataset]:
        return self._catalog.list_datasets(source_name=source_name)

    def delete_dataset(self, source_name: str, dataset_name: str, permanent: bool = False) -> DeleteResult:
        dataset_root = self._dataset_root(source_name, dataset_name)
        if not dataset_root.exists():
            raise DatasetNotFoundError(f"Dataset '{source_name}/{dataset_name}' does not exist")

        if permanent:
            shutil.rmtree(dataset_root)
            location = str(dataset_root)
        else:
            deleted_at = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            trash_target = self._config.trash_root / source_name / dataset_name / deleted_at
            trash_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dataset_root), str(trash_target))
            location = str(trash_target)

        self._catalog.remove(source_name, dataset_name)
        return DeleteResult(
            source_name=source_name,
            dataset_name=dataset_name,
            storage_mode="local",
            deleted=True,
            location=location,
            permanent=permanent,
        )

    def _ensure_roots(self) -> None:
        self._config.models_root.mkdir(parents=True, exist_ok=True)
        self._config.catalog_root.mkdir(parents=True, exist_ok=True)
        self._config.datasets_root.mkdir(parents=True, exist_ok=True)
        self._config.trash_root.mkdir(parents=True, exist_ok=True)

    def _dataset_root(self, source_name: str, dataset_name: str) -> Path:
        return self._config.datasets_root / source_name / dataset_name

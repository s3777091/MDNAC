from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from libs.data.config import DataConfig
from libs.data.contracts import DatasetRepository
from libs.data.entities import DatasetArtifact, DeleteResult, FetchRequest, ManagedDataset, SequenceRecord
from libs.data.utilities.exceptions import DataNotFoundError, DatasetNotFoundError
from libs.data.utilities.storage import (
    build_prebuilt_dataset_bundle,
    build_dataset_bundle,
    managed_dataset_from_row,
    parse_catalog_csv,
    render_catalog_csv,
    utc_snapshot_id,
)


class LocalDatasetRepository(DatasetRepository):
    def __init__(self, config: DataConfig) -> None:
        self._config = config
        self._ensure_roots()

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
        self._upsert_catalog_row(artifact)
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
        self._upsert_catalog_row(artifact)
        return artifact

    def list_datasets(self, source_name: str | None = None) -> list[ManagedDataset]:
        rows = self._read_catalog_rows()
        if source_name is not None:
            rows = [row for row in rows if row.get("source_name") == source_name]
        return [managed_dataset_from_row(row) for row in rows]

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

        self._remove_catalog_row(source_name, dataset_name)
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

    def _catalog_path(self) -> Path:
        return self._config.catalog_root / "datasets.csv"

    def _read_catalog_rows(self) -> list[dict[str, str]]:
        catalog_path = self._catalog_path()
        if not catalog_path.exists():
            return []
        return parse_catalog_csv(catalog_path.read_text(encoding="utf-8"))

    def _write_catalog_rows(self, rows: list[dict[str, str]]) -> None:
        self._catalog_path().write_text(render_catalog_csv(rows), encoding="utf-8")

    def _upsert_catalog_row(self, artifact: DatasetArtifact) -> None:
        rows = [
            row
            for row in self._read_catalog_rows()
            if not (row.get("source_name") == artifact.source_name and row.get("dataset_name") == artifact.dataset_name)
        ]
        rows.append(
            {
                "source_name": artifact.source_name,
                "dataset_name": artifact.dataset_name,
                "storage_mode": artifact.storage_mode,
                "record_count": str(artifact.record_count),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "snapshot_id": artifact.snapshot_id,
                "current_location": artifact.current_location,
            }
        )
        rows.sort(key=lambda row: (row.get("source_name", ""), row.get("dataset_name", "")))
        self._write_catalog_rows(rows)

    def _remove_catalog_row(self, source_name: str, dataset_name: str) -> None:
        rows = [
            row
            for row in self._read_catalog_rows()
            if not (row.get("source_name") == source_name and row.get("dataset_name") == dataset_name)
        ]
        self._write_catalog_rows(rows)

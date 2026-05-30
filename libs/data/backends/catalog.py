from __future__ import annotations

from datetime import datetime, timezone

from libs.data.backends.object_store import ObjectStore
from libs.data.entities import DatasetArtifact, ManagedDataset
from libs.data.utilities.storage import managed_dataset_from_row, parse_catalog_csv, render_catalog_csv


class CatalogRepository:
    """Shared catalog CRUD logic for both local and S3 backends."""

    def __init__(self, object_store: ObjectStore, catalog_key: str) -> None:
        self._store = object_store
        self._catalog_key = catalog_key

    def read_rows(self) -> list[dict[str, str]]:
        catalog_text = self._store.get_text(self._catalog_key)
        if not catalog_text:
            return []
        return parse_catalog_csv(catalog_text)

    def write_rows(self, rows: list[dict[str, str]]) -> None:
        self._store.put_text(self._catalog_key, render_catalog_csv(rows))

    def upsert(self, artifact: DatasetArtifact) -> None:
        rows = [
            row
            for row in self.read_rows()
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
        self.write_rows(rows)

    def remove(self, source_name: str, dataset_name: str) -> None:
        rows = [
            row
            for row in self.read_rows()
            if not (row.get("source_name") == source_name and row.get("dataset_name") == dataset_name)
        ]
        self.write_rows(rows)

    def list_datasets(self, source_name: str | None = None) -> list[ManagedDataset]:
        rows = self.read_rows()
        if source_name is not None:
            rows = [row for row in rows if row.get("source_name") == source_name]
        return [managed_dataset_from_row(row) for row in rows]

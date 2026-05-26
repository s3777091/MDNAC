from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import boto3

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


class MinioDatasetRepository(DatasetRepository):
    def __init__(self, config: DataConfig, s3_client=None) -> None:
        self._config = config
        self._bucket_name = config.minio.bucket_name
        self._client = s3_client or self._build_client()
        self._ensure_bucket()

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

        if self._prefix_exists(current_prefix):
            history_prefix = self._history_prefix(source_name, request.dataset_name, snapshot_id)
            self._copy_prefix(current_prefix, history_prefix)
            history_location = self._uri(history_prefix)

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
            self._put_text(key, content)
            file_locations[file_name] = self._uri(key)

        artifact = DatasetArtifact(
            source_name=source_name,
            dataset_name=request.dataset_name,
            storage_mode="minio",
            snapshot_id=snapshot_id,
            current_location=self._uri(current_prefix),
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

        snapshot_id = utc_snapshot_id()
        current_prefix = self._current_prefix(source_name, dataset_name)
        history_location: str | None = None

        if self._prefix_exists(current_prefix):
            history_prefix = self._history_prefix(source_name, dataset_name, snapshot_id)
            self._copy_prefix(current_prefix, history_prefix)
            history_location = self._uri(history_prefix)

        bundle = build_prebuilt_dataset_bundle(train_text=train_text, tokenizer_map_text=tokenizer_map_text)
        file_locations: dict[str, str] = {}
        for file_name, content in bundle.items():
            key = f"{current_prefix}/{file_name}"
            self._put_text(key, content)
            file_locations[file_name] = self._uri(key)

        artifact = DatasetArtifact(
            source_name=source_name,
            dataset_name=dataset_name,
            storage_mode="minio",
            snapshot_id=snapshot_id,
            current_location=self._uri(current_prefix),
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
        dataset_prefix = self._dataset_prefix(source_name, dataset_name)
        if not self._prefix_exists(dataset_prefix):
            raise DatasetNotFoundError(f"Dataset '{source_name}/{dataset_name}' does not exist")

        location = self._uri(dataset_prefix)
        if permanent:
            self._delete_prefix(dataset_prefix)
        else:
            deleted_at = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            trash_prefix = self._trash_prefix(source_name, dataset_name, deleted_at)
            self._copy_prefix(dataset_prefix, trash_prefix)
            self._delete_prefix(dataset_prefix)
            location = self._uri(trash_prefix)

        self._remove_catalog_row(source_name, dataset_name)
        return DeleteResult(
            source_name=source_name,
            dataset_name=dataset_name,
            storage_mode="minio",
            deleted=True,
            location=location,
            permanent=permanent,
        )

    def _build_client(self):
        return boto3.client(
            "s3",
            endpoint_url=self._config.minio.normalized_endpoint_url,
            aws_access_key_id=self._config.minio.access_key,
            aws_secret_access_key=self._config.minio.secret_key,
            region_name=self._config.minio.region_name,
            use_ssl=self._config.minio.secure,
        )

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket_name)
        except Exception:
            self._client.create_bucket(Bucket=self._bucket_name)

    def _catalog_key(self) -> str:
        return f"{self._config.minio.root_prefix}/catalog/datasets.csv"

    def _dataset_prefix(self, source_name: str, dataset_name: str) -> str:
        return f"{self._config.minio.root_prefix}/datasets/{source_name}/{dataset_name}"

    def _current_prefix(self, source_name: str, dataset_name: str) -> str:
        return f"{self._dataset_prefix(source_name, dataset_name)}/current"

    def _history_prefix(self, source_name: str, dataset_name: str, snapshot_id: str) -> str:
        return f"{self._dataset_prefix(source_name, dataset_name)}/history/{snapshot_id}"

    def _trash_prefix(self, source_name: str, dataset_name: str, deleted_at: str) -> str:
        return f"{self._config.minio.root_prefix}/trash/{source_name}/{dataset_name}/{deleted_at}"

    def _uri(self, key: str) -> str:
        return f"s3://{self._bucket_name}/{key}"

    def _put_text(self, key: str, content: str) -> None:
        self._client.put_object(Bucket=self._bucket_name, Key=key, Body=content.encode("utf-8"))

    def _get_text(self, key: str) -> str:
        try:
            response = self._client.get_object(Bucket=self._bucket_name, Key=key)
        except Exception:
            return ""
        return response["Body"].read().decode("utf-8")

    def _list_keys(self, prefix: str) -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self._bucket_name, Prefix=prefix):
            for item in page.get("Contents", []):
                keys.append(item["Key"])
        return keys

    def _prefix_exists(self, prefix: str) -> bool:
        response = self._client.list_objects_v2(Bucket=self._bucket_name, Prefix=prefix, MaxKeys=1)
        return response.get("KeyCount", 0) > 0

    def _copy_prefix(self, source_prefix: str, target_prefix: str) -> None:
        for key in self._list_keys(source_prefix):
            relative_key = key[len(source_prefix) :].lstrip("/")
            target_key = f"{target_prefix}/{relative_key}" if relative_key else target_prefix
            self._client.copy_object(
                Bucket=self._bucket_name,
                CopySource={"Bucket": self._bucket_name, "Key": key},
                Key=target_key,
            )

    def _delete_prefix(self, prefix: str) -> None:
        keys = self._list_keys(prefix)
        if not keys:
            return
        for start_index in range(0, len(keys), 1000):
            chunk = keys[start_index : start_index + 1000]
            self._client.delete_objects(
                Bucket=self._bucket_name,
                Delete={"Objects": [{"Key": key} for key in chunk]},
            )

    def _read_catalog_rows(self) -> list[dict[str, str]]:
        catalog_text = self._get_text(self._catalog_key())
        if not catalog_text:
            return []
        return parse_catalog_csv(catalog_text)

    def _write_catalog_rows(self, rows: list[dict[str, str]]) -> None:
        self._put_text(self._catalog_key(), render_catalog_csv(rows))

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

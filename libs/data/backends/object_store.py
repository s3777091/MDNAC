from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from libs.data.config import MinioConfig


class ObjectStore(ABC):
    """Port for object storage operations (local filesystem or S3-compatible)."""

    @abstractmethod
    def put_text(self, key: str, content: str) -> None:
        """Write UTF-8 text to the given key."""

    @abstractmethod
    def get_text(self, key: str) -> str | None:
        """Read UTF-8 text from the given key. Returns None if key does not exist."""

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]:
        """List all keys under the given prefix."""

    @abstractmethod
    def prefix_exists(self, prefix: str) -> bool:
        """Return True if at least one object exists under the given prefix."""

    @abstractmethod
    def copy_prefix(self, source_prefix: str, target_prefix: str) -> None:
        """Copy all objects under source_prefix to target_prefix."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None:
        """Delete all objects under the given prefix."""

    @abstractmethod
    def uri(self, key: str) -> str:
        """Return a URI string for the given key."""


class LocalObjectStore(ObjectStore):
    """Object store backed by the local filesystem."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def put_text(self, key: str, content: str) -> None:
        file_path = self._root / key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    def get_text(self, key: str) -> str | None:
        file_path = self._root / key
        if not file_path.exists():
            return None
        return file_path.read_text(encoding="utf-8")

    def list_keys(self, prefix: str) -> list[str]:
        prefix_path = self._root / prefix
        if not prefix_path.exists():
            return []
        keys: list[str] = []
        for item in prefix_path.rglob("*"):
            if item.is_file():
                keys.append(str(item.relative_to(self._root)).replace("\\", "/"))
        return keys

    def prefix_exists(self, prefix: str) -> bool:
        prefix_path = self._root / prefix
        if not prefix_path.exists():
            return False
        if prefix_path.is_file():
            return True
        return any(prefix_path.iterdir())

    def copy_prefix(self, source_prefix: str, target_prefix: str) -> None:
        source_path = self._root / source_prefix
        target_path = self._root / target_prefix
        if source_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_path, target_path)

    def delete_prefix(self, prefix: str) -> None:
        prefix_path = self._root / prefix
        if prefix_path.exists():
            shutil.rmtree(prefix_path)

    def uri(self, key: str) -> str:
        return str(self._root / key)


class S3ObjectStore(ObjectStore):
    """Object store backed by S3-compatible storage (MinIO, AWS S3)."""

    def __init__(self, minio_config: MinioConfig, s3_client=None) -> None:
        self._config = minio_config
        self._bucket_name = minio_config.bucket_name
        self._client = s3_client or self._build_client()
        self._ensure_bucket()

    def put_text(self, key: str, content: str) -> None:
        self._client.put_object(Bucket=self._bucket_name, Key=key, Body=content.encode("utf-8"))

    def get_text(self, key: str) -> str | None:
        try:
            response = self._client.get_object(Bucket=self._bucket_name, Key=key)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                return None
            raise
        return response["Body"].read().decode("utf-8")

    def list_keys(self, prefix: str) -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self._bucket_name, Prefix=prefix):
            for item in page.get("Contents", []):
                keys.append(item["Key"])
        return keys

    def prefix_exists(self, prefix: str) -> bool:
        response = self._client.list_objects_v2(Bucket=self._bucket_name, Prefix=prefix, MaxKeys=1)
        return response.get("KeyCount", 0) > 0

    def copy_prefix(self, source_prefix: str, target_prefix: str) -> None:
        for key in self.list_keys(source_prefix):
            relative_key = key[len(source_prefix):].lstrip("/")
            target_key = f"{target_prefix}/{relative_key}" if relative_key else target_prefix
            self._client.copy_object(
                Bucket=self._bucket_name,
                CopySource={"Bucket": self._bucket_name, "Key": key},
                Key=target_key,
            )

    def delete_prefix(self, prefix: str) -> None:
        keys = self.list_keys(prefix)
        if not keys:
            return
        for start_index in range(0, len(keys), 1000):
            chunk = keys[start_index: start_index + 1000]
            self._client.delete_objects(
                Bucket=self._bucket_name,
                Delete={"Objects": [{"Key": key} for key in chunk]},
            )

    def uri(self, key: str) -> str:
        return f"s3://{self._bucket_name}/{key}"

    def _build_client(self):
        return boto3.client(
            "s3",
            endpoint_url=self._config.normalized_endpoint_url,
            aws_access_key_id=self._config.access_key,
            aws_secret_access_key=self._config.secret_key,
            region_name=self._config.region_name,
            use_ssl=self._config.secure,
        )

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket_name)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchBucket"):
                self._client.create_bucket(Bucket=self._bucket_name)
            else:
                raise

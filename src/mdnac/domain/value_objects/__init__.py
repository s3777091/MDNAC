"""Domain value objects - typed literals and small immutable types."""

from typing import Literal

MergeStrategy = Literal["replace", "skip_duplicates", "upsert"]
StorageMode = Literal["local", "minio"]
SequenceType = Literal["protein"]

__all__ = ["MergeStrategy", "SequenceType", "StorageMode"]

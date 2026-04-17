from __future__ import annotations

from libs.data.backends.local import LocalDatasetRepository
from libs.data.backends.minio import MinioDatasetRepository
from libs.data.config import DataConfig
from libs.data.contracts import DatasetRepository


def build_dataset_repository(config: DataConfig) -> DatasetRepository:
    if config.storage_mode == "minio":
        return MinioDatasetRepository(config=config)
    return LocalDatasetRepository(config=config)

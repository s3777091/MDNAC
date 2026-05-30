"""Domain exceptions hierarchy."""


class DataIngestionError(Exception):
    """Base exception for all data ingestion errors."""


class DataNotFoundError(DataIngestionError):
    """Raised when no records are returned from a source or all are filtered out."""


class DatasetNotFoundError(DataIngestionError):
    """Raised when a referenced dataset does not exist in storage."""


class SourceConfigurationError(DataIngestionError):
    """Raised when source configuration is invalid or incomplete."""


class StorageOperationError(DataIngestionError):
    """Raised when a storage backend operation fails."""


__all__ = [
    "DataIngestionError",
    "DataNotFoundError",
    "DatasetNotFoundError",
    "SourceConfigurationError",
    "StorageOperationError",
]

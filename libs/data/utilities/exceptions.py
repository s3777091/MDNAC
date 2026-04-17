class DataIngestionError(Exception):
    """Base error for ingestion failures."""


class DataNotFoundError(DataIngestionError):
    """Raised when a source returns no usable records."""


class DatasetNotFoundError(DataIngestionError):
    """Raised when a managed dataset cannot be found in storage."""


class SourceConfigurationError(DataIngestionError):
    """Raised when a request is incompatible with a source."""


class StorageOperationError(DataIngestionError):
    """Raised when dataset storage backends fail."""


class OptionalDependencyError(DataIngestionError):
    """Raised when an optional package is required but missing."""

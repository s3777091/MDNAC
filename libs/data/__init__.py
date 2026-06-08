from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "DATA_CONFIG": "libs.data.config",
    "DataConfig": "libs.data.config",
    "MinioConfig": "libs.data.config",
    "DatasetArtifact": "libs.data.entities",
    "DeleteResult": "libs.data.entities",
    "FetchRequest": "libs.data.entities",
    "ManagedDataset": "libs.data.entities",
    "PreparationSessionArtifact": "libs.data.entities",
    "SequenceRecord": "libs.data.entities",
    "TrainingDatasetArtifact": "libs.data.entities",
    "MicrobialDataHub": "libs.data.hub",
    "KmerTokenizer": "libs.data.training",
    "NormalizationReport": "libs.data.training",
    "ProfileBPETokenizer": "libs.data.training",
    "RawLabelAuditReport": "libs.data.training",
    "RawLabelAuditRow": "libs.data.training",
    "RawLabelAuditSummary": "libs.data.training",
    "ProfileKeywordRule": "libs.data.training",
    "ProfileSequencePair": "libs.data.training",
    "RawDataPipeline": "libs.data.training",
    "RawTensorDatasetArtifact": "libs.data.training",
    "SequenceNormalizationConfig": "libs.data.training",
    "SequenceTokenizer": "libs.data.training",
    "audit_fasta_and_gff": "libs.data.training",
    "audit_genbank": "libs.data.training",
    "normalize_records": "libs.data.training",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value

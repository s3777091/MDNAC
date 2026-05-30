from __future__ import annotations

from libs.data.contracts import DatasetRepository
from libs.data.entities import DatasetArtifact, DeleteResult, FetchRequest, ManagedDataset, MergeStrategy, SequenceRecord


class DatasetManager:
    def __init__(self, repository: DatasetRepository) -> None:
        self._repository = repository

    def save_records(
        self,
        source_name: str,
        request: FetchRequest,
        records: list[SequenceRecord],
        merge_strategy: MergeStrategy = "replace",
    ) -> DatasetArtifact:
        if merge_strategy not in ("replace", "skip_duplicates", "upsert"):
            raise NotImplementedError(f"Unsupported merge_strategy: {merge_strategy!r}")
        deduplicated_records = self._deduplicate(records)
        return self._repository.save_dataset(
            source_name=source_name,
            request=request,
            records=deduplicated_records,
            merge_strategy=merge_strategy,
        )

    def save_prebuilt_dataset(
        self,
        source_name: str,
        dataset_name: str,
        train_text: str,
        tokenizer_map_text: str,
        record_count: int,
    ) -> DatasetArtifact:
        return self._repository.save_prebuilt_dataset(
            source_name=source_name,
            dataset_name=dataset_name,
            train_text=train_text,
            tokenizer_map_text=tokenizer_map_text,
            record_count=record_count,
        )

    def list_datasets(self, source_name: str | None = None) -> list[ManagedDataset]:
        return self._repository.list_datasets(source_name=source_name)

    def delete_dataset(self, source_name: str, dataset_name: str, permanent: bool = False) -> DeleteResult:
        return self._repository.delete_dataset(
            source_name=source_name,
            dataset_name=dataset_name,
            permanent=permanent,
        )

    def _deduplicate(self, records: list[SequenceRecord]) -> list[SequenceRecord]:
        deduplicated_by_accession: dict[str, SequenceRecord] = {}
        for record in records:
            deduplicated_by_accession[record.accession] = record
        return list(deduplicated_by_accession.values())

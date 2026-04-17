from __future__ import annotations

import logging
import shutil

from libs.data.backends.manager import DatasetManager
from libs.data.config import DataConfig
from libs.data.entities import FetchRequest, PreparationSessionArtifact
from libs.data.training.normalization import SequenceNormalizationConfig
from libs.data.training.tokenizer import SequenceTokenizer
from libs.data.utilities.exceptions import DataNotFoundError, SourceConfigurationError
from libs.data.utilities.storage import render_tokenizer_map_payload

from .helpers import accession_key_for_record, chunked, sequence_type_from_text
from .rebuild import build_fetch_plan, needs_rebuild, raw_index_entry, rebuild_dataset, resolve_sequence_versions
from .state import (
    artifact_from_manifest,
    normalize_requested_accessions,
    read_manifest,
    read_raw_index,
    request_payload,
    request_signature,
    session_dir,
    write_manifest,
    write_raw_index,
)

logger = logging.getLogger(__name__)


class ResumableTrainingDataPreparer:
    def __init__(self, dataset_manager: DatasetManager, config: DataConfig) -> None:
        self._dataset_manager = dataset_manager
        self._config = config

    def prepare(
        self,
        source_name: str,
        source,
        request: FetchRequest,
        sequence_type: str,
        normalization: SequenceNormalizationConfig,
        vocab_size: int | None = None,
        restart: bool = False,
    ) -> PreparationSessionArtifact:
        active_session_dir = session_dir(self._config, source_name, request.dataset_name)
        manifest_path = active_session_dir / "manifest.json"
        accessions_path = active_session_dir / "accessions.txt"
        raw_index_path = active_session_dir / "raw_index.json"
        stage_train_path = active_session_dir / "train.txt"
        stage_tokenizer_map_path = active_session_dir / "tokenizer_map.json"

        if restart and active_session_dir.exists():
            shutil.rmtree(active_session_dir)

        active_session_dir.mkdir(parents=True, exist_ok=True)

        signature = self._request_signature(
            source_name=source_name,
            request=request,
            sequence_type=sequence_type,
            normalization=normalization,
            vocab_size=vocab_size,
        )
        manifest = read_manifest(manifest_path) if manifest_path.exists() else {}
        raw_index = read_raw_index(raw_index_path)
        for entry in raw_index.values():
            entry.pop("dna_sequence", None)

        resolved_accessions = self._resolve_accessions(source_name=source_name, source=source, request=request)
        accessions, accession_aliases, duplicate_accession_count = normalize_requested_accessions(resolved_accessions)
        if not accessions:
            raise DataNotFoundError(f"No accessions resolved for dataset '{request.dataset_name}'")

        accessions_path.write_text(
            "\n".join(accession_aliases[accession] for accession in accessions) + "\n",
            encoding="utf-8",
        )

        version_map = resolve_sequence_versions(
            source=source,
            accessions=accessions,
            accession_aliases=accession_aliases,
            raw_index=raw_index,
            request=request,
        )
        fetch_plan = build_fetch_plan(
            accessions=accessions,
            accession_aliases=accession_aliases,
            raw_index=raw_index,
            version_map=version_map,
        )

        if not needs_rebuild(
            manifest=manifest,
            raw_index=raw_index,
            accessions=accessions,
            normalization=normalization,
            sequence_type=sequence_type,
            vocab_size=vocab_size,
            duplicate_accession_count=duplicate_accession_count,
            fetch_plan=fetch_plan,
            storage_mode=self._config.storage_mode,
        ):
            return artifact_from_manifest(self._config, manifest, active_session_dir, manifest_path)

        manifest.update(
            {
                "signature": signature,
                "source_name": source_name,
                "dataset_name": request.dataset_name,
                "storage_mode": self._config.storage_mode,
                "sequence_type": sequence_type,
                "vocab_size": vocab_size,
                "processed_count": len(fetch_plan.unchanged_accessions),
                "total_count": len(accessions),
                "record_count": 0,
                "dropped_record_count": 0,
                "dropped_reasons": {},
                "is_complete": False,
                "request": request_payload(
                    request=request,
                    accessions=accessions,
                    duplicate_accession_count=duplicate_accession_count,
                ),
                "normalization": {
                    "sequence_type": normalization.sequence_type,
                    "min_length": normalization.min_length,
                    "max_length": normalization.max_length,
                    "invalid_base_policy": normalization.invalid_base_policy,
                    "max_ambiguous_ratio": normalization.max_ambiguous_ratio,
                    "deduplicate_sequences": normalization.deduplicate_sequences,
                },
                "raw_index_path": str(raw_index_path),
                "indexed_accession_count": len(raw_index),
                "fetch_summary": {
                    "new_accessions": len(fetch_plan.new_accessions),
                    "updated_accessions": len(fetch_plan.updated_accessions),
                    "unchanged_accessions": len(fetch_plan.unchanged_accessions),
                    "fetched_accessions": 0,
                },
            }
        )
        write_manifest(manifest_path, manifest)

        batch_size = request.batch_size or self._config.default_batch_size
        fetched_accession_count = 0

        for accession_batch in chunked(fetch_plan.accessions_to_fetch, batch_size):
            requested_accession_batch = tuple(accession_aliases[accession] for accession in accession_batch)
            batch_request = FetchRequest(
                dataset_name=request.dataset_name,
                accessions=requested_accession_batch,
                limit=len(requested_accession_batch),
                batch_size=len(requested_accession_batch),
                extra_fields=request.extra_fields,
                include_suppressed=request.include_suppressed,
            )
            fetched_records = source.fetch(batch_request)
            fetched_by_accession = {
                accession_key_for_record(record): record
                for record in fetched_records
            }

            for accession in accession_batch:
                record = fetched_by_accession.get(accession)
                if record is None:
                    continue
                raw_index[accession] = raw_index_entry(
                    record=record,
                    requested_accession=accession_aliases[accession],
                )

            fetched_accession_count += len(accession_batch)
            manifest["processed_count"] = len(fetch_plan.unchanged_accessions) + fetched_accession_count
            manifest["indexed_accession_count"] = len(raw_index)
            manifest["fetch_summary"] = {
                "new_accessions": len(fetch_plan.new_accessions),
                "updated_accessions": len(fetch_plan.updated_accessions),
                "unchanged_accessions": len(fetch_plan.unchanged_accessions),
                "fetched_accessions": fetched_accession_count,
            }
            write_raw_index(raw_index_path, raw_index)
            write_manifest(manifest_path, manifest)

        rebuild_result = rebuild_dataset(
            raw_index=raw_index,
            accessions=accessions,
            accession_aliases=accession_aliases,
            normalization=normalization,
            duplicate_accession_count=duplicate_accession_count,
        )
        write_raw_index(raw_index_path, raw_index)

        train_text = rebuild_result.train_text
        if not train_text.strip():
            raise DataNotFoundError(
                f"All fetched records were filtered out while preparing training data for '{request.dataset_name}'"
            )

        stage_train_path.write_text(train_text, encoding="utf-8")

        effective_sequence_type = sequence_type_from_text(train_text)
        tokenizer = SequenceTokenizer.from_text(
            train_text,
            sequence_type=effective_sequence_type,
            vocab_size=vocab_size,
        )
        tokenizer_map_text = render_tokenizer_map_payload(
            source_name=source_name,
            record_count=rebuild_result.record_count,
            tokenizer=tokenizer,
        )
        stage_tokenizer_map_path.write_text(tokenizer_map_text, encoding="utf-8")

        dataset_artifact = self._dataset_manager.save_prebuilt_dataset(
            source_name=source_name,
            dataset_name=request.dataset_name,
            train_text=train_text,
            tokenizer_map_text=tokenizer_map_text,
            record_count=rebuild_result.record_count,
        )

        manifest.update(
            {
                "storage_mode": dataset_artifact.storage_mode,
                "processed_count": len(accessions),
                "total_count": len(accessions),
                "record_count": rebuild_result.record_count,
                "dropped_record_count": rebuild_result.dropped_count,
                "dropped_reasons": rebuild_result.dropped_reasons,
                "is_complete": True,
                "current_location": dataset_artifact.current_location,
                "snapshot_id": dataset_artifact.snapshot_id,
                "train_txt_path": dataset_artifact.file_locations["train.txt"],
                "tokenizer_map_path": dataset_artifact.file_locations["tokenizer_map.json"],
                "sequence_type": tokenizer.sequence_type,
                "vocab_size": tokenizer.vocab_size,
                "indexed_accession_count": len(raw_index),
                "request": request_payload(
                    request=request,
                    accessions=accessions,
                    duplicate_accession_count=duplicate_accession_count,
                ),
                "fetch_summary": {
                    "new_accessions": len(fetch_plan.new_accessions),
                    "updated_accessions": len(fetch_plan.updated_accessions),
                    "unchanged_accessions": len(fetch_plan.unchanged_accessions),
                    "fetched_accessions": len(fetch_plan.accessions_to_fetch),
                },
            }
        )
        write_manifest(manifest_path, manifest)

        return artifact_from_manifest(self._config, manifest, active_session_dir, manifest_path)

    def _resolve_accessions(self, source_name: str, source, request: FetchRequest) -> tuple[str, ...]:
        resolver = getattr(source, "resolve_accessions", None)
        if callable(resolver):
            return tuple(resolver(request))

        if request.accessions:
            if request.effective_limit is None:
                return request.accessions
            return request.accessions[: request.effective_limit]

        raise SourceConfigurationError(f"Resumable preparation does not support source '{source_name}'.")

    def _request_signature(
        self,
        source_name: str,
        request: FetchRequest,
        sequence_type: str,
        normalization: SequenceNormalizationConfig,
        vocab_size: int | None,
    ) -> str:
        return request_signature(
            config=self._config,
            source_name=source_name,
            request=request,
            sequence_type=sequence_type,
            normalization=normalization,
            vocab_size=vocab_size,
        )

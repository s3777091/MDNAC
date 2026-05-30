from __future__ import annotations

import logging

from libs.data.contracts import HttpTransport, SequenceSource
from libs.data.entities import FetchRequest, SequenceRecord
from libs.data.utilities.exceptions import DataNotFoundError, SourceConfigurationError
from libs.data.utilities.parsers import parse_ddbj_flatfile, parse_fasta
from libs.data.utilities.retry import RetryPolicy

logger = logging.getLogger(__name__)

_DDBJ_RETRY_POLICY = RetryPolicy(max_retries=3, backoff_base=5.0)


class DdbjSequenceSource(SequenceSource):
    name = "ddbj"

    _GETENTRY_URL = "https://getentry.ddbj.nig.ac.jp/getentry"

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def fetch(self, request: FetchRequest) -> list[SequenceRecord]:
        if not request.accessions:
            raise SourceConfigurationError(
                "DDBJ ingestion currently requires explicit accession lists because the stable getentry API is accession-based."
            )

        accessions = self._limited_accessions(request)
        metadata_by_accession = {}
        fasta_entries = []
        batch_size = request.batch_size or len(accessions) or 1

        for accession_batch in self._chunked(accessions, batch_size):
            metadata_by_accession.update(self._fetch_metadata(accession_batch, request))
            fasta_entries.extend(self._fetch_fasta_entries(accession_batch, request))

        if not fasta_entries:
            raise DataNotFoundError(f"DDBJ returned no FASTA sequences for dataset '{request.dataset_name}'")

        records: list[SequenceRecord] = []
        for entry in fasta_entries:
            metadata_entry = metadata_by_accession.get(entry.accession)
            description = self._header_description(entry.header)
            organism = ""
            sequence_version = None
            metadata = {}

            if metadata_entry:
                description = metadata_entry.description or description
                organism = metadata_entry.organism
                sequence_version = metadata_entry.version

            if entry.header:
                metadata["fasta_header"] = entry.header

            records.append(
                SequenceRecord(
                    accession=entry.accession,
                    source_name=self.name,
                    description=description,
                    organism=organism,
                    sequence=entry.sequence,
                    sequence_length=len(entry.sequence),
                    sequence_version=sequence_version,
                    metadata=metadata,
                )
            )

        return records

    def resolve_sequence_versions(
        self,
        accessions: tuple[str, ...],
        request: FetchRequest,
    ) -> dict[str, str | None]:
        if not accessions:
            return {}

        versions: dict[str, str | None] = {}
        batch_size = request.batch_size or len(accessions) or 1
        for accession_batch in self._chunked(accessions, batch_size):
            metadata_by_accession = self._fetch_metadata(accession_batch, request)
            for accession in accession_batch:
                metadata_entry = metadata_by_accession.get(accession)
                versions[accession] = metadata_entry.version if metadata_entry is not None else None
        return versions

    def _fetch_fasta_entries(self, accessions: tuple[str, ...], request: FetchRequest):
        params = self._base_params(accessions, request)
        params["format"] = "fasta"

        def _do_fetch():
            response_text = self._transport.get_text(self._GETENTRY_URL, params=params)
            return parse_fasta(response_text)

        return _DDBJ_RETRY_POLICY.execute(
            operation=_do_fetch,
            is_empty=lambda entries: not entries,
            context=f"DDBJ FASTA for {len(accessions)} accessions",
        )

    def _fetch_metadata(self, accessions: tuple[str, ...], request: FetchRequest):
        params = self._base_params(accessions, request)
        params["format"] = "flatfile"

        def _do_fetch():
            response_text = self._transport.get_text(self._GETENTRY_URL, params=params)
            return parse_ddbj_flatfile(response_text)

        return _DDBJ_RETRY_POLICY.execute(
            operation=_do_fetch,
            is_empty=lambda parsed: not parsed,
            context=f"DDBJ flatfile for {len(accessions)} accessions",
        )

    def _base_params(self, accessions: tuple[str, ...], request: FetchRequest) -> dict[str, object]:
        params: dict[str, object] = {
            "database": "aa",
            "accession_number": ",".join(accessions),
            "filetype": "text",
            "limit": 0 if request.effective_limit is None else len(accessions),
        }
        if request.include_suppressed:
            params["show_suppressed"] = "true"
        return params

    def _limited_accessions(self, request: FetchRequest) -> tuple[str, ...]:
        effective_limit = request.effective_limit
        if effective_limit is None:
            return request.accessions
        return request.accessions[:effective_limit]

    def _chunked(self, items: tuple[str, ...], batch_size: int) -> list[tuple[str, ...]]:
        return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]

    def _header_description(self, header: str) -> str:
        _, _, description = header.partition(" ")
        return description.strip()

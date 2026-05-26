from __future__ import annotations

import logging
import time

from libs.data.contracts import HttpTransport, SequenceSource
from libs.data.entities import FetchRequest, SequenceRecord
from libs.data.utilities.exceptions import DataNotFoundError, SourceConfigurationError
from libs.data.utilities.parsers import parse_ena_coding_embl, parse_fasta, parse_tsv_rows

logger = logging.getLogger(__name__)

_ENA_EMPTY_RESPONSE_MAX_RETRIES = 3
_ENA_EMPTY_RESPONSE_BACKOFF_BASE = 5.0


class EnaSequenceSource(SequenceSource):
    name = "ena"

    _PORTAL_SEARCH_URL = "https://www.ebi.ac.uk/ena/portal/api/search"
    _BROWSER_FASTA_SEARCH_URL = "https://www.ebi.ac.uk/ena/browser/api/fasta/search"
    _BROWSER_EMBL_URL = "https://www.ebi.ac.uk/ena/browser/api/embl"
    _DEFAULT_RESULT_TYPE = "coding"
    _CODING_RESULT_TYPE = "coding"
    _DEFAULT_PORTAL_PAGE_SIZE = 1000
    _DEFAULT_EMBL_BATCH_SIZE = 25
    _DEFAULT_FIELDS = ("accession", "description", "scientific_name", "base_count")
    _CODING_DEFAULT_FIELDS = (
        "accession",
        "description",
        "scientific_name",
        "base_count",
        "protein_id",
        "parent_accession",
        "sequence_version",
    )

    def __init__(self, transport: HttpTransport, result_type: str | None = None) -> None:
        self._transport = transport
        self._result_type = result_type or self._DEFAULT_RESULT_TYPE

    def fetch(self, request: FetchRequest) -> list[SequenceRecord]:
        query = self._build_query(request)
        metadata_rows = self._fetch_metadata_rows(query, request)
        if self._uses_coding_records():
            coding_entries = self._fetch_coding_entries(request=request, metadata_rows=metadata_rows)
            if not coding_entries:
                raise DataNotFoundError(
                    f"ENA returned no translated coding sequences for dataset '{request.dataset_name}'"
                )
            metadata_by_accession = self._build_metadata_index(metadata_rows)
            return self._coding_records_from_entries(coding_entries, metadata_by_accession)

        fasta_entries = self._fetch_fasta_entries(query, request)
        if not fasta_entries:
            raise DataNotFoundError(f"ENA returned no FASTA sequences for dataset '{request.dataset_name}'")

        metadata_by_accession = self._build_metadata_index(metadata_rows)
        records: list[SequenceRecord] = []
        for entry in fasta_entries:
            resolved_accession = self._resolve_entry_accession(entry.header, entry.accession, metadata_by_accession)
            row = metadata_by_accession.get(resolved_accession, {})
            description = row.get("description") or self._header_description(entry.header)
            organism = row.get("scientific_name", "")
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"accession", "description", "scientific_name", "base_count"} and value
            }
            if entry.header:
                metadata.setdefault("fasta_header", entry.header)

            records.append(
                SequenceRecord(
                    accession=resolved_accession,
                    source_name=self.name,
                    description=description,
                    organism=organism,
                    sequence=entry.sequence,
                    sequence_length=self._sequence_length(row.get("base_count"), entry.sequence),
                    metadata=metadata,
                )
            )

        return records

    def resolve_accessions(self, request: FetchRequest) -> tuple[str, ...]:
        if request.accessions:
            return self._limited_accessions(request)

        accession_field = self._portal_accession_field()
        rows = self._fetch_portal_rows(
            query=request.query or "",
            request=request,
            fields=(accession_field,),
            empty_label="accessions",
        )
        accessions = tuple(
            accession
            for row in rows
            if (accession := (row.get(accession_field, "") or row.get("accession", "")).strip())
        )
        if request.effective_limit is None:
            return accessions
        return accessions[: request.effective_limit]

    def _build_query(self, request: FetchRequest) -> str:
        if request.query:
            return request.query

        requested_accessions = self._limited_accessions(request)
        query_terms: list[str] = []
        for accession in requested_accessions:
            query_terms.extend(self._accession_query_terms(accession))
        return " OR ".join(query_terms)

    def _fetch_metadata_rows(self, query: str, request: FetchRequest) -> list[dict[str, str]]:
        default_fields = self._CODING_DEFAULT_FIELDS if self._uses_coding_records() else self._DEFAULT_FIELDS
        fields = list(dict.fromkeys([*default_fields, *request.extra_fields]))
        return self._fetch_portal_rows(
            query=query,
            request=request,
            fields=tuple(fields),
            empty_label="metadata",
        )

    def _fetch_fasta_entries(self, query: str, request: FetchRequest):
        params = {
            "result": self._result_type,
            "query": query,
            "limit": self._request_limit_value(request),
        }
        for retry in range(_ENA_EMPTY_RESPONSE_MAX_RETRIES + 1):
            response_text = self._transport.get_text(self._BROWSER_FASTA_SEARCH_URL, params=params)
            entries = parse_fasta(response_text)
            if entries:
                break
            if retry < _ENA_EMPTY_RESPONSE_MAX_RETRIES:
                delay = _ENA_EMPTY_RESPONSE_BACKOFF_BASE * (retry + 1)
                logger.warning(
                    "ENA browser returned no FASTA entries (attempt %d/%d), retrying in %.0fs",
                    retry + 1, _ENA_EMPTY_RESPONSE_MAX_RETRIES + 1, delay,
                )
                time.sleep(delay)
        effective_limit = request.effective_limit
        if effective_limit is None:
            return entries
        return entries[:effective_limit]

    def _fetch_coding_entries(self, request: FetchRequest, metadata_rows: list[dict[str, str]]):
        accessions = self._coding_record_accessions(request, metadata_rows)
        if not accessions:
            return []

        entries = []
        batch_size = request.batch_size or self._DEFAULT_EMBL_BATCH_SIZE
        for accession_batch in self._chunked(accessions, batch_size):
            url = f"{self._BROWSER_EMBL_URL}/{','.join(accession_batch)}"
            batch_entries = []
            for retry in range(_ENA_EMPTY_RESPONSE_MAX_RETRIES + 1):
                response_text = self._transport.get_text(url)
                batch_entries = parse_ena_coding_embl(response_text)
                if batch_entries:
                    break
                if retry < _ENA_EMPTY_RESPONSE_MAX_RETRIES:
                    delay = _ENA_EMPTY_RESPONSE_BACKOFF_BASE * (retry + 1)
                    logger.warning(
                        "ENA browser returned no coding EMBL entries (attempt %d/%d), retrying in %.0fs",
                        retry + 1, _ENA_EMPTY_RESPONSE_MAX_RETRIES + 1, delay,
                    )
                    time.sleep(delay)
            entries.extend(batch_entries)
            if request.effective_limit is not None and len(entries) >= request.effective_limit:
                break

        effective_limit = request.effective_limit
        if effective_limit is None:
            return entries
        return entries[:effective_limit]

    def _limited_accessions(self, request: FetchRequest) -> tuple[str, ...]:
        accessions = request.accessions
        effective_limit = request.effective_limit
        if effective_limit is None:
            return accessions
        return accessions[:effective_limit]

    def _request_limit_value(self, request: FetchRequest) -> int:
        if request.effective_limit is None:
            return 0
        if request.accessions:
            return min(request.effective_limit, len(request.accessions))
        return request.effective_limit

    def _fetch_portal_rows(
        self,
        query: str,
        request: FetchRequest,
        fields: tuple[str, ...],
        empty_label: str,
    ) -> list[dict[str, str]]:
        if request.accessions:
            return self._fetch_portal_page(
                query=query,
                fields=fields,
                limit=self._request_limit_value(request),
                empty_label=empty_label,
            )

        effective_limit = request.effective_limit
        if effective_limit is None:
            raise SourceConfigurationError(
                "ENA query requests require a positive limit. "
                "The ENA portal API no longer supports offset pagination for full-query scans."
            )

        return self._fetch_portal_page(
            query=query,
            fields=fields,
            limit=effective_limit,
            empty_label=empty_label,
        )

    def _fetch_portal_page(
        self,
        query: str,
        fields: tuple[str, ...],
        limit: int,
        empty_label: str,
        offset: int = 0,
    ) -> list[dict[str, str]]:
        params: dict[str, object] = {
            "result": self._result_type,
            "query": query,
            "format": "tsv",
            "fields": ",".join(fields),
            "limit": limit,
        }
        if offset > 0:
            params["offset"] = offset

        for retry in range(_ENA_EMPTY_RESPONSE_MAX_RETRIES + 1):
            response_text = self._transport.get_text(self._PORTAL_SEARCH_URL, params=params)
            rows = parse_tsv_rows(response_text)
            if rows:
                return rows
            if retry < _ENA_EMPTY_RESPONSE_MAX_RETRIES:
                delay = _ENA_EMPTY_RESPONSE_BACKOFF_BASE * (retry + 1)
                logger.warning(
                    "ENA portal returned empty %s page at offset=%d (attempt %d/%d), retrying in %.0fs",
                    empty_label,
                    offset,
                    retry + 1,
                    _ENA_EMPTY_RESPONSE_MAX_RETRIES + 1,
                    delay,
                )
                time.sleep(delay)
        return []

    def _uses_coding_records(self) -> bool:
        return self._result_type == self._CODING_RESULT_TYPE

    def _portal_accession_field(self) -> str:
        return "protein_id" if self._uses_coding_records() else "accession"

    def _accession_query_terms(self, accession: str) -> tuple[str, ...]:
        cleaned = accession.strip()
        if not cleaned:
            return ()
        if not self._uses_coding_records():
            return (f'accession="{cleaned}"',)

        canonical = self._canonical_accession(cleaned)
        terms: list[str] = []
        if cleaned != canonical:
            terms.append(f'protein_id="{cleaned}"')
        if canonical:
            terms.append(f'accession="{canonical}"')
        return tuple(dict.fromkeys(terms))

    def _coding_record_accessions(self, request: FetchRequest, metadata_rows: list[dict[str, str]]) -> tuple[str, ...]:
        raw_accessions = self._limited_accessions(request)
        if not raw_accessions:
            raw_accessions = tuple(
                accession
                for row in metadata_rows
                if (accession := row.get("accession", "").strip())
            )

        ordered_accessions: list[str] = []
        seen: set[str] = set()
        for accession in raw_accessions:
            resolved = self._canonical_accession(accession) or accession.strip()
            if not resolved or resolved in seen:
                continue
            seen.add(resolved)
            ordered_accessions.append(resolved)
        return tuple(ordered_accessions)

    def _sequence_length(self, raw_length: str | None, sequence: str) -> int:
        if raw_length and raw_length.isdigit():
            return int(raw_length)
        return len(sequence)

    def _header_description(self, header: str) -> str:
        _, _, description = header.partition(" ")
        return description.strip()

    def _build_metadata_index(self, rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
        indexed: dict[str, dict[str, str]] = {}
        for row in rows:
            for field_name in ("accession", "protein_id"):
                accession = row.get(field_name, "").strip()
                if not accession:
                    continue
                indexed[accession] = row
                canonical = self._canonical_accession(accession)
                if canonical and canonical not in indexed:
                    indexed[canonical] = row
        return indexed

    def _coding_records_from_entries(
        self,
        coding_entries,
        metadata_by_accession: dict[str, dict[str, str]],
    ) -> list[SequenceRecord]:
        records: list[SequenceRecord] = []
        for entry in coding_entries:
            resolved_accession = self._resolve_coding_entry_accession(entry.accession, entry.protein_id, metadata_by_accession)
            row = metadata_by_accession.get(resolved_accession, {})
            description = row.get("description") or entry.description
            organism = row.get("scientific_name") or entry.organism
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"accession", "description", "scientific_name", "base_count"} and value
            }
            if row.get("base_count"):
                metadata.setdefault("coding_base_count", row["base_count"])
            if entry.parent_accession:
                metadata.setdefault("parent_accession", entry.parent_accession)
            metadata.setdefault("ena_record_type", "coding_translation")

            records.append(
                SequenceRecord(
                    accession=resolved_accession,
                    source_name=self.name,
                    description=description,
                    organism=organism,
                    sequence=entry.translation,
                    sequence_length=len(entry.translation),
                    sequence_version=entry.protein_id or entry.sequence_version,
                    metadata=metadata,
                )
            )

        return records

    def _resolve_coding_entry_accession(
        self,
        accession: str,
        protein_id: str | None,
        metadata_by_accession: dict[str, dict[str, str]],
    ) -> str:
        for candidate in (protein_id or "", accession):
            cleaned = candidate.strip()
            if not cleaned:
                continue
            canonical = self._canonical_accession(cleaned)
            if cleaned in metadata_by_accession:
                return canonical or cleaned
            if canonical and canonical in metadata_by_accession:
                return canonical
        return self._canonical_accession(accession) or accession

    def _resolve_entry_accession(
        self,
        header: str,
        parsed_accession: str,
        metadata_by_accession: dict[str, dict[str, str]],
    ) -> str:
        for candidate in self._accession_candidates(header, parsed_accession):
            if candidate in metadata_by_accession:
                return candidate
        return self._canonical_accession(parsed_accession) or parsed_accession

    def _accession_candidates(self, header: str, parsed_accession: str) -> tuple[str, ...]:
        candidates: list[str] = []
        seen: set[str] = set()
        header_token = header.split()[0] if header else ""

        def add_candidate(value: str) -> None:
            cleaned = value.strip().strip("|")
            if not cleaned:
                return
            if cleaned not in seen:
                candidates.append(cleaned)
                seen.add(cleaned)
            canonical = self._canonical_accession(cleaned)
            if canonical and canonical not in seen:
                candidates.append(canonical)
                seen.add(canonical)

        add_candidate(parsed_accession)
        add_candidate(header_token)

        for token in (parsed_accession, header_token):
            if "|" not in token:
                continue
            for part in token.split("|"):
                add_candidate(part)

        return tuple(candidates)

    def _canonical_accession(self, accession: str) -> str:
        value = accession.strip()
        if not value:
            return ""
        prefix, separator, suffix = value.rpartition(".")
        if separator and suffix.isdigit() and prefix:
            return prefix
        return value

    def _chunked(self, items: tuple[str, ...], batch_size: int) -> list[tuple[str, ...]]:
        return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]

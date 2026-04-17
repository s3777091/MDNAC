from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

from libs.data.contracts import HttpTransport, SequenceSource

logger = logging.getLogger(__name__)

_NCBI_EMPTY_RESPONSE_MAX_RETRIES = 3
_NCBI_EMPTY_RESPONSE_BACKOFF_BASE = 5.0
from libs.data.entities import FetchRequest, SequenceRecord
from libs.data.utilities.exceptions import DataNotFoundError, SourceConfigurationError
from libs.data.utilities.parsers import parse_fasta


@dataclass(slots=True, frozen=True)
class NcbiSummaryEntry:
    accession: str
    accession_version: str | None
    description: str
    organism: str
    sequence_length: int | None
    metadata: dict[str, str]


class NcbiSequenceSource(SequenceSource):
    name = "ncbi"

    _ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    _ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    _EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    _DEFAULT_FETCH_BATCH_SIZE = 200
    _DEFAULT_SEARCH_PAGE_SIZE = 500
    _MAX_ESEARCH_PAGE_SIZE = 10_000
    _DEFAULT_METADATA_FIELDS = (
        "uid",
        "caption",
        "sourcedb",
        "biomol",
        "moltype",
        "topology",
        "genome",
        "completeness",
        "tech",
        "assemblyacc",
        "biosample",
        "strain",
        "projectid",
        "taxid",
        "subtype",
        "subname",
    )

    def __init__(
        self,
        transport: HttpTransport,
        tool_name: str | None = None,
        email: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._transport = transport
        self._tool_name = self._clean_optional_value(tool_name) or os.getenv(
            "MICROBIAL_DATA_NCBI_TOOL",
            "microbial-dna-compiler",
        )
        self._email = self._clean_optional_value(email) or self._clean_optional_value(
            os.getenv("MICROBIAL_DATA_NCBI_EMAIL")
        )
        self._api_key = self._clean_optional_value(api_key) or self._clean_optional_value(
            os.getenv("MICROBIAL_DATA_NCBI_API_KEY")
        )

    def _require_credentials(self) -> None:
        if not self._email:
            raise SourceConfigurationError(
                "NCBI E-utilities requires an email address. "
                "Set the MICROBIAL_DATA_NCBI_EMAIL environment variable or pass email= to NcbiSequenceSource. "
                "Optionally set MICROBIAL_DATA_NCBI_API_KEY for higher rate limits (10 req/s vs 3 req/s)."
            )

    def fetch(self, request: FetchRequest) -> list[SequenceRecord]:
        self._require_credentials()
        accessions = self.resolve_accessions(request)
        if not accessions:
            raise DataNotFoundError(f"NCBI returned no accessions for dataset '{request.dataset_name}'")

        metadata_by_accession: dict[str, NcbiSummaryEntry] = {}
        fasta_entries = []
        batch_size = request.batch_size or self._DEFAULT_FETCH_BATCH_SIZE

        for accession_batch in self._chunked(accessions, batch_size):
            metadata_by_accession.update(self._fetch_metadata(accession_batch, request))
            fasta_entries.extend(self._fetch_fasta_entries(accession_batch))

        if not fasta_entries:
            raise DataNotFoundError(f"NCBI returned no FASTA sequences for dataset '{request.dataset_name}'")

        records: list[SequenceRecord] = []
        for entry in fasta_entries:
            metadata_entry = metadata_by_accession.get(entry.accession)
            description = self._header_description(entry.header)
            organism = ""
            sequence_version = None
            metadata: dict[str, str] = {}

            if metadata_entry is not None:
                description = metadata_entry.description or description
                organism = metadata_entry.organism
                sequence_version = metadata_entry.accession_version
                metadata = dict(metadata_entry.metadata)

            if entry.header:
                metadata["fasta_header"] = entry.header

            records.append(
                SequenceRecord(
                    accession=entry.accession,
                    source_name=self.name,
                    description=description,
                    organism=organism,
                    sequence=entry.sequence,
                    sequence_length=metadata_entry.sequence_length if metadata_entry and metadata_entry.sequence_length else len(entry.sequence),
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
        batch_size = request.batch_size or self._DEFAULT_FETCH_BATCH_SIZE
        for accession_batch in self._chunked(accessions, batch_size):
            metadata_by_accession = self._fetch_metadata(accession_batch, request)
            for accession in accession_batch:
                metadata_entry = metadata_by_accession.get(accession)
                versions[self._canonical_accession(accession)] = metadata_entry.accession_version if metadata_entry is not None else None
        return versions

    def resolve_accessions(self, request: FetchRequest) -> tuple[str, ...]:
        self._require_credentials()
        if request.accessions:
            return self._limited_accessions(request)

        accessions: list[str] = []
        seen: set[str] = set()
        effective_limit = request.effective_limit

        for accession in self.iter_accessions(request):
            if accession in seen:
                continue
            seen.add(accession)
            accessions.append(accession)
            if effective_limit is not None and len(accessions) >= effective_limit:
                break

        return tuple(accessions)

    def iter_accessions(self, request: FetchRequest):
        """Yield NCBI accessions page by page without materializing the full result set."""
        self._require_credentials()
        if request.accessions:
            for accession in self._limited_accessions(request):
                yield accession
            return

        yielded_count = 0
        for page_accessions in self._iter_accession_pages(request):
            for accession in page_accessions:
                yield accession
                yielded_count += 1
                if request.effective_limit is not None and yielded_count >= request.effective_limit:
                    return

    def _iter_accession_pages(self, request: FetchRequest):
        retstart = 0
        total_count: int | None = None
        yielded_count = 0
        effective_limit = request.effective_limit
        consecutive_empty_pages = 0

        while effective_limit is None or yielded_count < effective_limit:
            retmax = self._search_page_size(effective_limit, current_count=yielded_count)
            if retmax <= 0:
                return

            params = self._base_params()
            params.update(
                {
                    "db": "protein",
                    "term": request.query or "",
                    "idtype": "acc",
                    "retmode": "json",
                    "retstart": retstart,
                    "retmax": retmax,
                }
            )
            response_text = self._transport.get_text(self._ESEARCH_URL, params=params)
            page_accessions, page_total_count = self._parse_esearch_response(response_text)
            if total_count is None:
                total_count = page_total_count

            if not page_accessions:
                consecutive_empty_pages += 1
                if consecutive_empty_pages <= _NCBI_EMPTY_RESPONSE_MAX_RETRIES:
                    delay = _NCBI_EMPTY_RESPONSE_BACKOFF_BASE * consecutive_empty_pages
                    logger.warning(
                        "NCBI esearch returned empty page at retstart=%d (attempt %d/%d), retrying in %.0fs",
                        retstart, consecutive_empty_pages, _NCBI_EMPTY_RESPONSE_MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                return
            consecutive_empty_pages = 0

            yield page_accessions

            yielded_count += len(page_accessions)
            retstart += len(page_accessions)
            if len(page_accessions) < retmax:
                return
            if total_count is not None and retstart >= total_count:
                return

    def _fetch_metadata(self, accessions: tuple[str, ...], request: FetchRequest) -> dict[str, NcbiSummaryEntry]:
        params = self._base_params()
        params.update(
            {
                "db": "protein",
                "id": ",".join(accessions),
                "retmode": "json",
            }
        )
        result: dict = {}
        for retry in range(_NCBI_EMPTY_RESPONSE_MAX_RETRIES + 1):
            response_text = self._transport.get_text(self._ESUMMARY_URL, params=params)
            if not response_text or not response_text.strip():
                if retry < _NCBI_EMPTY_RESPONSE_MAX_RETRIES:
                    delay = _NCBI_EMPTY_RESPONSE_BACKOFF_BASE * (retry + 1)
                    logger.warning(
                        "NCBI esummary returned empty response (attempt %d/%d), retrying in %.0fs",
                        retry + 1, _NCBI_EMPTY_RESPONSE_MAX_RETRIES + 1, delay,
                    )
                    time.sleep(delay)
                    continue
                return {}
            try:
                payload = json.loads(response_text)
            except json.JSONDecodeError:
                if retry < _NCBI_EMPTY_RESPONSE_MAX_RETRIES:
                    delay = _NCBI_EMPTY_RESPONSE_BACKOFF_BASE * (retry + 1)
                    logger.warning(
                        "NCBI esummary returned invalid JSON (attempt %d/%d), retrying in %.0fs",
                        retry + 1, _NCBI_EMPTY_RESPONSE_MAX_RETRIES + 1, delay,
                    )
                    time.sleep(delay)
                    continue
                return {}
            result = payload.get("result", {})
            break

        parsed: dict[str, NcbiSummaryEntry] = {}
        for uid in result.get("uids", []):
            docsum = result.get(str(uid), {})
            if not isinstance(docsum, dict):
                continue

            accession_version = self._string_value(docsum.get("accessionversion"))
            accession = accession_version or self._string_value(docsum.get("caption"))
            if not accession:
                continue

            metadata = self._build_metadata(docsum, request.extra_fields)
            entry = NcbiSummaryEntry(
                accession=accession,
                accession_version=accession_version,
                description=self._string_value(docsum.get("title")) or "",
                organism=self._string_value(docsum.get("organism")) or "",
                sequence_length=self._int_value(docsum.get("slen")),
                metadata=metadata,
            )
            parsed[accession] = entry

            caption = self._string_value(docsum.get("caption"))
            if caption and caption not in parsed:
                parsed[caption] = entry

        return parsed

    def _removed_cds_dna_mapping(
        self,
        accessions: tuple[str, ...],
        batch_size: int = 200,
    ) -> dict[str, str]:
        del accessions, batch_size
        raise ValueError("NCBI CDS DNA back-mapping has been removed from the protein-only pipeline.")

    def _extract_protein_accession_from_cds_header(self, header: str) -> str:
        """Extract the protein accession from a CDS FASTA header.

        NCBI fasta_cds_na headers look like:
          >lcl|WP_123456.1_cds_NZ_ABC123.1_1 [protein=hypothetical protein] [protein_id=WP_123456.1] ...
          >lcl|ABC12345.1_cds_1 [protein_id=ABC12345.1] ...
        """
        # Try [protein_id=...] first — most reliable
        for part in header.split("["):
            if part.startswith("protein_id="):
                raw_accession = part.split("]")[0][len("protein_id="):].strip()
                canonical = self._canonical_accession(raw_accession)
                if canonical:
                    return canonical

        # Fallback: parse lcl|<accession>_cds_
        first_token = header.split()[0] if header else ""
        if first_token.startswith("lcl|"):
            body = first_token[4:]
            cds_pos = body.find("_cds_")
            if cds_pos == -1:
                cds_pos = body.find("_cds")
            if cds_pos > 0:
                raw_accession = body[:cds_pos]
                return self._canonical_accession(raw_accession)

        return ""

    def _fetch_fasta_entries(self, accessions: tuple[str, ...]):
        params = self._base_params()
        params.update(
            {
                "db": "protein",
                "id": ",".join(accessions),
                "rettype": "fasta",
                "retmode": "text",
            }
        )
        response_text = self._transport.get_text(self._EFETCH_URL, params=params)
        return parse_fasta(response_text)

    def _base_params(self) -> dict[str, object]:
        params: dict[str, object] = {"tool": self._tool_name}
        if self._email:
            params["email"] = self._email
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    def _build_metadata(self, docsum: dict[str, object], extra_fields: tuple[str, ...]) -> dict[str, str]:
        metadata: dict[str, str] = {}
        field_names = dict.fromkeys([*self._DEFAULT_METADATA_FIELDS, *extra_fields])
        for field_name in field_names:
            raw_value = docsum.get(field_name)
            value = self._stringify_metadata_value(raw_value)
            if value:
                metadata[field_name] = value
        return metadata

    def _parse_esearch_response(self, response_text: str) -> tuple[list[str], int | None]:
        if not response_text or not response_text.strip():
            logger.warning("NCBI esearch returned empty response body")
            return [], None
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("NCBI esearch returned non-JSON response: %s", response_text[:500])
            return [], None

        result = payload.get("esearchresult", {})

        error_list = result.get("errorlist", {})
        phrase_errors = error_list.get("phrasesnotfound", []) if isinstance(error_list, dict) else []
        if phrase_errors:
            logger.warning("NCBI esearch phrase errors: %s", phrase_errors)

        api_error = result.get("error")
        if api_error:
            raise DataNotFoundError(f"NCBI esearch API error: {api_error}")

        raw_ids = result.get("idlist", [])
        total_count = self._int_value(result.get("count"))
        accessions = [str(value).strip() for value in raw_ids if str(value).strip()]
        return accessions, total_count

    def _search_page_size(self, effective_limit: int | None, current_count: int) -> int:
        page_size = self._DEFAULT_SEARCH_PAGE_SIZE
        if effective_limit is None:
            return min(page_size, self._MAX_ESEARCH_PAGE_SIZE)

        remaining = effective_limit - current_count
        if remaining <= 0:
            return 0
        return min(remaining, page_size, self._MAX_ESEARCH_PAGE_SIZE)

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

    def _int_value(self, raw_value: object) -> int | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, int):
            return raw_value
        value = str(raw_value).strip()
        if value.isdigit():
            return int(value)
        return None

    def _string_value(self, raw_value: object) -> str | None:
        if raw_value is None:
            return None
        value = str(raw_value).strip()
        return value or None

    def _stringify_metadata_value(self, raw_value: object) -> str | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, dict):
            if "value" in raw_value:
                return self._string_value(raw_value.get("value"))
            return json.dumps(raw_value, ensure_ascii=False, sort_keys=True)
        if isinstance(raw_value, list):
            if not raw_value:
                return None
            return json.dumps(raw_value, ensure_ascii=False)
        return self._string_value(raw_value)

    def _clean_optional_value(self, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    def _canonical_accession(self, accession: str) -> str:
        value = accession.strip()
        prefix, separator, suffix = value.rpartition(".")
        if separator and suffix.isdigit() and prefix:
            return prefix
        return value

"""UniProt protein source — a globally reachable alternative to NCBI.

NCBI E-utilities can be slow or unreliable from some regions (e.g. Vietnam);
UniProtKB (EBI/SIB, hosted in Europe) exposes a clean keyword REST search that
returns protein FASTA directly, with no API key. Records returned here expose
the same attributes the semantic ranker and span builder expect:
``accession``, ``description``, ``organism``, ``sequence`` and ``metadata``.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

# Keys that appear after the protein name in a UniProtKB FASTA header.
_HEADER_FIELD_KEYS = ("OS", "OX", "GN", "PE", "SV")
_HEADER_FIELD_RE = re.compile(r"\b(OS|OX|GN|PE|SV)=")


@dataclass(frozen=True)
class UniProtRecord:
    accession: str
    description: str
    organism: str
    sequence: str
    metadata: dict[str, str] = field(default_factory=dict)
    sequence_length: int = 0


def fetch_uniprot_records(
    query: str,
    limit: int = 5,
    *,
    timeout: float = 30.0,
    max_retries: int = 3,
    opener: Any | None = None,
) -> list[UniProtRecord]:
    """Search UniProtKB by free-text keywords and return protein records.

    ``opener`` lets callers inject a urllib opener (e.g. with a custom SSL
    context) for testing or constrained networks; production uses the default.
    """
    clean_query = " ".join(str(query or "").split())
    if not clean_query:
        raise ValueError("UniProt query must not be empty.")
    params = urllib.parse.urlencode(
        {"query": clean_query, "format": "fasta", "size": max(1, int(limit))}
    )
    url = f"{UNIPROT_SEARCH_URL}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "MDNAC/0.2 (protein-span)"})

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            do_open = opener.open if opener is not None else urllib.request.urlopen
            with do_open(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                body = response.read().decode(charset, "replace")
            return parse_uniprot_fasta(body)
        except Exception as exc:  # network hiccups / upstream flaps: retry with backoff
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(min(2.0 ** attempt, 8.0))

    # rest.uniprot.org intermittently serves HTTP 5xx "Service Unavailable" (and
    # 429 when throttling) for tens of seconds before recovering. Surface that as
    # an obviously transient upstream outage so it isn't mistaken for a bad query
    # or a bug in this code.
    status = getattr(last_error, "code", None)
    if isinstance(last_error, urllib.error.HTTPError) and status is not None and (status >= 500 or status == 429):
        raise RuntimeError(
            f"UniProt is temporarily unavailable (HTTP {status}). This is an upstream outage "
            f"at rest.uniprot.org, not your query -- it usually recovers within a minute. "
            f"Retry shortly, or switch source to 'ena'."
        ) from last_error
    raise RuntimeError(f"UniProt search failed: {last_error}") from last_error


def parse_uniprot_fasta(text: str) -> list[UniProtRecord]:
    records: list[UniProtRecord] = []
    header: str | None = None
    seq_parts: list[str] = []

    def flush() -> None:
        if header is None:
            return
        sequence = "".join(seq_parts)
        if sequence:
            records.append(_record_from_header(header, sequence))

    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            header = line[1:].strip()
            seq_parts = []
        elif header is not None:
            seq_parts.append(line.strip())
    flush()
    return records


def _record_from_header(header: str, sequence: str) -> UniProtRecord:
    # Header form: db|ACCESSION|ENTRYNAME Protein name OS=.. OX=.. GN=.. PE=.. SV=..
    accession = ""
    remainder = header
    if header.startswith(("sp|", "tr|")) or header.count("|") >= 2:
        parts = header.split("|", 2)
        if len(parts) == 3:
            accession = parts[1].strip()
            # parts[2] is "ENTRYNAME Protein name OS=..."; drop the entry name token.
            _entry_name, _, remainder = parts[2].strip().partition(" ")
            remainder = remainder.strip() or parts[2].strip()
    fields = _parse_header_fields(remainder)
    protein_name = fields.pop("_name", "").strip()
    organism = fields.get("OS", "").strip()
    gene = fields.get("GN", "").strip()
    if not accession:
        accession = (remainder.split() or [header])[0]

    metadata: dict[str, str] = {}
    if gene:
        metadata["gene"] = gene
    if protein_name:
        metadata["product"] = protein_name
    if fields.get("OX"):
        metadata["taxid"] = fields["OX"].strip()
    metadata["source"] = "uniprot"

    return UniProtRecord(
        accession=accession,
        description=protein_name or remainder,
        organism=organism,
        sequence="".join(sequence.split()).upper(),
        metadata=metadata,
        sequence_length=len(sequence),
    )


def _parse_header_fields(remainder: str) -> dict[str, str]:
    """Split a UniProt header tail into the protein name and OS/OX/GN/... fields."""
    match = _HEADER_FIELD_RE.search(remainder)
    name = remainder[: match.start()].strip() if match else remainder.strip()
    fields: dict[str, str] = {"_name": name}
    if not match:
        return fields

    tail = remainder[match.start() :]
    # Find each "KEY=" boundary and slice the value up to the next key.
    boundaries = [(m.group(1), m.start(), m.end()) for m in _HEADER_FIELD_RE.finditer(tail)]
    for index, (key, _start, value_start) in enumerate(boundaries):
        value_end = boundaries[index + 1][1] if index + 1 < len(boundaries) else len(tail)
        fields[key] = tail[value_start:value_end].strip()
    return fields


__all__ = ["UniProtRecord", "fetch_uniprot_records", "parse_uniprot_fasta"]

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256

from libs.data.entities import SequenceRecord


def clean_optional_string(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def canonical_accession(accession: str | None) -> str:
    value = clean_optional_string(accession)
    if value is None:
        return ""
    prefix, separator, suffix = value.rpartition(".")
    if separator and suffix.isdigit():
        return prefix
    return value


def version_from_accession_token(accession: str) -> str | None:
    value = clean_optional_string(accession)
    if value is None:
        return None
    prefix, separator, suffix = value.rpartition(".")
    if separator and suffix.isdigit() and prefix:
        return value
    return None


def accession_key_for_record(record: SequenceRecord) -> str:
    candidates = [record.accession, record.sequence_version]
    for candidate in candidates:
        accession = canonical_accession(candidate)
        if accession:
            return accession
    raise ValueError("Record does not contain an accession that can be indexed")


def accession_hash(accessions: tuple[str, ...]) -> str:
    return sha256("\n".join(accessions).encode("utf-8")).hexdigest()


def sequence_hash(sequence: str) -> str:
    return sha256(sequence.encode("utf-8")).hexdigest()


def chunked(items: tuple[str, ...], batch_size: int) -> list[tuple[str, ...]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def has_raw_index_entry(entry: dict[str, object] | None) -> bool:
    if not isinstance(entry, dict):
        return False
    sequence = str(entry.get("sequence", ""))
    return bool(sequence.strip())


def sequence_type_from_text(train_text: str) -> str:
    if "<|dna|>" in train_text or "<|rna|>" in train_text:
        raise ValueError("Only protein training text is supported.")
    return "protein"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bump_reason(dropped_reasons: dict[str, int], reason: str) -> None:
    dropped_reasons[reason] = dropped_reasons.get(reason, 0) + 1

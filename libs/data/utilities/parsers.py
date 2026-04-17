from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Iterable, Iterator
from urllib.parse import unquote


@dataclass(slots=True, frozen=True)
class ParsedFastaEntry:
    accession: str
    header: str
    sequence: str


@dataclass(slots=True, frozen=True)
class ParsedFlatfileEntry:
    accession: str
    description: str
    organism: str
    version: str | None


@dataclass(slots=True, frozen=True)
class ParsedEnaCodingEntry:
    accession: str
    description: str
    organism: str
    protein_id: str | None
    parent_accession: str | None
    translation: str
    sequence_version: str | None


@dataclass(slots=True, frozen=True)
class ParsedAnnotationFeature:
    sequence_id: str
    feature_type: str
    start: int
    end: int
    strand: str
    qualifiers: dict[str, tuple[str, ...]]
    segments: tuple[tuple[int, int], ...]


@dataclass(slots=True, frozen=True)
class ParsedGenbankRecord:
    accession: str
    organism: str
    sequence: str
    features: tuple[ParsedAnnotationFeature, ...]


def parse_csv_rows(raw_text: str, delimiter: str = ",") -> list[dict[str, str]]:
    text = raw_text.strip()
    if not text:
        return []

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows: list[dict[str, str]] = []
    for row in reader:
        rows.append({key: (value or "").strip() for key, value in row.items() if key})
    return rows


def parse_tsv_rows(raw_text: str) -> list[dict[str, str]]:
    return parse_csv_rows(raw_text, delimiter="\t")


def parse_fasta(raw_text: str) -> list[ParsedFastaEntry]:
    entries: list[ParsedFastaEntry] = []
    header: str | None = None
    sequence_chunks: list[str] = []

    def flush_current() -> None:
        nonlocal header, sequence_chunks
        if header is None:
            return
        accession = header.split()[0]
        sequence = "".join(sequence_chunks).strip()
        if accession and sequence:
            entries.append(ParsedFastaEntry(accession=accession, header=header, sequence=sequence))
        header = None
        sequence_chunks = []

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush_current()
            header = line[1:].strip()
            continue
        sequence_chunks.append(line)

    flush_current()
    return entries


def iter_fasta_entries(lines: Iterable[str]) -> Iterator[ParsedFastaEntry]:
    header: str | None = None
    sequence_chunks: list[str] = []

    def flush_current() -> ParsedFastaEntry | None:
        nonlocal header, sequence_chunks
        if header is None:
            return None

        accession = header.split()[0]
        sequence = "".join(sequence_chunks).strip()
        current_header = header
        header = None
        sequence_chunks = []

        if accession and sequence:
            return ParsedFastaEntry(accession=accession, header=current_header, sequence=sequence)
        return None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            entry = flush_current()
            if entry is not None:
                yield entry
            header = line[1:].strip()
            continue
        sequence_chunks.append(line)

    entry = flush_current()
    if entry is not None:
        yield entry


def parse_ddbj_flatfile(raw_text: str) -> dict[str, ParsedFlatfileEntry]:
    normalized = raw_text.replace("\r\n", "\n")
    blocks = [block.strip() for block in re.split(r"\n//\s*", normalized) if block.strip()]
    parsed: dict[str, ParsedFlatfileEntry] = {}

    for block in blocks:
        fields: dict[str, list[str]] = {}
        current_field: str | None = None

        for line in block.splitlines():
            if not line.strip():
                continue

            if line.startswith("  ORGANISM"):
                value = line[12:].strip()
                fields.setdefault("ORGANISM", []).append(value)
                current_field = "ORGANISM"
                continue

            field_name = line[:12].strip()
            if field_name and not line.startswith(" "):
                value = line[12:].strip()
                fields.setdefault(field_name, []).append(value)
                current_field = field_name
                continue

            if current_field in {"DEFINITION", "ACCESSION", "VERSION", "SOURCE", "ORGANISM"}:
                value = line[12:].strip()
                fields.setdefault(current_field, []).append(value)

        accession = " ".join(fields.get("ACCESSION", [])).split()
        if not accession:
            continue

        accession_value = accession[0]
        description = " ".join(fields.get("DEFINITION", [])).strip()
        organism = " ".join(fields.get("ORGANISM", [])).strip() or " ".join(fields.get("SOURCE", [])).strip()
        version_tokens = " ".join(fields.get("VERSION", [])).split()
        version = version_tokens[0] if version_tokens else None

        parsed[accession_value] = ParsedFlatfileEntry(
            accession=accession_value,
            description=description,
            organism=organism,
            version=version,
        )

    return parsed


def parse_ena_coding_embl(raw_text: str) -> list[ParsedEnaCodingEntry]:
    normalized = raw_text.replace("\r\n", "\n")
    blocks = [block.strip() for block in re.split(r"\n//\s*", normalized) if block.strip()]
    entries: list[ParsedEnaCodingEntry] = []

    for block in blocks:
        accession = ""
        description_lines: list[str] = []
        organism_lines: list[str] = []
        feature_lines: list[str] = []
        parent_accession: str | None = None
        sequence_version: str | None = None

        for line in block.splitlines():
            if line.startswith("ID"):
                payload = line[2:].strip()
                parts = [part.strip() for part in payload.split(";") if part.strip()]
                if parts:
                    accession = parts[0]
                for part in parts[1:]:
                    if not part.startswith("SV "):
                        continue
                    version_suffix = part[3:].strip()
                    if accession and version_suffix.isdigit():
                        sequence_version = f"{accession}.{version_suffix}"
                    elif version_suffix:
                        sequence_version = version_suffix
                continue

            if line.startswith("DE"):
                description_lines.append(line[2:].strip())
                continue

            if line.startswith("OS"):
                organism_lines.append(line[2:].strip())
                continue

            if line.startswith("PA"):
                parent_value = line[2:].strip()
                if parent_value:
                    parent_accession = parent_value
                continue

            if line.startswith("FT"):
                feature_lines.append(line[21:].rstrip() if len(line) > 21 else "")

        if not accession:
            continue

        feature_text = "\n".join(feature_lines)
        translation_match = re.search(r'/translation="([^"]+)"', feature_text, re.S)
        if translation_match is None:
            continue

        translation = re.sub(r"[^A-Za-z*]", "", translation_match.group(1)).upper()
        if not translation:
            continue

        protein_id_match = re.search(r'/protein_id="([^"]+)"', feature_text)
        protein_id = protein_id_match.group(1).strip() if protein_id_match else None

        entries.append(
            ParsedEnaCodingEntry(
                accession=accession,
                description=" ".join(description_lines).strip(),
                organism=" ".join(organism_lines).strip(),
                protein_id=protein_id,
                parent_accession=parent_accession,
                translation=translation,
                sequence_version=sequence_version,
            )
        )

    return entries


def parse_gff_features(raw_text: str) -> list[ParsedAnnotationFeature]:
    features: list[ParsedAnnotationFeature] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = raw_line.rstrip("\n").split("\t")
        if len(parts) != 9:
            continue

        seqid, source, feature_type, start, end, score, strand, phase, attributes = parts
        del source, score, phase
        if not start.isdigit() or not end.isdigit():
            continue

        qualifiers = _parse_gff_attributes(attributes)
        start_int = int(start)
        end_int = int(end)
        features.append(
            ParsedAnnotationFeature(
                sequence_id=seqid.strip(),
                feature_type=feature_type.strip(),
                start=start_int,
                end=end_int,
                strand=strand.strip() or "+",
                qualifiers=qualifiers,
                segments=((start_int, end_int),),
            )
        )

    return features


def parse_genbank_records(raw_text: str) -> list[ParsedGenbankRecord]:
    normalized = raw_text.replace("\r\n", "\n")
    blocks = [block.strip() for block in re.split(r"\n//\s*", normalized) if block.strip()]
    records: list[ParsedGenbankRecord] = []

    for block in blocks:
        accession = _extract_single_value(block, "ACCESSION") or _extract_single_value(block, "VERSION")
        if accession is None:
            continue
        accession = accession.split()[0]

        organism = _extract_single_value(block, "ORGANISM") or _extract_single_value(block, "SOURCE") or ""
        sequence = _extract_origin_sequence(block)
        features = tuple(_parse_genbank_features(block, accession))
        records.append(
            ParsedGenbankRecord(
                accession=accession,
                organism=organism.strip(),
                sequence=sequence,
                features=features,
            )
        )

    return records


def _parse_gff_attributes(raw_attributes: str) -> dict[str, tuple[str, ...]]:
    parsed: dict[str, tuple[str, ...]] = {}
    for part in raw_attributes.split(";"):
        entry = part.strip()
        if not entry:
            continue
        if "=" in entry:
            key, value = entry.split("=", 1)
            values = tuple(unquote(item.strip()) for item in value.split(",") if item.strip())
            parsed[key.strip()] = values or ("",)
            continue
        parsed[entry] = ("true",)
    return parsed


def _extract_single_value(block: str, field_name: str) -> str | None:
    for line in block.splitlines():
        if line.startswith("  " + field_name):
            return line[12:].strip()
        if line.startswith(field_name):
            return line[12:].strip()
    return None


def _extract_origin_sequence(block: str) -> str:
    if "ORIGIN" not in block:
        return ""
    origin = block.split("ORIGIN", 1)[1]
    lines = []
    for raw_line in origin.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        letters_only = re.sub(r"[^A-Za-z]", "", line)
        if letters_only:
            lines.append(letters_only.upper())
    return "".join(lines)


def _parse_genbank_features(block: str, accession: str) -> list[ParsedAnnotationFeature]:
    if "FEATURES" not in block:
        return []

    feature_section = block.split("FEATURES", 1)[1]
    if "ORIGIN" in feature_section:
        feature_section = feature_section.split("ORIGIN", 1)[0]

    features: list[ParsedAnnotationFeature] = []
    current_feature: dict[str, object] | None = None
    current_qualifier: str | None = None

    for raw_line in feature_section.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        if len(line) >= 21 and line.startswith("     ") and line[5:21].strip():
            feature_type = line[5:21].strip()
            location = line[21:].strip()
            strand, segments = _parse_location_segments(location)
            if not segments:
                current_feature = None
                current_qualifier = None
                continue
            current_feature = {
                "sequence_id": accession,
                "feature_type": feature_type,
                "start": min(segment[0] for segment in segments),
                "end": max(segment[1] for segment in segments),
                "strand": strand,
                "qualifiers": {},
                "segments": tuple(segments),
            }
            features.append(
                ParsedAnnotationFeature(
                    sequence_id=accession,
                    feature_type=feature_type,
                    start=int(current_feature["start"]),
                    end=int(current_feature["end"]),
                    strand=strand,
                    qualifiers={},
                    segments=tuple(segments),
                )
            )
            current_qualifier = None
            continue

        if current_feature is None or not features:
            continue

        if line.startswith("                     /"):
            qualifier_payload = line[21:].strip()
            qualifier_name, qualifier_value = _parse_genbank_qualifier(qualifier_payload)
            feature = features[-1]
            updated_qualifiers = {key: tuple(values) for key, values in feature.qualifiers.items()}
            updated_qualifiers.setdefault(qualifier_name, ())
            updated_qualifiers[qualifier_name] = (*updated_qualifiers[qualifier_name], qualifier_value)
            features[-1] = ParsedAnnotationFeature(
                sequence_id=feature.sequence_id,
                feature_type=feature.feature_type,
                start=feature.start,
                end=feature.end,
                strand=feature.strand,
                qualifiers=updated_qualifiers,
                segments=feature.segments,
            )
            current_qualifier = qualifier_name
            continue

        if line.startswith("                     ") and current_qualifier is not None:
            continuation = line[21:].strip().strip('"')
            if not continuation:
                continue
            feature = features[-1]
            updated_qualifiers = {key: tuple(values) for key, values in feature.qualifiers.items()}
            existing_values = list(updated_qualifiers.get(current_qualifier, ()))
            if not existing_values:
                existing_values = [continuation]
            else:
                existing_values[-1] = f"{existing_values[-1]} {continuation}".strip()
            updated_qualifiers[current_qualifier] = tuple(existing_values)
            features[-1] = ParsedAnnotationFeature(
                sequence_id=feature.sequence_id,
                feature_type=feature.feature_type,
                start=feature.start,
                end=feature.end,
                strand=feature.strand,
                qualifiers=updated_qualifiers,
                segments=feature.segments,
            )

    return features


def _parse_genbank_qualifier(raw_value: str) -> tuple[str, str]:
    payload = raw_value.lstrip("/")
    if "=" not in payload:
        return payload.strip(), "true"
    key, value = payload.split("=", 1)
    return key.strip(), value.strip().strip('"')


def _parse_location_segments(raw_location: str) -> tuple[str, list[tuple[int, int]]]:
    location = raw_location.strip()
    strand = "+"

    while True:
        if location.startswith("complement(") and location.endswith(")"):
            strand = "-"
            location = location[len("complement(") : -1].strip()
            continue
        if location.startswith("join(") and location.endswith(")"):
            location = location[len("join(") : -1].strip()
            continue
        if location.startswith("order(") and location.endswith(")"):
            location = location[len("order(") : -1].strip()
            continue
        break

    segments: list[tuple[int, int]] = []
    for part in location.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        match = re.match(r"[<>]?(\d+)\.\.[<>]?(\d+)$", chunk)
        if match:
            segments.append((int(match.group(1)), int(match.group(2))))
            continue
        single_match = re.match(r"[<>]?(\d+)$", chunk)
        if single_match:
            position = int(single_match.group(1))
            segments.append((position, position))

    return strand, segments

from __future__ import annotations

import re
from typing import Iterable, Sequence

from libs.data.training.raw_pipeline.models import ProfileKeywordRule
from libs.data.utilities.parsers import ParsedAnnotationFeature


def match_rules(
    annotation_text: str,
    rules: Sequence[ProfileKeywordRule],
) -> list[tuple[ProfileKeywordRule, tuple[str, ...]]]:
    normalized = _normalize_for_matching(annotation_text)
    matches: list[tuple[ProfileKeywordRule, tuple[str, ...]]] = []
    for rule in rules:
        matched_keywords = tuple(
            keyword
            for keyword in rule.keywords
            if _normalize_for_matching(keyword).strip() and _normalize_for_matching(keyword) in normalized
        )
        if matched_keywords:
            matches.append((rule, matched_keywords))
    return matches


def feature_text(feature: ParsedAnnotationFeature) -> str:
    ordered_keys = (
        "name",
        "gene",
        "gene_synonym",
        "product",
        "product_synonym",
        "prot_desc",
        "note",
        "description",
        "function",
        "functional_annotation",
        "pathway",
        "ec_number",
        "go_process",
        "go_function",
        "go_component",
        "ontology_term",
        "experiment",
        "inference",
        "ncrna_class",
    )
    lowered = {key.lower(): tuple(values) for key, values in feature.qualifiers.items()}
    values: list[str] = [feature.feature_type]
    for key in ordered_keys:
        values.extend(_clean_qualifier_values(key, lowered.get(key, ())))
    if not values:
        return ""
    return " ".join(value.strip() for value in values if value and value.strip())


def extract_feature_sequence(
    feature: ParsedAnnotationFeature,
    full_sequence: str,
    sequence_type: str,
) -> str:
    fragments: list[str] = []
    for start, end in feature.segments:
        if start <= 0 or end <= 0 or start > end:
            continue
        fragments.append(full_sequence[start - 1 : end])
    sequence = "".join(fragments)
    if feature.strand == "-":
        sequence = reverse_complement(sequence, sequence_type=sequence_type)
    return normalize_sequence(sequence, sequence_type=sequence_type)


def reverse_complement(sequence: str, sequence_type: str) -> str:
    normalized = normalize_sequence(sequence, sequence_type=sequence_type)
    if sequence_type == "rna":
        table = str.maketrans("ACGUN", "UGCAN")
    else:
        table = str.maketrans("ACGTN", "TGCAN")
    return normalized.translate(table)[::-1]


def normalize_sequence(sequence: str, sequence_type: str) -> str:
    compact = "".join(character for character in sequence.upper() if not character.isspace())
    if sequence_type == "rna":
        compact = compact.replace("T", "U")
        valid = {"A", "C", "G", "U", "N"}
    else:
        compact = compact.replace("U", "T")
        valid = {"A", "C", "G", "T", "N"}

    normalized: list[str] = []
    for base in compact:
        normalized.append(base if base in valid else "N")
    return "".join(normalized)


def slugify(value: str) -> str:
    stripped = value.strip().lower()
    if not stripped:
        raise ValueError("dataset_name must not be empty")

    parts: list[str] = []
    last_was_separator = False
    for character in stripped:
        if character.isalnum():
            parts.append(character)
            last_was_separator = False
            continue
        if not last_was_separator:
            parts.append("-")
            last_was_separator = True

    slug = "".join(parts).strip("-")
    if not slug:
        raise ValueError("dataset_name must contain at least one alphanumeric character")
    return slug


def infer_organism_from_headers(headers: Iterable[str]) -> str | None:
    for header in headers:
        parts = header.split()
        if len(parts) >= 3 and parts[1][:1].isupper() and parts[2][:1].islower():
            return f"{parts[1]} {parts[2]}"
    return None


def _normalize_for_matching(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    collapsed = re.sub(r"\s+", " ", normalized).strip()
    return f" {collapsed} " if collapsed else ""


def _clean_qualifier_values(key: str, values: tuple[str, ...]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for raw_value in values:
        value = raw_value.strip()
        if not value:
            continue
        if key in {"go_process", "go_function", "go_component"}:
            value = _go_label(value)
        cleaned.append(value)
    return tuple(cleaned)


def _go_label(value: str) -> str:
    if "|" in value:
        label, _, _ = value.partition("|")
        return label.strip()
    if " - " in value and value.upper().startswith("GO:"):
        _, _, label = value.partition(" - ")
        return label.strip()
    return value.strip()

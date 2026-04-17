from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from libs.data.training.raw_pipeline.defaults import DEFAULT_KEYWORD_RULES
from libs.data.training.raw_pipeline.helpers import (
    extract_feature_sequence,
    feature_text,
    infer_organism_from_headers,
    match_rules,
)
from libs.data.training.raw_pipeline.models import ProfileKeywordRule
from libs.data.utilities.parsers import ParsedAnnotationFeature, parse_fasta, parse_genbank_records, parse_gff_features


@dataclass(slots=True, frozen=True)
class RawLabelAuditRow:
    accession: str
    organism: str
    feature_type: str
    start: int
    end: int
    strand: str
    annotation_text: str
    sequence_fragment_length: int
    matched_labels: tuple[str, ...]
    matched_keywords: tuple[str, ...]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RawLabelAuditSummary:
    total_features: int
    eligible_feature_count: int
    matched_feature_count: int
    unmatched_feature_count: int
    multi_label_feature_count: int
    missing_annotation_count: int
    missing_sequence_count: int
    empty_sequence_fragment_count: int
    label_counts: dict[str, int]
    keyword_counts: dict[str, int]
    feature_type_counts: dict[str, int]
    unmatched_feature_type_counts: dict[str, int]
    top_unmatched_annotations: tuple[tuple[str, int], ...]
    top_multi_label_annotations: tuple[tuple[str, int], ...]

    @property
    def match_rate(self) -> float:
        if self.eligible_feature_count == 0:
            return 0.0
        return self.matched_feature_count / self.eligible_feature_count

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["match_rate"] = self.match_rate
        return payload


@dataclass(slots=True, frozen=True)
class RawLabelAuditReport:
    summary: RawLabelAuditSummary
    rows: tuple[RawLabelAuditRow, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
        }

    def write_json(self, path: Path | str) -> str:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return str(resolved)


def audit_fasta_and_gff(
    fasta_path: Path | str,
    annotation_path: Path | str,
    organism: str | None = None,
    sequence_type: str = "dna",
    keyword_rules: Sequence[ProfileKeywordRule] | None = None,
    top_k_examples: int = 20,
) -> RawLabelAuditReport:
    fasta_entries = parse_fasta(Path(fasta_path).read_text(encoding="utf-8"))
    sequences_by_id = {entry.accession: entry.sequence for entry in fasta_entries}
    headers_by_id = {entry.accession: entry.header for entry in fasta_entries}
    features = parse_gff_features(Path(annotation_path).read_text(encoding="utf-8"))

    inferred_organism = organism or infer_organism_from_headers(headers_by_id.values())
    organism_by_id = (
        {feature.sequence_id: inferred_organism for feature in features if inferred_organism}
        if inferred_organism
        else {}
    )
    return _audit_features(
        features=features,
        sequences_by_id=sequences_by_id,
        organism_by_id=organism_by_id,
        sequence_type=sequence_type,
        keyword_rules=keyword_rules,
        top_k_examples=top_k_examples,
    )


def audit_genbank(
    genbank_path: Path | str,
    sequence_type: str = "dna",
    keyword_rules: Sequence[ProfileKeywordRule] | None = None,
    top_k_examples: int = 20,
) -> RawLabelAuditReport:
    records = parse_genbank_records(Path(genbank_path).read_text(encoding="utf-8"))
    features: list[ParsedAnnotationFeature] = []
    sequences_by_id: dict[str, str] = {}
    organism_by_id: dict[str, str] = {}
    for record in records:
        sequences_by_id[record.accession] = record.sequence
        if record.organism:
            organism_by_id[record.accession] = record.organism
        features.extend(record.features)

    return _audit_features(
        features=features,
        sequences_by_id=sequences_by_id,
        organism_by_id=organism_by_id,
        sequence_type=sequence_type,
        keyword_rules=keyword_rules,
        top_k_examples=top_k_examples,
    )


def _audit_features(
    features: Iterable[ParsedAnnotationFeature],
    sequences_by_id: dict[str, str],
    organism_by_id: dict[str, str],
    sequence_type: str,
    keyword_rules: Sequence[ProfileKeywordRule] | None,
    top_k_examples: int,
) -> RawLabelAuditReport:
    active_rules = tuple(keyword_rules or DEFAULT_KEYWORD_RULES)
    rows: list[RawLabelAuditRow] = []

    label_counts: Counter[str] = Counter()
    keyword_counts: Counter[str] = Counter()
    feature_type_counts: Counter[str] = Counter()
    unmatched_feature_type_counts: Counter[str] = Counter()
    unmatched_annotations: Counter[str] = Counter()
    multi_label_annotations: Counter[str] = Counter()

    missing_annotation_count = 0
    missing_sequence_count = 0
    empty_sequence_fragment_count = 0
    eligible_feature_count = 0
    matched_feature_count = 0
    unmatched_feature_count = 0
    multi_label_feature_count = 0

    for feature in features:
        feature_type_counts[feature.feature_type] += 1
        organism = organism_by_id.get(feature.sequence_id, "")
        annotation_text = feature_text(feature)
        if not annotation_text:
            missing_annotation_count += 1
            rows.append(
                RawLabelAuditRow(
                    accession=feature.sequence_id,
                    organism=organism,
                    feature_type=feature.feature_type,
                    start=feature.start,
                    end=feature.end,
                    strand=feature.strand,
                    annotation_text="",
                    sequence_fragment_length=0,
                    matched_labels=(),
                    matched_keywords=(),
                    status="missing_annotation",
                )
            )
            continue

        sequence = sequences_by_id.get(feature.sequence_id)
        if not sequence:
            missing_sequence_count += 1
            rows.append(
                RawLabelAuditRow(
                    accession=feature.sequence_id,
                    organism=organism,
                    feature_type=feature.feature_type,
                    start=feature.start,
                    end=feature.end,
                    strand=feature.strand,
                    annotation_text=annotation_text,
                    sequence_fragment_length=0,
                    matched_labels=(),
                    matched_keywords=(),
                    status="missing_sequence",
                )
            )
            continue

        sequence_fragment = extract_feature_sequence(feature, sequence, sequence_type)
        if not sequence_fragment:
            empty_sequence_fragment_count += 1
            rows.append(
                RawLabelAuditRow(
                    accession=feature.sequence_id,
                    organism=organism,
                    feature_type=feature.feature_type,
                    start=feature.start,
                    end=feature.end,
                    strand=feature.strand,
                    annotation_text=annotation_text,
                    sequence_fragment_length=0,
                    matched_labels=(),
                    matched_keywords=(),
                    status="empty_sequence_fragment",
                )
            )
            continue

        eligible_feature_count += 1
        matched_rules = match_rules(annotation_text, active_rules)
        if not matched_rules:
            unmatched_feature_count += 1
            unmatched_feature_type_counts[feature.feature_type] += 1
            unmatched_annotations[annotation_text] += 1
            rows.append(
                RawLabelAuditRow(
                    accession=feature.sequence_id,
                    organism=organism,
                    feature_type=feature.feature_type,
                    start=feature.start,
                    end=feature.end,
                    strand=feature.strand,
                    annotation_text=annotation_text,
                    sequence_fragment_length=len(sequence_fragment),
                    matched_labels=(),
                    matched_keywords=(),
                    status="unmatched",
                )
            )
            continue

        matched_feature_count += 1
        matched_labels = tuple(rule.label for rule, _ in matched_rules)
        matched_keywords = tuple(
            keyword
            for _, keywords in matched_rules
            for keyword in keywords
        )
        if len(matched_labels) > 1:
            multi_label_feature_count += 1
            multi_label_annotations[annotation_text] += 1

        label_counts.update(matched_labels)
        keyword_counts.update(matched_keywords)
        rows.append(
            RawLabelAuditRow(
                accession=feature.sequence_id,
                organism=organism,
                feature_type=feature.feature_type,
                start=feature.start,
                end=feature.end,
                strand=feature.strand,
                annotation_text=annotation_text,
                sequence_fragment_length=len(sequence_fragment),
                matched_labels=matched_labels,
                matched_keywords=matched_keywords,
                status="matched",
            )
        )

    summary = RawLabelAuditSummary(
        total_features=len(rows),
        eligible_feature_count=eligible_feature_count,
        matched_feature_count=matched_feature_count,
        unmatched_feature_count=unmatched_feature_count,
        multi_label_feature_count=multi_label_feature_count,
        missing_annotation_count=missing_annotation_count,
        missing_sequence_count=missing_sequence_count,
        empty_sequence_fragment_count=empty_sequence_fragment_count,
        label_counts=dict(label_counts.most_common()),
        keyword_counts=dict(keyword_counts.most_common()),
        feature_type_counts=dict(feature_type_counts.most_common()),
        unmatched_feature_type_counts=dict(unmatched_feature_type_counts.most_common()),
        top_unmatched_annotations=tuple(unmatched_annotations.most_common(top_k_examples)),
        top_multi_label_annotations=tuple(multi_label_annotations.most_common(top_k_examples)),
    )
    return RawLabelAuditReport(summary=summary, rows=tuple(rows))

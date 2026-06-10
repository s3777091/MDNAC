from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from libs.protein_completion.masking import is_standard_amino_acid_text


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")

SEMANTIC_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "crop": (
        "plant",
        "agriculture",
        "yield",
        "growth",
        "productivity",
        "rhizosphere",
        "phytostimulation",
    ),
    "yield": (
        "crop",
        "plant",
        "growth",
        "productivity",
        "biomass",
        "stress",
        "tolerance",
    ),
    "nang": ("yield", "increase", "growth", "productivity", "biomass"),
    "suat": ("yield", "productivity", "biomass"),
    "cay": ("plant", "crop", "rhizosphere"),
    "trong": ("plant", "crop", "agriculture"),
    "plant": (
        "crop",
        "yield",
        "growth",
        "rhizosphere",
        "nitrogen",
        "phosphate",
        "auxin",
        "iaa",
        "acc",
        "deaminase",
        "siderophore",
        "biocontrol",
    ),
    "growth": (
        "plant",
        "crop",
        "yield",
        "auxin",
        "iaa",
        "gibberellin",
        "cytokinin",
        "nitrogen",
        "phosphate",
    ),
    "nitrogen": ("nif", "nifh", "nitrogenase", "fixation", "diazotroph"),
    "phosphate": ("phosphatase", "phytase", "solubilization", "ppx", "ppk"),
    "drought": ("stress", "tolerance", "osmoprotectant", "trehalose"),
    "salt": ("salinity", "stress", "tolerance", "osmoprotectant"),
}


@dataclass(frozen=True)
class ProteinSemanticMatch:
    record: Any
    score: float
    reasons: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accession": getattr(self.record, "accession", ""),
            "description": getattr(self.record, "description", ""),
            "organism": getattr(self.record, "organism", ""),
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
            "matched_terms": list(self.matched_terms),
            "metadata": dict(getattr(self.record, "metadata", {}) or {}),
        }


@dataclass(frozen=True)
class MaskSpanChoice:
    start: int
    end: int
    score: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_start": self.start,
            "mask_end": self.end,
            "mask_length": self.end - self.start,
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
        }


def rank_protein_records(
    query: str,
    records: Iterable[Any],
    *,
    evidence_texts: Iterable[str] = (),
    min_length: int = 1,
    top_k: int = 5,
) -> list[ProteinSemanticMatch]:
    query_terms = _expanded_terms([query, *evidence_texts])
    matches: list[ProteinSemanticMatch] = []

    for record in records:
        sequence = _compact_sequence(getattr(record, "sequence", ""))
        if len(sequence) < min_length:
            continue

        record_text = _record_text(record)
        record_terms = set(_tokenize(record_text))
        matched_terms = sorted(query_terms & record_terms)
        lexical_score = _overlap_score(query_terms, record_terms)
        phrase_score = _phrase_score(query, record_text)
        quality_score = _sequence_quality_score(sequence)
        metadata_score = _metadata_specificity_score(record)
        score = (
            lexical_score * 0.52
            + phrase_score * 0.18
            + quality_score * 0.18
            + metadata_score * 0.12
        )

        reasons = []
        if matched_terms:
            reasons.append("matched semantic terms in record metadata")
        if phrase_score:
            reasons.append("matched query phrase fragments")
        if quality_score >= 0.95:
            reasons.append("sequence is standard amino-acid rich")
        if metadata_score >= 0.75:
            reasons.append("record has useful biological metadata")

        matches.append(
            ProteinSemanticMatch(
                record=record,
                score=_clamp01(score),
                reasons=reasons or ["ranked by sequence quality and metadata"],
                matched_terms=matched_terms[:20],
            )
        )

    matches.sort(key=lambda item: item.score, reverse=True)
    return matches[:top_k]


def choose_semantic_mask_span(
    sequence: str,
    *,
    mask_length: int,
    requested_start: int | None = None,
    left_flank_size: int = 64,
    right_flank_size: int = 64,
) -> MaskSpanChoice:
    normalized = _compact_sequence(sequence)
    if mask_length <= 0:
        raise ValueError("mask_length must be positive.")
    if len(normalized) < mask_length:
        raise ValueError("sequence is shorter than mask_length.")

    if requested_start is not None:
        end = requested_start + mask_length
        if not 0 <= requested_start < end <= len(normalized):
            raise ValueError("requested mask span is outside the selected protein sequence.")
        span = normalized[requested_start:end]
        if not is_standard_amino_acid_text(span):
            raise ValueError("requested mask span contains non-standard amino acids.")
        return MaskSpanChoice(
            start=requested_start,
            end=end,
            score=1.0,
            reasons=["used user-provided mask_start"],
        )

    best: MaskSpanChoice | None = None
    max_start = len(normalized) - mask_length
    for start in range(max_start + 1):
        end = start + mask_length
        span = normalized[start:end]
        if not is_standard_amino_acid_text(span):
            continue
        if span in f"{normalized[:start]}{normalized[end:]}":
            continue

        left_available = min(left_flank_size, start)
        right_available = min(right_flank_size, len(normalized) - end)
        flank_score = _flank_score(
            left_available=left_available,
            right_available=right_available,
            left_flank_size=left_flank_size,
            right_flank_size=right_flank_size,
        )
        entropy_score = _amino_acid_diversity(span)
        centrality_score = _centrality_score(start, mask_length, len(normalized))
        repeat_penalty = _repeat_penalty(span)
        score = (
            entropy_score * 0.42
            + flank_score * 0.34
            + centrality_score * 0.24
            - repeat_penalty
        )

        candidate = MaskSpanChoice(
            start=start,
            end=end,
            score=_clamp01(score),
            reasons=[
                "auto-selected standard amino-acid span",
                "balanced available left/right flanks",
                "avoids low-complexity sequence where possible",
            ],
        )
        if best is None or candidate.score > best.score:
            best = candidate

    if best is None:
        raise ValueError("No standard amino-acid span is available for the requested mask_length.")
    return best


def _expanded_terms(texts: Iterable[str]) -> set[str]:
    terms: set[str] = set()
    for text in texts:
        for token in _tokenize(text):
            terms.add(token)
            terms.update(SEMANTIC_EXPANSIONS.get(token, ()))
    return {term for term in terms if len(term) > 1}


def _tokenize(value: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(value or "")]


def _record_text(record: Any) -> str:
    metadata = getattr(record, "metadata", {}) or {}
    metadata_text = " ".join(f"{key} {value}" for key, value in dict(metadata).items())
    return " ".join(
        str(value or "")
        for value in (
            getattr(record, "accession", ""),
            getattr(record, "description", ""),
            getattr(record, "organism", ""),
            metadata_text,
        )
    )


def _overlap_score(query_terms: set[str], record_terms: set[str]) -> float:
    if not query_terms or not record_terms:
        return 0.0
    overlap = len(query_terms & record_terms)
    return _clamp01(overlap / math.sqrt(len(query_terms) * len(record_terms)))


def _phrase_score(query: str, record_text: str) -> float:
    query_tokens = _tokenize(query)
    record_normalized = " ".join(_tokenize(record_text))
    if not query_tokens or not record_normalized:
        return 0.0

    hits = 0
    total = 0
    for size in (2, 3):
        for index in range(0, max(0, len(query_tokens) - size + 1)):
            total += 1
            phrase = " ".join(query_tokens[index : index + size])
            if phrase in record_normalized:
                hits += 1
    return 0.0 if total == 0 else _clamp01(hits / total)


def _sequence_quality_score(sequence: str) -> float:
    if not sequence:
        return 0.0
    standard_count = sum(1 for amino_acid in sequence if amino_acid in "ACDEFGHIKLMNPQRSTVWY")
    return standard_count / len(sequence)


def _metadata_specificity_score(record: Any) -> float:
    metadata = getattr(record, "metadata", {}) or {}
    filled = 0
    for field_name in ("gene", "product", "host", "keywords"):
        if str(dict(metadata).get(field_name) or "").strip():
            filled += 1
    if str(getattr(record, "description", "") or "").strip():
        filled += 1
    if str(getattr(record, "organism", "") or "").strip():
        filled += 1
    return filled / 6


def _flank_score(
    *,
    left_available: int,
    right_available: int,
    left_flank_size: int,
    right_flank_size: int,
) -> float:
    left_score = 1.0 if left_flank_size == 0 else left_available / left_flank_size
    right_score = 1.0 if right_flank_size == 0 else right_available / right_flank_size
    return _clamp01((left_score + right_score) / 2)


def _amino_acid_diversity(span: str) -> float:
    if not span:
        return 0.0
    return _clamp01(len(set(span)) / min(20, len(span)))


def _centrality_score(start: int, mask_length: int, sequence_length: int) -> float:
    if sequence_length <= mask_length:
        return 1.0
    span_center = start + mask_length / 2
    sequence_center = sequence_length / 2
    distance = abs(span_center - sequence_center)
    max_distance = sequence_length / 2
    return _clamp01(1.0 - distance / max_distance)


def _repeat_penalty(span: str) -> float:
    if not span:
        return 0.0
    longest = 1
    current = 1
    for previous, current_char in zip(span, span[1:]):
        if current_char == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return min(0.35, longest / max(1, len(span)) * 0.7)


def _compact_sequence(value: Any) -> str:
    return "".join(str(value or "").split()).upper()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "MaskSpanChoice",
    "ProteinSemanticMatch",
    "choose_semantic_mask_span",
    "rank_protein_records",
]

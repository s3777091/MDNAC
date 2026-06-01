"""Candidate generation and validation pipeline for protein sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .scoring import (
    ambiguity_fraction,
    compact_protein_sequence,
    length_window_score,
    valid_amino_acid_fraction,
)
from .types import StructurePrediction, VALID_PROTEIN_AMINO_ACIDS

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np


@dataclass(slots=True)
class GeneratedProteinCandidate:
    profile: str
    sequence: str
    generation_score: float | None = None
    prediction: StructurePrediction | None = None
    validation_score: float = 0.0
    passed: bool = False
    reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CandidateValidationConfig:
    min_length: int = 30
    max_length: int = 1024
    max_x_fraction: float = 0.05
    min_confidence: float | None = None
    min_plddt: float | None = None
    min_ptm: float | None = None
    min_iptm: float | None = None
    min_triangle_consistency: float | None = None
    contact_distance_threshold: float = 8.0


@dataclass(slots=True, frozen=True)
class CandidateValidationResult:
    candidate: GeneratedProteinCandidate
    score: float
    passed: bool
    reasons: tuple[str, ...]
    component_scores: dict[str, float]


def validate_sequence_basic(
    sequence: str,
    config: CandidateValidationConfig,
) -> tuple[dict[str, float], list[str]]:
    """Validate sequence using cheap filters. Returns component scores and failure reasons."""
    normalized = compact_protein_sequence(sequence)
    scores: dict[str, float] = {}
    reasons: list[str] = []

    if not normalized:
        reasons.append("empty_sequence")
        scores["validity"] = 0.0
        scores["length"] = 0.0
        scores["ambiguity"] = 0.0
        return scores, reasons

    validity = valid_amino_acid_fraction(normalized)
    scores["validity"] = validity
    if validity < 1.0:
        reasons.append("invalid_amino_acid")

    length_score = length_window_score(
        normalized, min_length=config.min_length, max_length=config.max_length
    )
    scores["length"] = length_score
    if length_score < 1.0:
        reasons.append("length_outside_window")

    x_fraction = ambiguity_fraction(normalized)
    scores["ambiguity"] = 1.0 - x_fraction
    if x_fraction > config.max_x_fraction:
        reasons.append("too_many_x")

    return scores, reasons


def validate_structure_prediction(
    prediction: StructurePrediction | None,
    config: CandidateValidationConfig,
    *,
    coordinates_loader: Callable[[str], np.ndarray] | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Validate structure prediction against thresholds. Returns component scores and failure reasons."""
    scores: dict[str, float] = {}
    reasons: list[str] = []

    if prediction is None:
        reasons.append("missing_structure_provider")
        scores["model_confidence"] = 0.0
        return scores, reasons

    # Confidence checks
    confidence = _best_confidence(prediction)
    scores["model_confidence"] = confidence

    if config.min_confidence is not None and (prediction.confidence or 0.0) < config.min_confidence:
        reasons.append("low_confidence")

    if config.min_plddt is not None:
        plddt = prediction.plddt or 0.0
        if plddt < config.min_plddt:
            reasons.append("low_plddt")

    if config.min_ptm is not None:
        ptm = prediction.ptm or 0.0
        if ptm < config.min_ptm:
            reasons.append("low_ptm")

    if config.min_iptm is not None:
        iptm = prediction.iptm or 0.0
        if iptm < config.min_iptm:
            reasons.append("low_iptm")

    # Geometry checks if coordinates are available
    if config.min_triangle_consistency is not None and prediction.coordinates_path is not None:
        if coordinates_loader is not None:
            try:
                from .geometry import pairwise_distances, triangle_consistency_score

                coordinates = coordinates_loader(prediction.coordinates_path)
                distances = pairwise_distances(coordinates)
                tri_score = triangle_consistency_score(distances)
                scores["geometry_confidence"] = tri_score
                if tri_score < config.min_triangle_consistency:
                    reasons.append("low_triangle_consistency")
            except Exception:
                reasons.append("geometry_evaluation_failed")
                scores["geometry_confidence"] = 0.0
        else:
            reasons.append("missing_coordinates_loader")
            scores["geometry_confidence"] = 0.0

    return scores, reasons


def validate_generated_candidate(
    candidate: GeneratedProteinCandidate,
    config: CandidateValidationConfig,
    *,
    coordinates_loader: Callable[[str], np.ndarray] | None = None,
) -> GeneratedProteinCandidate:
    """Validate a generated candidate and update its validation state in-place."""
    seq_scores, seq_reasons = validate_sequence_basic(candidate.sequence, config)
    struct_scores, struct_reasons = validate_structure_prediction(
        candidate.prediction, config, coordinates_loader=coordinates_loader
    )

    all_scores = {**seq_scores, **struct_scores}
    all_reasons = seq_reasons + struct_reasons

    # Compute composite score from available components
    score_values = [v for v in all_scores.values() if v > 0.0]
    composite_score = sum(score_values) / len(score_values) if score_values else 0.0

    candidate.validation_score = composite_score
    candidate.passed = len(all_reasons) == 0
    candidate.reasons = tuple(all_reasons)
    candidate.metadata["component_scores"] = all_scores

    return candidate


def rank_candidates(
    candidates: Sequence[GeneratedProteinCandidate],
) -> list[GeneratedProteinCandidate]:
    """Sort candidates by validation_score descending."""
    return sorted(candidates, key=lambda c: c.validation_score, reverse=True)


def _best_confidence(prediction: StructurePrediction) -> float:
    """Extract best available confidence metric, normalized to [0, 1]."""
    for value in (prediction.confidence, prediction.plddt, prediction.ptm, prediction.iptm):
        if value is None:
            continue
        normalized = float(value)
        if normalized > 1.0:
            normalized = normalized / 100.0
        return max(0.0, min(1.0, normalized))
    return 0.0

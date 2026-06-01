from __future__ import annotations

from .types import (
    VALID_PROTEIN_AMINO_ACIDS,
    ProteinStructureScore,
    StructurePrediction,
    StructureScoringWeights,
)


def compact_protein_sequence(sequence: str) -> str:
    return "".join(str(sequence).split()).upper()


def valid_amino_acid_fraction(sequence: str) -> float:
    normalized = compact_protein_sequence(sequence)
    if not normalized:
        return 0.0
    valid_count = sum(1 for residue in normalized if residue in VALID_PROTEIN_AMINO_ACIDS)
    return valid_count / len(normalized)


def ambiguity_fraction(sequence: str) -> float:
    normalized = compact_protein_sequence(sequence)
    if not normalized:
        return 1.0
    return normalized.count("X") / len(normalized)


def length_window_score(sequence: str, *, min_length: int, max_length: int) -> float:
    if min_length <= 0:
        raise ValueError("min_length must be greater than 0.")
    if max_length < min_length:
        raise ValueError("max_length must be greater than or equal to min_length.")

    length = len(compact_protein_sequence(sequence))
    if min_length <= length <= max_length:
        return 1.0
    if length <= 0:
        return 0.0

    if length < min_length:
        return max(length / min_length, 0.0)
    return max(max_length / length, 0.0)


def score_protein_candidate(
    sequence: str,
    *,
    prediction: StructurePrediction | None = None,
    min_length: int = 30,
    max_length: int = 1024,
    max_x_fraction: float = 0.05,
    geometry_score: float | None = None,
    contact_score: float | None = None,
    weights: StructureScoringWeights | None = None,
) -> ProteinStructureScore:
    resolved_weights = weights or StructureScoringWeights()
    normalized = compact_protein_sequence(sequence)

    validity = valid_amino_acid_fraction(normalized)
    length = length_window_score(normalized, min_length=min_length, max_length=max_length)
    ambiguity = 1.0 - ambiguity_fraction(normalized)
    model_confidence = _normalize_model_confidence(prediction)
    geo_confidence = _clamp01(geometry_score) if geometry_score is not None else 0.0
    contact_consistency = _clamp01(contact_score) if contact_score is not None else 0.0

    component_scores = {
        "validity": validity,
        "length": length,
        "ambiguity": ambiguity,
        "model_confidence": model_confidence,
        "geometry_confidence": geo_confidence,
        "contact_consistency": contact_consistency,
    }
    total_weight = (
        resolved_weights.validity
        + resolved_weights.length
        + resolved_weights.ambiguity
        + resolved_weights.model_confidence
        + resolved_weights.geometry_confidence
        + resolved_weights.contact_consistency
    )
    if total_weight <= 0:
        raise ValueError("At least one scoring weight must be positive.")

    total_score = (
        resolved_weights.validity * validity
        + resolved_weights.length * length
        + resolved_weights.ambiguity * ambiguity
        + resolved_weights.model_confidence * model_confidence
        + resolved_weights.geometry_confidence * geo_confidence
        + resolved_weights.contact_consistency * contact_consistency
    ) / total_weight

    reasons: list[str] = []
    if not normalized:
        reasons.append("empty_sequence")
    if validity < 1.0:
        reasons.append("invalid_amino_acid")
    if ambiguity_fraction(normalized) > max_x_fraction:
        reasons.append("too_many_x")
    if length < 1.0:
        reasons.append("length_outside_window")

    return ProteinStructureScore(
        sequence=normalized,
        total_score=_clamp01(total_score),
        passed=not reasons,
        component_scores=component_scores,
        reasons=tuple(reasons),
        prediction=prediction,
    )


def _normalize_model_confidence(prediction: StructurePrediction | None) -> float:
    if prediction is None:
        return 0.0
    for value in (prediction.confidence, prediction.plddt, prediction.ptm, prediction.iptm):
        if value is None:
            continue
        normalized = float(value)
        if normalized > 1.0:
            normalized = normalized / 100.0
        return _clamp01(normalized)
    return 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

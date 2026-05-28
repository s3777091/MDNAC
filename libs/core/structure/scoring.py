from __future__ import annotations

import math
from collections import Counter

from .types import (
    PROSTT5_3DI_TOKENS,
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


def structure_3di_plausibility_score(structure_3di: str | None) -> float:
    if not structure_3di:
        return 0.0

    normalized = "".join(str(structure_3di).split()).lower()
    if not normalized:
        return 0.0

    valid_fraction = sum(1 for token in normalized if token in PROSTT5_3DI_TOKENS) / len(normalized)
    repetition_penalty = _max_run_fraction(normalized)
    entropy_score = _normalized_entropy(normalized)
    return _clamp01((0.45 * valid_fraction) + (0.35 * entropy_score) + (0.20 * (1.0 - repetition_penalty)))


def score_protein_candidate(
    sequence: str,
    *,
    prediction: StructurePrediction | None = None,
    min_length: int = 30,
    max_length: int = 1024,
    max_x_fraction: float = 0.05,
    weights: StructureScoringWeights | None = None,
) -> ProteinStructureScore:
    resolved_weights = weights or StructureScoringWeights()
    normalized = compact_protein_sequence(sequence)

    validity = valid_amino_acid_fraction(normalized)
    length = length_window_score(normalized, min_length=min_length, max_length=max_length)
    ambiguity = 1.0 - ambiguity_fraction(normalized)
    structure_plausibility = structure_3di_plausibility_score(
        prediction.structure_3di if prediction is not None else None
    )
    model_confidence = _normalize_model_confidence(prediction)

    component_scores = {
        "validity": validity,
        "length": length,
        "ambiguity": ambiguity,
        "structure_plausibility": structure_plausibility,
        "model_confidence": model_confidence,
    }
    total_weight = (
        resolved_weights.validity
        + resolved_weights.length
        + resolved_weights.ambiguity
        + resolved_weights.structure_plausibility
        + resolved_weights.model_confidence
    )
    if total_weight <= 0:
        raise ValueError("At least one scoring weight must be positive.")

    total_score = (
        resolved_weights.validity * validity
        + resolved_weights.length * length
        + resolved_weights.ambiguity * ambiguity
        + resolved_weights.structure_plausibility * structure_plausibility
        + resolved_weights.model_confidence * model_confidence
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


def _max_run_fraction(text: str) -> float:
    if not text:
        return 1.0

    max_run = 1
    current_run = 1
    previous = text[0]
    for character in text[1:]:
        if character == previous:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            previous = character
            current_run = 1
    return max_run / len(text)


def _normalized_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    if len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / len(text)
        entropy -= probability * math.log(probability)
    return _clamp01(entropy / math.log(len(counts)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

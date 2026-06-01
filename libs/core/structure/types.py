from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


VALID_PROTEIN_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYX")
CANONICAL_PROTEIN_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(slots=True, frozen=True)
class StructurePrediction:
    sequence: str
    model_name: str
    confidence: float | None = None
    plddt: float | None = None
    ptm: float | None = None
    iptm: float | None = None
    affinity: float | None = None
    coordinates_path: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class StructureScoringWeights:
    validity: float = 0.30
    length: float = 0.15
    ambiguity: float = 0.10
    model_confidence: float = 0.25
    geometry_confidence: float = 0.20
    contact_consistency: float = 0.0


@dataclass(slots=True, frozen=True)
class ProteinStructureScore:
    sequence: str
    total_score: float
    passed: bool
    component_scores: dict[str, float]
    reasons: tuple[str, ...] = ()
    prediction: StructurePrediction | None = None


@dataclass(slots=True, frozen=True)
class CoevolutionContact:
    i: int
    j: int
    score: float


@dataclass(slots=True, frozen=True)
class ExternalStructureProviderSpec:
    name: str
    provider_type: str
    recommended_role: str
    strengths: tuple[str, ...]
    limitations: tuple[str, ...]
    install_hint: str | None = None
    license_note: str | None = None

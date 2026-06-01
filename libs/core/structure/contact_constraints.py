"""Contact constraint evaluation using coevolution and geometry modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .coevolution import top_coevolving_pairs
from .geometry import pairwise_distances, triangle_consistency_score


@dataclass(slots=True, frozen=True)
class ContactConstraint:
    i: int
    j: int
    min_distance: float | None = None
    max_distance: float | None = 8.0
    score: float | None = None


def build_contact_constraints_from_msa(
    msa: Sequence[str],
    *,
    top_k: int = 32,
    min_separation: int = 5,
) -> tuple[ContactConstraint, ...]:
    """Build contact constraints from MSA coevolution analysis.

    Uses mutual information with APC correction to identify the top coevolving pairs.
    """
    pairs = top_coevolving_pairs(msa, top_k=top_k, min_separation=min_separation)
    return tuple(
        ContactConstraint(i=pair.i, j=pair.j, max_distance=8.0, score=pair.score)
        for pair in pairs
    )


def evaluate_contact_constraints(
    distance_matrix: Sequence[Sequence[float]] | np.ndarray,
    constraints: Sequence[ContactConstraint],
) -> tuple[float, list[str]]:
    """Evaluate contact constraints against a distance matrix.

    Returns a score in [0, 1] and a list of failure reasons.
    A pair passes if distance <= max_distance (when max_distance is provided)
    and distance >= min_distance (when min_distance is provided).
    """
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D matrix.")

    if not constraints:
        return 1.0, []

    satisfied = 0
    reasons: list[str] = []

    for constraint in constraints:
        i, j = constraint.i, constraint.j
        if i < 0 or j < 0 or i >= distances.shape[0] or j >= distances.shape[0]:
            reasons.append(f"constraint_index_out_of_bounds({i},{j})")
            continue

        dist = float(distances[i, j])
        passes = True

        if constraint.max_distance is not None and dist > constraint.max_distance:
            passes = False
        if constraint.min_distance is not None and dist < constraint.min_distance:
            passes = False

        if passes:
            satisfied += 1
        else:
            reasons.append(f"contact_violated({i},{j},dist={dist:.2f})")

    total = len(constraints)
    score = satisfied / total if total > 0 else 1.0
    return score, reasons


def evaluate_triangle_geometry(
    coordinates: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """Evaluate triangle consistency of predicted coordinates.

    Returns the triangle consistency score in [0, 1].
    """
    distances = pairwise_distances(coordinates)
    return triangle_consistency_score(distances)

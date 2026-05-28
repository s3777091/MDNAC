from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def triangle_inequality_violation_rate(
    distance_matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> float:
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D matrix.")
    if tolerance < 0:
        raise ValueError("tolerance must be greater than or equal to 0.")

    length = int(distances.shape[0])
    if length < 3:
        return 0.0

    violations = 0
    checks = 0
    for i in range(length):
        for j in range(i + 1, length):
            for k in range(j + 1, length):
                dij = float(distances[i, j])
                dik = float(distances[i, k])
                djk = float(distances[j, k])
                if min(dij, dik, djk) < 0:
                    raise ValueError("Distances must be non-negative.")
                checks += 3
                if dij + dik + tolerance < djk:
                    violations += 1
                if dij + djk + tolerance < dik:
                    violations += 1
                if dik + djk + tolerance < dij:
                    violations += 1

    return violations / checks if checks else 0.0


def triangle_consistency_score(
    distance_matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> float:
    return 1.0 - triangle_inequality_violation_rate(distance_matrix, tolerance=tolerance)


def pairwise_distances(coordinates: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("coordinates must be shaped [n, 3].")
    deltas = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=-1))


def contact_precision_at_k(
    predicted_pairs: Sequence[tuple[int, int]],
    distance_matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    k: int,
    contact_threshold: float = 8.0,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D matrix.")

    selected = list(predicted_pairs[:k])
    if not selected:
        return 0.0

    hits = 0
    for left, right in selected:
        if left < 0 or right < 0 or left >= distances.shape[0] or right >= distances.shape[0]:
            raise ValueError("predicted pair index is outside the distance matrix.")
        if float(distances[left, right]) <= contact_threshold:
            hits += 1
    return hits / len(selected)

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import CoevolutionContact


DEFAULT_MSA_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWYX-")


def parse_fasta_msa(text: str) -> tuple[str, ...]:
    sequences: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                sequences.append("".join(current).upper())
                current = []
            continue
        current.append(line)
    if current:
        sequences.append("".join(current).upper())
    return tuple(sequences)


def mutual_information_matrix(
    msa: Sequence[str],
    *,
    alphabet: Sequence[str] = DEFAULT_MSA_ALPHABET,
    pseudocount: float = 1e-3,
) -> np.ndarray:
    encoded, alphabet_size = _encode_msa(msa, alphabet=alphabet)
    sequence_count, length = encoded.shape
    if sequence_count < 2:
        raise ValueError("MSA must contain at least two sequences.")

    matrix = np.zeros((length, length), dtype=np.float64)
    for left in range(length):
        left_counts = np.bincount(encoded[:, left], minlength=alphabet_size).astype(np.float64)
        left_probs = _normalize_counts(left_counts, pseudocount=pseudocount)
        for right in range(left + 1, length):
            right_counts = np.bincount(encoded[:, right], minlength=alphabet_size).astype(np.float64)
            right_probs = _normalize_counts(right_counts, pseudocount=pseudocount)
            joint_counts = np.zeros((alphabet_size, alphabet_size), dtype=np.float64)
            np.add.at(joint_counts, (encoded[:, left], encoded[:, right]), 1.0)
            joint_probs = _normalize_counts(joint_counts, pseudocount=pseudocount)

            expected = left_probs[:, None] * right_probs[None, :]
            positive = joint_probs > 0
            mi = float(np.sum(joint_probs[positive] * np.log(joint_probs[positive] / expected[positive])))
            matrix[left, right] = mi
            matrix[right, left] = mi
    return matrix


def apc_correct(matrix: np.ndarray) -> np.ndarray:
    scores = np.asarray(matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("matrix must be a square 2D array.")

    corrected = scores.copy()
    np.fill_diagonal(corrected, 0.0)
    row_mean = corrected.mean(axis=1)
    col_mean = corrected.mean(axis=0)
    overall_mean = float(corrected.mean())
    if overall_mean == 0.0:
        return corrected
    corrected = corrected - (row_mean[:, None] * col_mean[None, :] / overall_mean)
    np.fill_diagonal(corrected, 0.0)
    return corrected


def top_coevolving_pairs(
    msa: Sequence[str],
    *,
    top_k: int = 32,
    min_separation: int = 5,
    use_apc: bool = True,
) -> tuple[CoevolutionContact, ...]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    if min_separation < 0:
        raise ValueError("min_separation must be greater than or equal to 0.")

    scores = mutual_information_matrix(msa)
    if use_apc:
        scores = apc_correct(scores)

    contacts: list[CoevolutionContact] = []
    length = int(scores.shape[0])
    for left in range(length):
        for right in range(left + 1, length):
            if right - left < min_separation:
                continue
            contacts.append(CoevolutionContact(i=left, j=right, score=float(scores[left, right])))
    contacts.sort(key=lambda contact: contact.score, reverse=True)
    return tuple(contacts[:top_k])


def _encode_msa(
    msa: Sequence[str],
    *,
    alphabet: Sequence[str],
) -> tuple[np.ndarray, int]:
    sequences = tuple("".join(str(sequence).split()).upper() for sequence in msa)
    if not sequences:
        raise ValueError("MSA must not be empty.")

    length = len(sequences[0])
    if length == 0:
        raise ValueError("MSA sequences must not be empty.")
    for sequence in sequences:
        if len(sequence) != length:
            raise ValueError("All MSA sequences must have the same aligned length.")

    token_to_index = {token: index for index, token in enumerate(alphabet)}
    unknown_index = token_to_index.get("X", len(token_to_index) - 1)
    encoded = np.empty((len(sequences), length), dtype=np.int64)
    for row, sequence in enumerate(sequences):
        for column, token in enumerate(sequence):
            encoded[row, column] = token_to_index.get(token, unknown_index)
    return encoded, len(token_to_index)


def _normalize_counts(counts: np.ndarray, *, pseudocount: float) -> np.ndarray:
    normalized = counts + pseudocount
    total = float(normalized.sum())
    if total <= 0.0:
        raise ValueError("Cannot normalize empty counts.")
    return normalized / total

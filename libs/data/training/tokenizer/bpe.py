from __future__ import annotations

from collections import Counter, deque


def find_freq_pair(token_id_sequences: list[list[int]]) -> tuple[int, int] | None:
    pairs = Counter(
        pair
        for token_ids in token_id_sequences
        for pair in zip(token_ids, token_ids[1:])
    )
    if not pairs:
        return None
    return max(pairs.items(), key=lambda item: item[1])[0]


def replace_pair(
    token_id_sequences: list[list[int]],
    pair_id: tuple[int, int],
    new_id: int,
) -> list[list[int]]:
    replaced_sequences: list[list[int]] = []
    for token_ids in token_id_sequences:
        dq = deque(token_ids)
        replaced: list[int] = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                dq.popleft()
            else:
                replaced.append(current)

        replaced_sequences.append(replaced)

    return replaced_sequences


def replace_pair_once(token_ids: list[int], pair_id: tuple[int, int], new_id: int) -> list[int]:
    replaced: list[int] = []
    index = 0
    while index < len(token_ids):
        if index < len(token_ids) - 1 and (token_ids[index], token_ids[index + 1]) == pair_id:
            replaced.append(new_id)
            index += 2
        else:
            replaced.append(token_ids[index])
            index += 1
    return replaced

"""Binary cache format I/O for token sequences."""

from __future__ import annotations

import struct
from array import array


def _array_typecode_for_vocab_size(vocab_size: int) -> str:
    if vocab_size <= 256:
        return "B"
    if vocab_size <= 65_536:
        return "H"
    return "I"


def _array_item_size(typecode: str) -> int:
    return array(typecode).itemsize


def _write_token_id_sequence(handle, token_ids: list[int], typecode: str) -> None:
    handle.write(struct.pack("<I", len(token_ids)))
    array(typecode, token_ids).tofile(handle)


def _read_token_id_sequence(handle, typecode: str) -> array | None:
    raw_length = handle.read(4)
    if not raw_length:
        return None
    if len(raw_length) != 4:
        raise ValueError("Truncated tokenizer cache record length.")

    length = struct.unpack("<I", raw_length)[0]
    token_ids = array(typecode)
    try:
        token_ids.fromfile(handle, length)
    except EOFError as exc:
        raise ValueError("Truncated tokenizer cache record payload.") from exc
    return token_ids


def _encoded_sequence_byte_size(token_ids: array, typecode: str) -> int:
    return 4 + (len(token_ids) * _array_item_size(typecode))


def _replace_pair_once_in_sequence(
    token_ids: array,
    pair: tuple[int, int],
    new_id: int,
) -> list[int]:
    replaced: list[int] = []
    index = 0
    while index < len(token_ids):
        current = int(token_ids[index])
        if index < len(token_ids) - 1 and (current, int(token_ids[index + 1])) == pair:
            replaced.append(new_id)
            index += 2
        else:
            replaced.append(current)
            index += 1
    return replaced

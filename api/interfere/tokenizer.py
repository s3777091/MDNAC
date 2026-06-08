from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_SPECIAL_TOKENS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|endoftext|>",
    "<|protein|>",
)
PROTEIN_START_TOKEN = "<|protein|>"
PROTEIN_END_TOKEN = "<|endoftext|>"


@dataclass(slots=True)
class ProteinTokenizer:
    str_to_int: dict[str, int]
    int_to_str: dict[int, str]
    special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS
    bpe_merges: dict[tuple[int, int], int] = field(default_factory=dict)
    merge_ranks: dict[tuple[int, int], int] = field(default_factory=dict)
    sequence_type: str = "protein"

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "ProteinTokenizer":
        tokenizer_payload = payload.get("tokenizer", payload)
        if not isinstance(tokenizer_payload, dict):
            raise ValueError("Tokenizer payload must be a JSON object.")

        str_to_int = {
            str(token): int(token_id)
            for token, token_id in dict(tokenizer_payload["str_to_int"]).items()
        }
        raw_int_to_str = tokenizer_payload.get("int_to_str")
        if isinstance(raw_int_to_str, dict):
            int_to_str = {
                int(token_id): str(token)
                for token_id, token in raw_int_to_str.items()
            }
        else:
            int_to_str = {token_id: token for token, token_id in str_to_int.items()}

        special_tokens = tuple(
            str(token)
            for token in tokenizer_payload.get("special_tokens", DEFAULT_SPECIAL_TOKENS)
        )
        bpe_merges: dict[tuple[int, int], int] = {}
        merge_ranks: dict[tuple[int, int], int] = {}
        for rank, merge in enumerate(tokenizer_payload.get("bpe_merges", [])):
            if not isinstance(merge, dict):
                raise ValueError("Tokenizer BPE merges must be objects.")
            pair = tuple(int(item) for item in merge["pair"])
            if len(pair) != 2:
                raise ValueError("Tokenizer BPE merge pairs must contain two token IDs.")
            new_id = int(merge["new_id"])
            bpe_merges[pair] = new_id
            merge_ranks[pair] = rank

        tokenizer = cls(
            str_to_int=str_to_int,
            int_to_str=int_to_str,
            special_tokens=special_tokens,
            bpe_merges=bpe_merges,
            merge_ranks=merge_ranks,
            sequence_type=str(tokenizer_payload.get("sequence_type", "protein")),
        )
        tokenizer._validate()
        return tokenizer

    @classmethod
    def load_map(cls, path: Path | str) -> "ProteinTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Tokenizer map must be a JSON object: {path}")
        return cls.from_payload(payload)

    @property
    def vocab_size(self) -> int:
        return len(self.str_to_int)

    @property
    def eos_token_id(self) -> int:
        return self.str_to_int[PROTEIN_END_TOKEN]

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        allowed_tokens = set(self.special_tokens if allowed_special is None else allowed_special)
        token_ids: list[int] = []
        for token in _pretokenize_text(
            text,
            special_tokens=self.special_tokens,
            allowed_special=allowed_tokens,
        ):
            if token in allowed_tokens:
                token_ids.append(self.str_to_int[token])
            else:
                token_ids.extend(self.tokenize_with_bpe(token))
        return token_ids

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = False) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            normalized_id = int(token_id)
            if normalized_id not in self.int_to_str:
                raise ValueError(f"Token ID {normalized_id} not found in tokenizer map.")
            token = self.int_to_str[normalized_id]
            if skip_special and token in self.special_tokens:
                continue
            pieces.append(token)
        return "".join(pieces)

    def tokenize_with_bpe(self, token: str) -> list[int]:
        token_ids = [self.str_to_int.get(character) for character in token]
        if None in token_ids:
            missing = [
                character
                for character, token_id in zip(token, token_ids, strict=True)
                if token_id is None
            ]
            raise ValueError(f"Characters not found in tokenizer vocabulary: {missing}")

        merged_ids = [int(token_id) for token_id in token_ids]
        while len(merged_ids) > 1:
            ranked_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in zip(merged_ids, merged_ids[1:])
                if pair in self.merge_ranks
            ]
            if not ranked_pairs:
                break

            _, best_pair = min(ranked_pairs, key=lambda item: item[0])
            merged_ids = _replace_pair_once(merged_ids, best_pair, self.bpe_merges[best_pair])
        return merged_ids

    def _validate(self) -> None:
        if self.sequence_type != "protein":
            raise ValueError("Only protein tokenizer maps are supported by this API.")
        for token in (PROTEIN_START_TOKEN, PROTEIN_END_TOKEN):
            if token not in self.str_to_int:
                raise ValueError(f"Tokenizer map is missing required token: {token}")


def extract_protein_sequence(text: str) -> str:
    start_index = text.rfind(PROTEIN_START_TOKEN)
    if start_index != -1:
        text = text[start_index + len(PROTEIN_START_TOKEN):]

    end_index = text.find(PROTEIN_END_TOKEN)
    if end_index != -1:
        text = text[:end_index]

    for special_token in DEFAULT_SPECIAL_TOKENS:
        text = text.replace(special_token, "")
    return "".join(text.split())


def normalize_protein_prompt(prompt: str, *, ensure_start_token: bool = True) -> str:
    normalized = prompt.strip()
    if not normalized:
        return PROTEIN_START_TOKEN
    if ensure_start_token and not normalized.startswith(PROTEIN_START_TOKEN):
        return f"{PROTEIN_START_TOKEN}{normalized}"
    return normalized


def _pretokenize_text(
    text: str,
    *,
    special_tokens: tuple[str, ...],
    allowed_special: set[str],
) -> list[str]:
    tokens: list[str] = []
    ordered_special_tokens = sorted(special_tokens, key=len, reverse=True)
    index = 0
    buffer: list[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            tokens.append("".join(buffer))
            buffer = []

    while index < len(text):
        matched_special = None
        for special_token in ordered_special_tokens:
            if text.startswith(special_token, index):
                matched_special = special_token
                break

        if matched_special is not None:
            if matched_special not in allowed_special:
                raise ValueError(f"Disallowed special token encountered in text: {matched_special}")
            flush_buffer()
            tokens.append(matched_special)
            index += len(matched_special)
            continue

        character = text[index]
        if character != "\r":
            buffer.append(character)
        index += 1

    flush_buffer()
    return tokens


def _replace_pair_once(token_ids: list[int], pair_id: tuple[int, int], new_id: int) -> list[int]:
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

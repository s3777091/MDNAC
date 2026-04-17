from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_PROFILE_SPECIAL_TOKENS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
)


@dataclass(slots=True)
class ProfileBPETokenizer:
    str_to_int: dict[str, int] = field(default_factory=dict)
    int_to_str: dict[int, str] = field(default_factory=dict)
    special_tokens: tuple[str, ...] = DEFAULT_PROFILE_SPECIAL_TOKENS
    bpe_merges: dict[tuple[int, int], int] = field(default_factory=dict)
    merge_ranks: dict[tuple[int, int], int] = field(default_factory=dict)
    base_charset: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_text(
        cls,
        text: str,
        vocab_size: int = 256,
        allowed_special: set[str] | None = None,
    ) -> "ProfileBPETokenizer":
        tokenizer = cls()
        tokenizer.train(
            text=text,
            vocab_size=vocab_size,
            allowed_special=allowed_special or set(tokenizer.special_tokens),
        )
        return tokenizer

    @classmethod
    def load_map(cls, path: Path | str) -> "ProfileBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer_payload = payload.get("tokenizer", payload)
        str_to_int = {str(token): int(token_id) for token, token_id in tokenizer_payload["str_to_int"].items()}
        int_to_str = {int(token_id): str(token) for token_id, token in tokenizer_payload["int_to_str"].items()}
        special_tokens = tuple(tokenizer_payload.get("special_tokens", DEFAULT_PROFILE_SPECIAL_TOKENS))
        base_charset = tuple(tokenizer_payload.get("base_charset", ()))

        bpe_merges: dict[tuple[int, int], int] = {}
        merge_ranks: dict[tuple[int, int], int] = {}
        for rank, merge in enumerate(tokenizer_payload.get("bpe_merges", [])):
            pair = tuple(int(item) for item in merge["pair"])
            new_id = int(merge["new_id"])
            bpe_merges[pair] = new_id
            merge_ranks[pair] = rank

        return cls(
            str_to_int=str_to_int,
            int_to_str=int_to_str,
            special_tokens=special_tokens,
            bpe_merges=bpe_merges,
            merge_ranks=merge_ranks,
            base_charset=base_charset,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.str_to_int)

    @property
    def pad_token_id(self) -> int:
        return self.str_to_int["<|pad|>"]

    def train(
        self,
        text: str,
        vocab_size: int,
        allowed_special: set[str] | None = None,
    ) -> None:
        allowed_tokens = set(self.special_tokens) if allowed_special is None else set(allowed_special)
        self._initialize_base_vocab(text)

        token_sequences = self._pretokenize_text(text, allowed_special=allowed_tokens)
        token_id_sequences = [
            [self.str_to_int[character] for character in token]
            for token in token_sequences
            if token and token not in allowed_tokens
        ]

        target_vocab_size = max(vocab_size, len(self.str_to_int))
        next_token_id = len(self.str_to_int)

        while next_token_id < target_vocab_size:
            pair = self.find_freq_pair(token_id_sequences)
            if pair is None:
                break

            merged_token = self.int_to_str[pair[0]] + self.int_to_str[pair[1]]
            if merged_token in self.str_to_int:
                token_id_sequences = self.replace_pair(token_id_sequences, pair, self.str_to_int[merged_token])
                continue

            self.str_to_int[merged_token] = next_token_id
            self.int_to_str[next_token_id] = merged_token
            self.bpe_merges[pair] = next_token_id
            self.merge_ranks[pair] = len(self.merge_ranks)
            token_id_sequences = self.replace_pair(token_id_sequences, pair, next_token_id)
            next_token_id += 1

    def encode(
        self,
        text: str,
        allowed_special: set[str] | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        allowed_tokens = set(self.special_tokens) if allowed_special is None else set(allowed_special)
        token_ids: list[int] = []

        if add_bos:
            token_ids.append(self.str_to_int["<|bos|>"])

        for token in self._pretokenize_text(text, allowed_special=allowed_tokens):
            if token in allowed_tokens:
                token_ids.append(self.str_to_int[token])
            else:
                token_ids.extend(self.tokenize_with_bpe(token))

        if add_eos:
            token_ids.append(self.str_to_int["<|eos|>"])

        return token_ids

    def decode(self, token_ids: Iterable[int], skip_special: bool = False) -> str:
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
            missing_chars = [character for character, token_id in zip(token, token_ids) if token_id is None]
            raise ValueError(f"Characters not found in tokenizer vocabulary: {missing_chars}")

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
            merged_ids = self._replace_pair_once(merged_ids, best_pair, self.bpe_merges[best_pair])

        return merged_ids

    def save_map(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def to_json(self) -> str:
        payload = {
            "special_tokens": list(self.special_tokens),
            "base_charset": list(self.base_charset),
            "str_to_int": self.str_to_int,
            "int_to_str": {str(token_id): token for token_id, token in self.int_to_str.items()},
            "bpe_merges": [
                {"pair": [pair[0], pair[1]], "new_id": new_id}
                for pair, new_id in self.bpe_merges.items()
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    def _initialize_base_vocab(self, text: str) -> None:
        seen_chars = []
        seen_set: set[str] = set()
        for character in text.replace("\r", ""):
            if character in seen_set:
                continue
            seen_chars.append(character)
            seen_set.add(character)

        self.base_charset = tuple(seen_chars)
        ordered_tokens = [*self.special_tokens, *self.base_charset]
        self.str_to_int = {token: token_id for token_id, token in enumerate(ordered_tokens)}
        self.int_to_str = {token_id: token for token, token_id in self.str_to_int.items()}
        self.bpe_merges = {}
        self.merge_ranks = {}

    def _pretokenize_text(self, text: str, allowed_special: set[str]) -> list[str]:
        tokens: list[str] = []
        ordered_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
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
            if character == "\r":
                index += 1
                continue

            buffer.append(character)
            index += 1

        flush_buffer()
        return tokens

    @staticmethod
    def find_freq_pair(token_id_sequences: list[list[int]]) -> tuple[int, int] | None:
        pairs = Counter(
            pair
            for token_ids in token_id_sequences
            for pair in zip(token_ids, token_ids[1:])
        )
        if not pairs:
            return None
        return max(pairs.items(), key=lambda item: item[1])[0]

    @staticmethod
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

    @staticmethod
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

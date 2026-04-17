from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from libs.data.entities import SequenceRecord
from libs.data.training.tokenizer.bpe import find_freq_pair, replace_pair, replace_pair_once
from libs.data.training.tokenizer.constants import DEFAULT_SPECIAL_TOKENS, DEFAULT_VOCAB_SIZES, base_tokens
from libs.data.training.tokenizer.pretokenization import pretokenize_text


@dataclass(slots=True)
class SequenceTokenizer:
    sequence_type: str = "protein"
    str_to_int: dict[str, int] = field(default_factory=dict)
    int_to_str: dict[int, str] = field(default_factory=dict)
    special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS
    bpe_merges: dict[tuple[int, int], int] = field(default_factory=dict)
    merge_ranks: dict[tuple[int, int], int] = field(default_factory=dict)

    @classmethod
    def from_sequence_type(cls, sequence_type: str = "protein") -> "SequenceTokenizer":
        normalized_type = _normalize_sequence_type(sequence_type)
        tokenizer = cls(sequence_type=normalized_type)
        tokenizer._initialize_base_vocab(normalized_type)
        return tokenizer

    @classmethod
    def from_text(
        cls,
        text: str,
        sequence_type: str = "protein",
        vocab_size: int | None = None,
        allowed_special: set[str] | None = None,
    ) -> "SequenceTokenizer":
        normalized_type = _normalize_sequence_type(sequence_type)
        tokenizer = cls.from_sequence_type(normalized_type)
        tokenizer.train(
            text=text,
            vocab_size=vocab_size or DEFAULT_VOCAB_SIZES["protein"],
            allowed_special=allowed_special or set(tokenizer.special_tokens),
        )
        return tokenizer

    @classmethod
    def from_records(
        cls,
        records: Iterable[SequenceRecord],
        vocab_size: int | None = None,
    ) -> "SequenceTokenizer":
        record_list = list(records)
        for record in record_list:
            sequence_type = record.metadata.get("sequence_type", "protein").strip().lower() or "protein"
            _normalize_sequence_type(sequence_type)

        train_text = "\n".join(record.to_training_line() for record in record_list) + "\n"
        return cls.from_text(train_text, sequence_type="protein", vocab_size=vocab_size)

    @classmethod
    def load_map(cls, path: Path | str) -> "SequenceTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer_payload = payload.get("tokenizer", payload)
        str_to_int = {str(token): int(token_id) for token, token_id in tokenizer_payload["str_to_int"].items()}
        int_to_str = {int(token_id): str(token) for token_id, token in tokenizer_payload["int_to_str"].items()}
        special_tokens = tuple(tokenizer_payload.get("special_tokens", DEFAULT_SPECIAL_TOKENS))
        sequence_type = _normalize_sequence_type(str(tokenizer_payload.get("sequence_type", "protein")))

        merges_payload = tokenizer_payload.get("bpe_merges", [])
        bpe_merges: dict[tuple[int, int], int] = {}
        merge_ranks: dict[tuple[int, int], int] = {}
        for rank, merge in enumerate(merges_payload):
            pair = tuple(int(item) for item in merge["pair"])
            new_id = int(merge["new_id"])
            bpe_merges[pair] = new_id
            merge_ranks[pair] = rank

        return cls(
            sequence_type=sequence_type,
            str_to_int=str_to_int,
            int_to_str=int_to_str,
            special_tokens=special_tokens,
            bpe_merges=bpe_merges,
            merge_ranks=merge_ranks,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.str_to_int)

    def train(
        self,
        text: str,
        vocab_size: int,
        allowed_special: set[str] | None = None,
    ) -> None:
        allowed_tokens = set(self.special_tokens) if allowed_special is None else set(allowed_special)
        self._initialize_base_vocab(self.sequence_type)

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
            "sequence_type": self.sequence_type,
            "special_tokens": list(self.special_tokens),
            "str_to_int": self.str_to_int,
            "int_to_str": {str(token_id): token for token_id, token in self.int_to_str.items()},
            "bpe_merges": [
                {"pair": [pair[0], pair[1]], "new_id": new_id}
                for pair, new_id in self.bpe_merges.items()
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    def _initialize_base_vocab(self, sequence_type: str) -> None:
        ordered_tokens = [*self.special_tokens, *base_tokens(sequence_type)]
        self.str_to_int = {token: token_id for token_id, token in enumerate(ordered_tokens)}
        self.int_to_str = {token_id: token for token, token_id in self.str_to_int.items()}
        self.bpe_merges = {}
        self.merge_ranks = {}

    def _pretokenize_text(self, text: str, allowed_special: set[str]) -> list[str]:
        return pretokenize_text(text, special_tokens=self.special_tokens, allowed_special=allowed_special)

    @staticmethod
    def find_freq_pair(token_id_sequences: list[list[int]]) -> tuple[int, int] | None:
        return find_freq_pair(token_id_sequences)

    @staticmethod
    def replace_pair(
        token_id_sequences: list[list[int]],
        pair_id: tuple[int, int],
        new_id: int,
    ) -> list[list[int]]:
        return replace_pair(token_id_sequences, pair_id, new_id)

    @staticmethod
    def _replace_pair_once(token_ids: list[int], pair_id: tuple[int, int], new_id: int) -> list[int]:
        return replace_pair_once(token_ids, pair_id, new_id)


def _normalize_sequence_type(sequence_type: str) -> str:
    normalized = sequence_type.strip().lower() or "protein"
    if normalized != "protein":
        raise ValueError("SequenceTokenizer only supports protein sequences.")
    return normalized

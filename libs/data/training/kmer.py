from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_KMER_SPECIAL_TOKENS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
)


@dataclass(slots=True)
class KmerTokenizer:
    kmer_size: int = 3
    stride: int = 1
    sequence_type: str = "dna"
    str_to_int: dict[str, int] = field(default_factory=dict)
    int_to_str: dict[int, str] = field(default_factory=dict)
    special_tokens: tuple[str, ...] = DEFAULT_KMER_SPECIAL_TOKENS

    def __post_init__(self) -> None:
        if self.kmer_size <= 0:
            raise ValueError("kmer_size must be greater than 0")
        if self.stride <= 0:
            raise ValueError("stride must be greater than 0")

    @classmethod
    def from_sequences(
        cls,
        sequences: Iterable[str],
        kmer_size: int = 3,
        stride: int = 1,
        sequence_type: str = "dna",
    ) -> "KmerTokenizer":
        tokenizer = cls(kmer_size=kmer_size, stride=stride, sequence_type=sequence_type)
        tokenizer.train(sequences)
        return tokenizer

    @classmethod
    def load_map(cls, path: Path | str) -> "KmerTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer_payload = payload.get("tokenizer", payload)
        str_to_int = {str(token): int(token_id) for token, token_id in tokenizer_payload["str_to_int"].items()}
        int_to_str = {int(token_id): str(token) for token_id, token in tokenizer_payload["int_to_str"].items()}
        return cls(
            kmer_size=int(tokenizer_payload["kmer_size"]),
            stride=int(tokenizer_payload.get("stride", 1)),
            sequence_type=str(tokenizer_payload.get("sequence_type", "dna")),
            str_to_int=str_to_int,
            int_to_str=int_to_str,
            special_tokens=tuple(tokenizer_payload.get("special_tokens", DEFAULT_KMER_SPECIAL_TOKENS)),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.str_to_int)

    @property
    def pad_token_id(self) -> int:
        return self.str_to_int["<|pad|>"]

    def train(self, sequences: Iterable[str]) -> None:
        ordered_tokens = list(self.special_tokens)
        seen: set[str] = set(ordered_tokens)
        for sequence in sequences:
            for token in self._sequence_to_tokens(sequence):
                if token in seen:
                    continue
                ordered_tokens.append(token)
                seen.add(token)

        self.str_to_int = {token: token_id for token_id, token in enumerate(ordered_tokens)}
        self.int_to_str = {token_id: token for token, token_id in self.str_to_int.items()}

    def encode(self, sequence: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not self.str_to_int:
            raise ValueError("Tokenizer vocabulary is empty. Train or load the tokenizer before encoding.")

        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.str_to_int["<|bos|>"])

        for token in self._sequence_to_tokens(sequence):
            if token not in self.str_to_int:
                raise ValueError(f"K-mer '{token}' not found in tokenizer vocabulary.")
            token_ids.append(self.str_to_int[token])

        if add_eos:
            token_ids.append(self.str_to_int["<|eos|>"])

        return token_ids

    def decode(self, token_ids: Iterable[int], skip_special: bool = False) -> str:
        kmers: list[str] = []
        for token_id in token_ids:
            normalized_id = int(token_id)
            if normalized_id not in self.int_to_str:
                raise ValueError(f"Token ID {normalized_id} not found in tokenizer map.")
            token = self.int_to_str[normalized_id]
            if token in self.special_tokens:
                if skip_special:
                    continue
                raise ValueError("Cannot decode special k-mer tokens unless skip_special=True.")
            kmers.append(token)

        if not kmers:
            return ""
        if len(kmers) == 1:
            return kmers[0]

        sequence = kmers[0]
        for kmer in kmers[1:]:
            overlap = min(max(self.kmer_size - 1, 0), len(sequence), len(kmer))
            expected_prefix = sequence[-overlap:] if overlap else ""
            actual_prefix = kmer[:overlap] if overlap else ""
            if overlap and expected_prefix != actual_prefix:
                raise ValueError("K-mer tokens cannot be decoded into a consistent sequence.")
            sequence += kmer[overlap:]
        return sequence

    def save_map(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def to_json(self) -> str:
        payload = {
            "kmer_size": self.kmer_size,
            "stride": self.stride,
            "sequence_type": self.sequence_type,
            "special_tokens": list(self.special_tokens),
            "str_to_int": self.str_to_int,
            "int_to_str": {str(token_id): token for token_id, token in self.int_to_str.items()},
        }
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    def _sequence_to_tokens(self, sequence: str) -> list[str]:
        normalized = _normalize_sequence(sequence, sequence_type=self.sequence_type)
        if not normalized:
            return []
        if len(normalized) <= self.kmer_size:
            return [normalized]

        tokens = [
            normalized[index : index + self.kmer_size]
            for index in range(0, len(normalized) - self.kmer_size + 1, self.stride)
        ]
        if not tokens:
            return [normalized]

        last_token = tokens[-1]
        final_window = normalized[-self.kmer_size :]
        if last_token != final_window:
            tokens.append(final_window)
        return tokens


def _normalize_sequence(sequence: str, sequence_type: str = "protein") -> str:
    compact = "".join(character for character in sequence.upper() if not character.isspace())
    if sequence_type == "protein":
        valid_bases = set("ACDEFGHIKLMNPQRSTVWYX")
        unknown_char = "X"
    elif sequence_type == "rna":
        compact = compact.replace("T", "U")
        valid_bases = {"A", "C", "G", "U", "N"}
        unknown_char = "N"
    else:
        compact = compact.replace("U", "T")
        valid_bases = {"A", "C", "G", "T", "N"}
        unknown_char = "N"

    normalized: list[str] = []
    for base in compact:
        normalized.append(base if base in valid_bases else unknown_char)
    return "".join(normalized)

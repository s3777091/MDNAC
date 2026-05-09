from __future__ import annotations

import hashlib
import os
import tempfile
from array import array
from collections import Counter
from collections.abc import Callable
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from libs.data.entities import SequenceRecord
from libs.data.training.tokenizer.bpe import find_freq_pair, replace_pair, replace_pair_once
from libs.data.training.tokenizer.constants import DEFAULT_SPECIAL_TOKENS, DEFAULT_VOCAB_SIZES, base_tokens
from libs.data.training.tokenizer.pretokenization import pretokenize_text


TOKENIZER_PROGRESS_INTERVAL_BYTES = 512 * 1024 * 1024
TokenizerProgressCallback = Callable[[dict[str, object]], None]


@dataclass(slots=True, frozen=True)
class SequenceTokenizerTextTrainingStats:
    record_count: int
    tokenizer_train_record_count: int
    token_sequence_count: int
    token_count: int


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
        tokenizer = cls()
        tokenizer._load_from_payload(payload)
        return tokenizer

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

    def train_from_text_file(
        self,
        path: Path | str,
        vocab_size: int,
        allowed_special: set[str] | None = None,
        line_limit: int | None = None,
        cache_dir: Path | str | None = None,
        progress_callback: TokenizerProgressCallback | None = None,
        progress_interval_bytes: int = TOKENIZER_PROGRESS_INTERVAL_BYTES,
        resume: bool = False,
    ) -> SequenceTokenizerTextTrainingStats:
        if line_limit is not None and line_limit <= 0:
            raise ValueError("line_limit must be greater than 0 when provided.")

        source_path = Path(path)
        allowed_tokens = set(self.special_tokens) if allowed_special is None else set(allowed_special)
        self._initialize_base_vocab(self.sequence_type)
        base_vocab_size = len(self.str_to_int)

        target_vocab_size = max(vocab_size, base_vocab_size)
        id_typecode = _array_typecode_for_vocab_size(target_vocab_size)
        temp_parent = Path(cache_dir) if cache_dir is not None else None
        state_path: Path | None = None
        success = False

        if resume:
            resume_parent = temp_parent or source_path.parent
            resume_parent.mkdir(parents=True, exist_ok=True)
            state_path, cache_path, rewritten_cache_path = _tokenizer_resume_paths(
                source_path=source_path,
                cache_dir=resume_parent,
                sequence_type=self.sequence_type,
                vocab_size=vocab_size,
                line_limit=line_limit,
                allowed_special=allowed_tokens,
            )
        else:
            cache_descriptor, cache_name = tempfile.mkstemp(
                prefix="sequence-tokenizer-",
                suffix=".bin",
                dir=temp_parent,
            )
            os.close(cache_descriptor)
            cache_path = Path(cache_name)
            rewritten_cache_path = cache_path.with_name(f"{cache_path.name}.rewrite")

        try:
            resume_state = (
                _load_tokenizer_resume_state(
                    state_path,
                    source_path=source_path,
                    sequence_type=self.sequence_type,
                    vocab_size=vocab_size,
                    target_vocab_size=target_vocab_size,
                    line_limit=line_limit,
                    allowed_special=allowed_tokens,
                    id_typecode=id_typecode,
                    progress_callback=progress_callback,
                )
                if state_path is not None
                else None
            )
            if resume_state is not None:
                self._load_from_payload(resume_state["tokenizer"])
                stats = _stats_from_resume_payload(resume_state["stats"])
                restored_cache_path = Path(str(resume_state["cache_path"]))
                if restored_cache_path == rewritten_cache_path:
                    rewritten_cache_path = cache_path
                cache_path = restored_cache_path
                _emit_progress(
                    progress_callback,
                    {
                        "event": "tokenizer_resume_loaded",
                        "path": str(state_path),
                        "cache_path": str(cache_path),
                        "cache_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
                        "completed_merges": len(self.merge_ranks),
                        "vocab_size": self.vocab_size,
                        "target_vocab_size": target_vocab_size,
                    },
                )
            else:
                cache_path.unlink(missing_ok=True)
                rewritten_cache_path.unlink(missing_ok=True)
                stats = self._write_token_cache_from_text_file(
                    source_path,
                    cache_path,
                    id_typecode=id_typecode,
                    allowed_special=allowed_tokens,
                    line_limit=line_limit,
                    progress_callback=progress_callback,
                    progress_interval_bytes=progress_interval_bytes,
                )
                if state_path is not None:
                    _write_tokenizer_resume_state(
                        state_path,
                        source_path=source_path,
                        cache_path=cache_path,
                        sequence_type=self.sequence_type,
                        vocab_size=vocab_size,
                        target_vocab_size=target_vocab_size,
                        line_limit=line_limit,
                        allowed_special=allowed_tokens,
                        id_typecode=id_typecode,
                        stats=stats,
                        tokenizer=self,
                        progress_callback=progress_callback,
                    )

            next_token_id = max(self.int_to_str, default=-1) + 1
            merge_total = max(target_vocab_size - base_vocab_size, 0)
            completed_reason = "target_reached"

            while next_token_id < target_vocab_size:
                merge_index = len(self.merge_ranks) + 1
                pair_counts = self._count_cached_pairs(
                    cache_path,
                    id_typecode=id_typecode,
                    merge_index=merge_index,
                    merge_total=merge_total,
                    target_vocab_size=target_vocab_size,
                    progress_callback=progress_callback,
                    progress_interval_bytes=progress_interval_bytes,
                )
                if not pair_counts:
                    completed_reason = "no_pairs"
                    break

                pair, frequency = max(pair_counts.items(), key=lambda item: item[1])
                merged_token = self.int_to_str[pair[0]] + self.int_to_str[pair[1]]
                if merged_token in self.str_to_int:
                    merged_token_id = self.str_to_int[merged_token]
                    reused_existing_token = True
                else:
                    merged_token_id = next_token_id
                    self.str_to_int[merged_token] = merged_token_id
                    self.int_to_str[merged_token_id] = merged_token
                    next_token_id += 1
                    reused_existing_token = False

                self.bpe_merges[pair] = merged_token_id
                self.merge_ranks[pair] = len(self.merge_ranks)
                _emit_progress(
                    progress_callback,
                    {
                        "event": "bpe_merge_selected",
                        "merge_index": merge_index,
                        "merge_total": merge_total,
                        "pair": pair,
                        "frequency": frequency,
                        "new_id": merged_token_id,
                        "vocab_size": self.vocab_size,
                        "target_vocab_size": target_vocab_size,
                        "reused_existing_token": reused_existing_token,
                    },
                )

                rewritten_cache_path.unlink(missing_ok=True)
                self._rewrite_token_cache_with_merge(
                    cache_path,
                    rewritten_cache_path,
                    id_typecode=id_typecode,
                    pair=pair,
                    new_id=merged_token_id,
                    merge_index=merge_index,
                    merge_total=merge_total,
                    target_vocab_size=target_vocab_size,
                    progress_callback=progress_callback,
                    progress_interval_bytes=progress_interval_bytes,
                )
                old_cache_path = cache_path
                old_cache_path.unlink()
                cache_path = rewritten_cache_path
                rewritten_cache_path = old_cache_path
                if state_path is not None:
                    _write_tokenizer_resume_state(
                        state_path,
                        source_path=source_path,
                        cache_path=cache_path,
                        sequence_type=self.sequence_type,
                        vocab_size=vocab_size,
                        target_vocab_size=target_vocab_size,
                        line_limit=line_limit,
                        allowed_special=allowed_tokens,
                        id_typecode=id_typecode,
                        stats=stats,
                        tokenizer=self,
                        progress_callback=progress_callback,
                    )

            _emit_progress(
                progress_callback,
                {
                    "event": "bpe_complete",
                    "reason": completed_reason,
                    "vocab_size": self.vocab_size,
                    "target_vocab_size": target_vocab_size,
                    "merge_count": len(self.merge_ranks),
                },
            )
            success = True
            return stats
        finally:
            rewritten_cache_path.unlink(missing_ok=True)
            if state_path is None:
                cache_path.unlink(missing_ok=True)
            elif success:
                cache_path.unlink(missing_ok=True)
                state_path.unlink(missing_ok=True)
            elif not state_path.exists():
                cache_path.unlink(missing_ok=True)

    def _write_token_cache_from_text_file(
        self,
        source_path: Path,
        cache_path: Path,
        *,
        id_typecode: str,
        allowed_special: set[str],
        line_limit: int | None,
        progress_callback: TokenizerProgressCallback | None,
        progress_interval_bytes: int,
    ) -> SequenceTokenizerTextTrainingStats:
        total_bytes = source_path.stat().st_size
        bytes_read = 0
        next_progress_bytes = progress_interval_bytes
        record_count = 0
        tokenizer_train_record_count = 0
        token_sequence_count = 0
        token_count = 0

        _emit_progress(
            progress_callback,
            {
                "event": "token_cache_start",
                "path": str(source_path),
                "total_bytes": total_bytes,
                "line_limit": line_limit,
            },
        )
        with source_path.open("r", encoding="utf-8", newline="") as source_handle, cache_path.open(
            "wb"
        ) as cache_handle:
            for raw_line_index, raw_line in enumerate(source_handle):
                bytes_read += len(raw_line.encode("utf-8"))
                line = raw_line.removeprefix("\ufeff") if raw_line_index == 0 else raw_line
                if line.strip():
                    record_count += 1
                    if line_limit is None or tokenizer_train_record_count < line_limit:
                        tokenizer_train_record_count += 1
                        for token in self._pretokenize_text(line, allowed_special=allowed_special):
                            if token in allowed_special:
                                continue
                            token_ids = [self.str_to_int.get(character) for character in token]
                            if None in token_ids:
                                missing_chars = [
                                    character
                                    for character, token_id in zip(token, token_ids)
                                    if token_id is None
                                ]
                                raise ValueError(f"Characters not found in tokenizer vocabulary: {missing_chars}")
                            normalized_token_ids = [int(token_id) for token_id in token_ids]
                            if len(normalized_token_ids) < 2:
                                continue
                            _write_token_id_sequence(cache_handle, normalized_token_ids, id_typecode)
                            token_sequence_count += 1
                            token_count += len(normalized_token_ids)

                if bytes_read >= next_progress_bytes:
                    _emit_progress(
                        progress_callback,
                        {
                            "event": "token_cache_progress",
                            "bytes_read": min(bytes_read, total_bytes),
                            "total_bytes": total_bytes,
                            "records_seen": record_count,
                            "records_used": tokenizer_train_record_count,
                            "token_sequences": token_sequence_count,
                            "token_count": token_count,
                        },
                    )
                    while bytes_read >= next_progress_bytes:
                        next_progress_bytes += progress_interval_bytes

        _emit_progress(
            progress_callback,
            {
                "event": "token_cache_complete",
                "bytes_read": min(bytes_read, total_bytes),
                "total_bytes": total_bytes,
                "records_seen": record_count,
                "records_used": tokenizer_train_record_count,
                "token_sequences": token_sequence_count,
                "token_count": token_count,
                "cache_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
            },
        )
        return SequenceTokenizerTextTrainingStats(
            record_count=record_count,
            tokenizer_train_record_count=tokenizer_train_record_count,
            token_sequence_count=token_sequence_count,
            token_count=token_count,
        )

    def _count_cached_pairs(
        self,
        cache_path: Path,
        *,
        id_typecode: str,
        merge_index: int,
        merge_total: int,
        target_vocab_size: int,
        progress_callback: TokenizerProgressCallback | None,
        progress_interval_bytes: int,
    ) -> Counter[tuple[int, int]]:
        total_bytes = cache_path.stat().st_size if cache_path.exists() else 0
        bytes_read = 0
        next_progress_bytes = progress_interval_bytes
        pair_counts: Counter[tuple[int, int]] = Counter()
        sequence_count = 0

        _emit_progress(
            progress_callback,
            {
                "event": "bpe_count_start",
                "merge_index": merge_index,
                "merge_total": merge_total,
                "vocab_size": self.vocab_size,
                "target_vocab_size": target_vocab_size,
                "total_bytes": total_bytes,
            },
        )
        with cache_path.open("rb") as cache_handle:
            while True:
                token_ids = _read_token_id_sequence(cache_handle, id_typecode)
                if token_ids is None:
                    break
                sequence_count += 1
                bytes_read += _encoded_sequence_byte_size(token_ids, id_typecode)
                for index in range(len(token_ids) - 1):
                    pair_counts[(int(token_ids[index]), int(token_ids[index + 1]))] += 1

                if bytes_read >= next_progress_bytes:
                    _emit_progress(
                        progress_callback,
                        {
                            "event": "bpe_count_progress",
                            "merge_index": merge_index,
                            "merge_total": merge_total,
                            "bytes_read": min(bytes_read, total_bytes),
                            "total_bytes": total_bytes,
                            "sequences": sequence_count,
                            "pair_kinds": len(pair_counts),
                        },
                    )
                    while bytes_read >= next_progress_bytes:
                        next_progress_bytes += progress_interval_bytes

        _emit_progress(
            progress_callback,
            {
                "event": "bpe_count_complete",
                "merge_index": merge_index,
                "merge_total": merge_total,
                "bytes_read": min(bytes_read, total_bytes),
                "total_bytes": total_bytes,
                "sequences": sequence_count,
                "pair_kinds": len(pair_counts),
            },
        )
        return pair_counts

    def _rewrite_token_cache_with_merge(
        self,
        source_path: Path,
        target_path: Path,
        *,
        id_typecode: str,
        pair: tuple[int, int],
        new_id: int,
        merge_index: int,
        merge_total: int,
        target_vocab_size: int,
        progress_callback: TokenizerProgressCallback | None,
        progress_interval_bytes: int,
    ) -> None:
        total_bytes = source_path.stat().st_size if source_path.exists() else 0
        bytes_read = 0
        next_progress_bytes = progress_interval_bytes
        sequence_count = 0
        rewritten_sequence_count = 0

        _emit_progress(
            progress_callback,
            {
                "event": "bpe_rewrite_start",
                "merge_index": merge_index,
                "merge_total": merge_total,
                "vocab_size": self.vocab_size,
                "target_vocab_size": target_vocab_size,
                "pair": pair,
                "new_id": new_id,
                "total_bytes": total_bytes,
            },
        )
        with source_path.open("rb") as source_handle, target_path.open("wb") as target_handle:
            while True:
                token_ids = _read_token_id_sequence(source_handle, id_typecode)
                if token_ids is None:
                    break
                sequence_count += 1
                bytes_read += _encoded_sequence_byte_size(token_ids, id_typecode)
                merged_token_ids = _replace_pair_once_in_sequence(token_ids, pair, new_id)
                if len(merged_token_ids) >= 2:
                    _write_token_id_sequence(target_handle, merged_token_ids, id_typecode)
                    rewritten_sequence_count += 1

                if bytes_read >= next_progress_bytes:
                    _emit_progress(
                        progress_callback,
                        {
                            "event": "bpe_rewrite_progress",
                            "merge_index": merge_index,
                            "merge_total": merge_total,
                            "bytes_read": min(bytes_read, total_bytes),
                            "total_bytes": total_bytes,
                            "sequences": sequence_count,
                            "rewritten_sequences": rewritten_sequence_count,
                        },
                    )
                    while bytes_read >= next_progress_bytes:
                        next_progress_bytes += progress_interval_bytes

        _emit_progress(
            progress_callback,
            {
                "event": "bpe_rewrite_complete",
                "merge_index": merge_index,
                "merge_total": merge_total,
                "bytes_read": min(bytes_read, total_bytes),
                "total_bytes": total_bytes,
                "sequences": sequence_count,
                "rewritten_sequences": rewritten_sequence_count,
                "cache_bytes": target_path.stat().st_size if target_path.exists() else 0,
            },
        )

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

    def _load_from_payload(self, payload: dict[str, object]) -> None:
        tokenizer_payload = payload.get("tokenizer", payload)
        if not isinstance(tokenizer_payload, dict):
            raise ValueError("Tokenizer payload must contain a tokenizer object.")

        self.str_to_int = {
            str(token): int(token_id)
            for token, token_id in tokenizer_payload["str_to_int"].items()
        }
        self.int_to_str = {
            int(token_id): str(token)
            for token_id, token in tokenizer_payload["int_to_str"].items()
        }
        self.special_tokens = tuple(tokenizer_payload.get("special_tokens", DEFAULT_SPECIAL_TOKENS))
        self.sequence_type = _normalize_sequence_type(str(tokenizer_payload.get("sequence_type", "protein")))

        self.bpe_merges = {}
        self.merge_ranks = {}
        for rank, merge in enumerate(tokenizer_payload.get("bpe_merges", [])):
            pair = tuple(int(item) for item in merge["pair"])
            if len(pair) != 2:
                raise ValueError("BPE merge pairs must contain two token IDs.")
            new_id = int(merge["new_id"])
            self.bpe_merges[pair] = new_id
            self.merge_ranks[pair] = rank

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


def _tokenizer_resume_paths(
    *,
    source_path: Path,
    cache_dir: Path,
    sequence_type: str,
    vocab_size: int,
    line_limit: int | None,
    allowed_special: set[str],
) -> tuple[Path, Path, Path]:
    fingerprint_payload = {
        "source_path": str(source_path.resolve()),
        "sequence_type": sequence_type,
        "vocab_size": vocab_size,
        "line_limit": line_limit,
        "allowed_special": sorted(allowed_special),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    base_path = cache_dir / f"sequence-tokenizer-resume-{fingerprint}"
    return (
        base_path.with_suffix(".state.json"),
        base_path.with_suffix(".cache.bin"),
        base_path.with_suffix(".rewrite.bin"),
    )


def _source_signature(source_path: Path) -> dict[str, object]:
    stat = source_path.stat()
    return {
        "path": str(source_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _stats_to_resume_payload(stats: SequenceTokenizerTextTrainingStats) -> dict[str, int]:
    return {
        "record_count": stats.record_count,
        "tokenizer_train_record_count": stats.tokenizer_train_record_count,
        "token_sequence_count": stats.token_sequence_count,
        "token_count": stats.token_count,
    }


def _stats_from_resume_payload(payload: object) -> SequenceTokenizerTextTrainingStats:
    if not isinstance(payload, dict):
        raise ValueError("Tokenizer resume state is missing training stats.")
    return SequenceTokenizerTextTrainingStats(
        record_count=int(payload["record_count"]),
        tokenizer_train_record_count=int(payload["tokenizer_train_record_count"]),
        token_sequence_count=int(payload["token_sequence_count"]),
        token_count=int(payload["token_count"]),
    )


def _load_tokenizer_resume_state(
    state_path: Path | None,
    *,
    source_path: Path,
    sequence_type: str,
    vocab_size: int,
    target_vocab_size: int,
    line_limit: int | None,
    allowed_special: set[str],
    id_typecode: str,
    progress_callback: TokenizerProgressCallback | None,
) -> dict[str, object] | None:
    if state_path is None or not state_path.exists():
        return None

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state_path.unlink(missing_ok=True)
        return None

    expected = {
        "version": 1,
        "source": _source_signature(source_path),
        "sequence_type": sequence_type,
        "vocab_size": vocab_size,
        "target_vocab_size": target_vocab_size,
        "line_limit": line_limit,
        "allowed_special": sorted(allowed_special),
        "id_typecode": id_typecode,
    }
    ignored_reason = ""
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            ignored_reason = f"{key}_mismatch"
            break

    raw_cache_path = payload.get("cache_path")
    cache_path = Path(raw_cache_path) if isinstance(raw_cache_path, str) and raw_cache_path else None
    if not ignored_reason and cache_path is None:
        ignored_reason = "cache_path_missing"
    if not ignored_reason and cache_path is not None and not cache_path.exists():
        ignored_reason = "cache_missing"

    if ignored_reason:
        _emit_progress(
            progress_callback,
            {
                "event": "tokenizer_resume_ignored",
                "path": str(state_path),
                "reason": ignored_reason,
            },
        )
        state_path.unlink(missing_ok=True)
        if cache_path is not None:
            cache_path.unlink(missing_ok=True)
        return None

    payload["cache_path"] = str(cache_path)
    return payload


def _write_tokenizer_resume_state(
    state_path: Path,
    *,
    source_path: Path,
    cache_path: Path,
    sequence_type: str,
    vocab_size: int,
    target_vocab_size: int,
    line_limit: int | None,
    allowed_special: set[str],
    id_typecode: str,
    stats: SequenceTokenizerTextTrainingStats,
    tokenizer: SequenceTokenizer,
    progress_callback: TokenizerProgressCallback | None,
) -> None:
    payload = {
        "version": 1,
        "source": _source_signature(source_path),
        "sequence_type": sequence_type,
        "vocab_size": vocab_size,
        "target_vocab_size": target_vocab_size,
        "line_limit": line_limit,
        "allowed_special": sorted(allowed_special),
        "id_typecode": id_typecode,
        "cache_path": str(cache_path),
        "cache_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
        "stats": _stats_to_resume_payload(stats),
        "tokenizer": json.loads(tokenizer.to_json()),
        "completed_merges": len(tokenizer.merge_ranks),
    }
    temp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        temp_path.replace(state_path)
    except PermissionError:
        state_path.unlink(missing_ok=True)
        temp_path.replace(state_path)
    _emit_progress(
        progress_callback,
        {
            "event": "tokenizer_checkpoint_saved",
            "path": str(state_path),
            "cache_path": str(cache_path),
            "cache_bytes": payload["cache_bytes"],
            "completed_merges": len(tokenizer.merge_ranks),
            "vocab_size": tokenizer.vocab_size,
            "target_vocab_size": target_vocab_size,
        },
    )


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


def _emit_progress(
    progress_callback: TokenizerProgressCallback | None,
    event: dict[str, object],
) -> None:
    if progress_callback is not None:
        progress_callback(event)

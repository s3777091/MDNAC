from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import torch
from .interfaces import FusedProfileSequenceBatch

@dataclass(slots=True, frozen=True)
class FusedVocabularyLayout:
    profile_vocab_size: int
    sequence_vocab_size: int
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    sep_token_id: int = 3
    profile_token_offset: int = 4
    sequence_token_offset: int | None = None

    def __post_init__(self) -> None:
        if self.profile_vocab_size <= 0:
            raise ValueError("profile_vocab_size must be greater than 0.")
        if self.sequence_vocab_size <= 0:
            raise ValueError("sequence_vocab_size must be greater than 0.")
        if self.profile_token_offset < 0:
            raise ValueError("profile_token_offset must be greater than or equal to 0.")
        if self.sequence_token_offset is not None and self.sequence_token_offset < 0:
            raise ValueError("sequence_token_offset must be greater than or equal to 0.")
        profile_range = range(self.profile_offset, self.profile_offset + self.profile_vocab_size)
        sequence_range = range(self.sequence_offset, self.sequence_offset + self.sequence_vocab_size)
        if _ranges_overlap(profile_range, sequence_range):
            raise ValueError("Profile and sequence token ranges must not overlap.")
        if self.sep_token_id in profile_range or self.sep_token_id in sequence_range:
            raise ValueError("sep_token_id must not overlap profile or sequence token ranges.")

    @property
    def profile_offset(self) -> int:
        return self.profile_token_offset

    @property
    def sequence_offset(self) -> int:
        if self.sequence_token_offset is not None:
            return self.sequence_token_offset
        return self.profile_offset + self.profile_vocab_size

    @property
    def vocab_size(self) -> int:
        return max(
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.sep_token_id,
            self.profile_offset + self.profile_vocab_size - 1,
            self.sequence_offset + self.sequence_vocab_size - 1,
        ) + 1

    @classmethod
    def from_raw_tensor_payload(cls, payload: Mapping[str, object]) -> "FusedVocabularyLayout":
        config = payload.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("Raw tensor payload must contain a config mapping.")
        return cls(
            profile_vocab_size=int(config["profile_vocab_size"]),
            sequence_vocab_size=int(config["sequence_vocab_size"]),
            pad_token_id=int(config.get("pad_token_id", 0)),
            bos_token_id=int(config.get("bos_token_id", 1)),
            eos_token_id=int(config.get("eos_token_id", 2)),
            sep_token_id=int(config.get("sep_token_id", 3)),
            profile_token_offset=int(config.get("profile_token_offset", 4)),
            sequence_token_offset=(
                int(config["sequence_token_offset"])
                if config.get("sequence_token_offset") is not None
                else None
            ),
        )

    def to_config_dict(self) -> dict[str, int]:
        return {
            "profile_vocab_size": self.profile_vocab_size,
            "sequence_vocab_size": self.sequence_vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "sep_token_id": self.sep_token_id,
            "profile_token_offset": self.profile_offset,
            "sequence_token_offset": self.sequence_offset,
        }

    def map_profile_token_id(self, token_id: int) -> int:
        normalized = int(token_id)
        if normalized < 0 or normalized >= self.profile_vocab_size:
            raise ValueError(
                f"Profile token id {normalized} is outside the configured vocabulary size "
                f"{self.profile_vocab_size}."
            )
        return self.profile_offset + normalized

    def map_sequence_token_id(self, token_id: int) -> int:
        normalized = int(token_id)
        if normalized < 0 or normalized >= self.sequence_vocab_size:
            raise ValueError(
                f"Sequence token id {normalized} is outside the configured vocabulary size "
                f"{self.sequence_vocab_size}."
        )
        return self.sequence_offset + normalized


def _ranges_overlap(left: range, right: range) -> bool:
    return left.start < right.stop and right.start < left.stop

@dataclass(slots=True, frozen=True)
class ProfileSequenceFusionConfig:
    add_global_bos: bool = True
    add_global_eos: bool = True
    insert_separator: bool = True
    trim_source_bos: bool = True
    trim_source_eos: bool = True
    source_bos_token_id: int = 1
    source_eos_token_id: int = 2

class ProfileSequenceBatchBuilder:
    def __init__(
        self,
        layout: FusedVocabularyLayout,
        config: ProfileSequenceFusionConfig | None = None,
    ) -> None:
        self.layout = layout
        self.config = config or ProfileSequenceFusionConfig()

    @classmethod
    def from_raw_tensor_payload(
        cls,
        payload: Mapping[str, object],
        config: ProfileSequenceFusionConfig | None = None,
    ) -> "ProfileSequenceBatchBuilder":
        return cls(
            layout=FusedVocabularyLayout.from_raw_tensor_payload(payload),
            config=config,
        )

    def build_from_raw_tensor_payload(
        self,
        payload: Mapping[str, object],
    ) -> FusedProfileSequenceBatch:
        metadata = payload.get("metadata")
        metadata_seq: Sequence[dict[str, object]] | None
        if isinstance(metadata, Sequence):
            metadata_seq = metadata  # type: ignore[assignment]
        else:
            metadata_seq = None

        return self.build(
            profile_input_ids=self._require_tensor(payload, "profile_input_ids"),
            profile_attention_mask=self._require_tensor(payload, "profile_attention_mask"),
            sequence_input_ids=self._require_tensor(payload, "sequence_input_ids"),
            sequence_attention_mask=self._require_tensor(payload, "sequence_attention_mask"),
            metadata=metadata_seq,
        )

    def build(
        self,
        profile_input_ids: torch.Tensor,
        profile_attention_mask: torch.Tensor,
        sequence_input_ids: torch.Tensor,
        sequence_attention_mask: torch.Tensor,
        metadata: Sequence[dict[str, object]] | None = None,
    ) -> FusedProfileSequenceBatch:
        self._validate_modalities(
            profile_input_ids=profile_input_ids,
            profile_attention_mask=profile_attention_mask,
            sequence_input_ids=sequence_input_ids,
            sequence_attention_mask=sequence_attention_mask,
        )

        fused_rows: list[list[int]] = []
        profile_spans: list[tuple[int, int]] = []
        separator_positions: list[int] = []
        sequence_spans: list[tuple[int, int]] = []

        batch_size = int(profile_input_ids.size(0))
        for row_idx in range(batch_size):
            profile_tokens = self._trim_source_boundaries(
                self._select_active_tokens(
                    input_ids=profile_input_ids[row_idx],
                    attention_mask=profile_attention_mask[row_idx],
                )
            )
            sequence_tokens = self._trim_source_boundaries(
                self._select_active_tokens(
                    input_ids=sequence_input_ids[row_idx],
                    attention_mask=sequence_attention_mask[row_idx],
                )
            )

            fused: list[int] = []
            if self.config.add_global_bos:
                fused.append(self.layout.bos_token_id)

            profile_start = len(fused)
            fused.extend(self.layout.map_profile_token_id(token_id) for token_id in profile_tokens)
            profile_end = len(fused)

            if self.config.insert_separator:
                separator_position = len(fused)
                fused.append(self.layout.sep_token_id)
            else:
                separator_position = profile_end

            sequence_start = len(fused)
            fused.extend(self.layout.map_sequence_token_id(token_id) for token_id in sequence_tokens)
            sequence_end = len(fused)

            if self.config.add_global_eos:
                fused.append(self.layout.eos_token_id)

            fused_rows.append(fused)
            profile_spans.append((profile_start, profile_end))
            separator_positions.append(separator_position)
            sequence_spans.append((sequence_start, sequence_end))

        token_ids, attention_mask = self._pad_fused_rows(fused_rows)

        return FusedProfileSequenceBatch(
            token_ids=token_ids,
            attention_mask=attention_mask,
            profile_spans=torch.tensor(profile_spans, dtype=torch.long),
            separator_positions=torch.tensor(separator_positions, dtype=torch.long),
            sequence_spans=torch.tensor(sequence_spans, dtype=torch.long),
            metadata=metadata,
        )

    @staticmethod
    def _require_tensor(payload: Mapping[str, object], key: str) -> torch.Tensor:
        value = payload.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected '{key}' to be a torch.Tensor in the raw tensor payload.")
        return value

    @staticmethod
    def _validate_modalities(
        profile_input_ids: torch.Tensor,
        profile_attention_mask: torch.Tensor,
        sequence_input_ids: torch.Tensor,
        sequence_attention_mask: torch.Tensor,
    ) -> None:
        shapes = {
            "profile_input_ids": tuple(profile_input_ids.shape),
            "profile_attention_mask": tuple(profile_attention_mask.shape),
            "sequence_input_ids": tuple(sequence_input_ids.shape),
            "sequence_attention_mask": tuple(sequence_attention_mask.shape),
        }
        if shapes["profile_input_ids"] != shapes["profile_attention_mask"]:
            raise ValueError("profile_input_ids and profile_attention_mask must have the same shape.")
        if shapes["sequence_input_ids"] != shapes["sequence_attention_mask"]:
            raise ValueError("sequence_input_ids and sequence_attention_mask must have the same shape.")
        if profile_input_ids.size(0) != sequence_input_ids.size(0):
            raise ValueError("Profile and sequence batches must have the same batch size.")

    @staticmethod
    def _select_active_tokens(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[int]:
        mask = attention_mask.to(dtype=torch.bool)
        return [int(token_id) for token_id in input_ids[mask].tolist()]

    def _trim_source_boundaries(self, token_ids: list[int]) -> list[int]:
        trimmed = list(token_ids)
        if self.config.trim_source_bos and trimmed and trimmed[0] == self.config.source_bos_token_id:
            trimmed = trimmed[1:]
        if self.config.trim_source_eos and trimmed and trimmed[-1] == self.config.source_eos_token_id:
            trimmed = trimmed[:-1]
        return trimmed

    def _pad_fused_rows(
        self,
        fused_rows: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not fused_rows:
            empty = torch.zeros((0, 0), dtype=torch.long)
            return empty, empty

        target_length = max(len(row) for row in fused_rows)
        token_ids = torch.full(
            (len(fused_rows), target_length),
            self.layout.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(fused_rows), target_length), dtype=torch.long)

        for row_idx, row in enumerate(fused_rows):
            row_tensor = torch.tensor(list(row), dtype=torch.long)
            length = row_tensor.size(0)
            token_ids[row_idx, :length] = row_tensor
            attention_mask[row_idx, :length] = 1

        return token_ids, attention_mask

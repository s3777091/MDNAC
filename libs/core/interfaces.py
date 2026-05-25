from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch


IGNORE_INDEX = -100

@dataclass(slots=True)
class CausalLMBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


@dataclass(slots=True)
class FusedProfileSequenceBatch:
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    profile_spans: torch.Tensor
    separator_positions: torch.Tensor
    sequence_spans: torch.Tensor
    metadata: Sequence[dict[str, object]] | None = None

    def to_causal_lm_batch(
        self,
        ignore_index: int = IGNORE_INDEX,
        train_on_prompt: bool = False,
        include_separator_in_loss: bool = False,
    ) -> CausalLMBatch:
        if self.token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, seq_len).")
        if self.token_ids.shape != self.attention_mask.shape:
            raise ValueError("token_ids and attention_mask must have identical shapes.")
        if self.token_ids.size(1) < 2:
            raise ValueError("Need at least 2 tokens to build a causal language-model batch.")

        input_ids = self.token_ids[:, :-1].contiguous()
        attention_mask = self.attention_mask[:, :-1].contiguous()
        labels = self.token_ids[:, 1:].clone().contiguous()

        valid_targets = self.attention_mask[:, 1:].bool()

        if not train_on_prompt:
            target_positions = (
                torch.arange(1, self.token_ids.size(1), device=self.token_ids.device)
                .unsqueeze(0)
                .expand(labels.size(0), -1)
            )
            loss_start = (
                self.separator_positions
                if include_separator_in_loss
                else self.sequence_spans[:, 0]
            ).unsqueeze(1)
            valid_targets = valid_targets & (target_positions >= loss_start)

        labels = labels.masked_fill(~valid_targets, ignore_index)
        return CausalLMBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

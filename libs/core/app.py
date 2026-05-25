from __future__ import annotations

from typing import Mapping
import torch
import torch.nn as nn
from .fusion import (
    FusedVocabularyLayout,
    ProfileSequenceBatchBuilder,
    ProfileSequenceFusionConfig,
)
from .interfaces import CausalLMBatch, FusedProfileSequenceBatch
from .mdc import MDCDecoderModel, MDCModelConfig

class MicrobialDecoderCoreApp(nn.Module):
    def __init__(
        self,
        model_config: MDCModelConfig,
        layout: FusedVocabularyLayout,
        fusion_config: ProfileSequenceFusionConfig | None = None,
    ) -> None:
        super().__init__()
        if model_config.vocab_size != layout.vocab_size:
            raise ValueError(
                f"Model vocab_size ({model_config.vocab_size}) must match fused layout vocab_size "
                f"({layout.vocab_size})."
            )

        self.layout = layout
        self.batch_builder = ProfileSequenceBatchBuilder(layout=layout, config=fusion_config)
        self.model = MDCDecoderModel(model_config)

    @classmethod
    def from_raw_tensor_payload(
        cls,
        payload: Mapping[str, object],
        model_config: MDCModelConfig,
        fusion_config: ProfileSequenceFusionConfig | None = None,
    ) -> "MicrobialDecoderCoreApp":
        layout = FusedVocabularyLayout.from_raw_tensor_payload(payload)
        resolved_config = (
            model_config
            if model_config.vocab_size == layout.vocab_size
            else model_config.with_vocab_size(layout.vocab_size)
        )
        return cls(
            model_config=resolved_config,
            layout=layout,
            fusion_config=fusion_config,
        )

    def prepare_batch(
        self,
        payload: Mapping[str, object],
    ) -> FusedProfileSequenceBatch:
        batch = self.batch_builder.build_from_raw_tensor_payload(payload)
        if batch.token_ids.size(1) > self.model.cfg["context_length"]:
            raise ValueError(
                f"Prepared sequence length {batch.token_ids.size(1)} exceeds model context length "
                f"{self.model.cfg['context_length']}."
            )
        return batch

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor:
        return self.model(
            token_ids,
            attn_mask=attention_mask,
            return_hidden_states=return_hidden_states,
        )

    def prepare_causal_lm_batch(
        self,
        payload: Mapping[str, object],
        *,
        train_on_prompt: bool = False,
        include_separator_in_loss: bool = False,
    ) -> CausalLMBatch:
        fused_batch = self.prepare_batch(payload)
        return fused_batch.to_causal_lm_batch(
            train_on_prompt=train_on_prompt,
            include_separator_in_loss=include_separator_in_loss,
        )

    def forward_causal_lm_batch(
        self,
        batch: CausalLMBatch,
        return_hidden_states: bool = False,
    ) -> torch.Tensor:
        return self.forward(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            return_hidden_states=return_hidden_states,
        )

    def forward_from_raw_tensor_payload(
        self,
        payload: Mapping[str, object],
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, FusedProfileSequenceBatch]:
        batch = self.prepare_batch(payload)
        logits = self.forward(
            batch.token_ids,
            attention_mask=batch.attention_mask,
            return_hidden_states=return_hidden_states,
        )
        return logits, batch

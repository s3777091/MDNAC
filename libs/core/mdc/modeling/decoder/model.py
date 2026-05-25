from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from ...config import MDCModelConfig
from ..blocks.transformer import TransformerBlock
from ..cache.decoder_cache import MDCDecoderCache
from ..components.normalization import RMSNorm
from ..components.rope import compute_rope_params


def _normalize_cfg(cfg: MDCModelConfig | Mapping[str, object]) -> dict[str, object]:
    if isinstance(cfg, MDCModelConfig):
        return cfg.to_dict()
    return dict(cfg)


class MDCDecoderModel(nn.Module):
    def __init__(self, cfg: MDCModelConfig | Mapping[str, object]) -> None:
        super().__init__()
        normalized_cfg = _normalize_cfg(cfg)
        self.cfg = normalized_cfg

        self.tok_emb = nn.Embedding(
            int(normalized_cfg["vocab_size"]),
            int(normalized_cfg["emb_dim"]),
            dtype=normalized_cfg["dtype"],
        )

        layer_types = normalized_cfg.get("layer_types", ["full_attention"] * int(normalized_cfg["n_layers"]))
        if len(layer_types) != int(normalized_cfg["n_layers"]):
            raise ValueError("len(layer_types) must equal n_layers")

        self.trf_blocks = nn.ModuleList(
            [
                TransformerBlock(normalized_cfg, layer_type, idx)
                for idx, layer_type in enumerate(layer_types)
            ]
        )
        self.final_norm = RMSNorm(
            int(normalized_cfg["emb_dim"]),
            eps=float(normalized_cfg.get("rms_norm_eps", 1e-6)),
        )
        self.out_head = nn.Linear(
            int(normalized_cfg["emb_dim"]),
            int(normalized_cfg["vocab_size"]),
            bias=False,
            dtype=normalized_cfg["dtype"],
        )

        cos, sin = compute_rope_params(
            head_dim=int(normalized_cfg["head_dim"]),
            theta_base=float(normalized_cfg["rope_base"]),
            context_length=int(normalized_cfg["context_length"]),
            partial_rotary_factor=float(normalized_cfg.get("partial_rotary_factor", 1.0)),
            dtype=torch.float32,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.current_pos = 0

    def create_mask(
        self,
        cur_len: int,
        device: torch.device,
        pos_start: int = 0,
        pos_end: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pos_end is None:
            pos_end = cur_len

        ones = torch.ones((pos_end, pos_end), device=device, dtype=torch.bool)
        mask_full = torch.triu(ones, diagonal=1)
        row_slice = slice(pos_start, pos_end)
        mask = mask_full[row_slice, :pos_end][None, None, :, :]

        if attn_mask is None:
            return mask

        key_padding_mask = (~attn_mask[:, :pos_end]).view(attn_mask.shape[0], 1, 1, pos_end)
        return mask | key_padding_mask

    def forward(
        self,
        in_idx: torch.Tensor,
        cache: MDCDecoderCache | None = None,
        attn_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor:
        x = self.tok_emb(in_idx)

        if attn_mask is not None:
            attn_mask = attn_mask.to(device=x.device, dtype=torch.bool)

        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = self.create_mask(
                cur_len=num_tokens,
                device=x.device,
                pos_start=pos_start,
                pos_end=pos_end,
                attn_mask=attn_mask,
            )
            cache_position = torch.arange(pos_start, pos_end, device=x.device, dtype=torch.long)
        else:
            pos_start = 0
            mask = self.create_mask(
                cur_len=num_tokens,
                device=x.device,
                pos_start=0,
                pos_end=num_tokens,
                attn_mask=attn_mask,
            )
            cache_position = None

        if attn_mask is not None:
            qmask = attn_mask[:, pos_start : pos_start + num_tokens].unsqueeze(-1)
            x = x * qmask.to(x.dtype)

        for index, block in enumerate(self.trf_blocks):
            block_cache = cache.get(index) if cache is not None else None
            x, new_block_cache = block(
                x,
                mask=mask,
                cos=self.cos,
                sin=self.sin,
                start_pos=pos_start,
                cache=block_cache,
                linear_cache=cache.linear_cache if cache is not None else None,
                cache_position=cache_position,
                attention_mask=(
                    attn_mask[:, pos_start : pos_start + num_tokens]
                    if attn_mask is not None
                    else None
                ),
            )
            if cache is not None and new_block_cache is not None:
                cache.update(index, new_block_cache)

        if cache is not None:
            cache.linear_cache.has_previous_state = True

        x = self.final_norm(x)
        if return_hidden_states:
            return x
        return self.out_head(x.to(self.cfg["dtype"]))

    def create_kv_cache(self) -> MDCDecoderCache:
        return MDCDecoderCache(n_layers=int(self.cfg["n_layers"]))

    def reset_kv_cache(self, cache: MDCDecoderCache | None = None) -> None:
        self.current_pos = 0
        if cache is not None:
            cache.reset()

from __future__ import annotations

from collections.abc import Mapping
import torch
import torch.nn as nn
from ..components.feed_forward import FeedForward
from ..components.normalization import RMSNorm
from ..factories.mixers import build_token_mixer
from ..interfaces.token_mixer import TokenMixerBuilder

class TransformerBlock(nn.Module):
    def __init__(
        self,
        cfg: Mapping[str, object],
        layer_type: str,
        layer_idx: int,
        token_mixer_builder: TokenMixerBuilder = build_token_mixer,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.token_mixer = token_mixer_builder(cfg, layer_type, layer_idx)
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(int(cfg["emb_dim"]), eps=float(cfg.get("rms_norm_eps", 1e-6)))
        self.norm2 = RMSNorm(int(cfg["emb_dim"]), eps=float(cfg.get("rms_norm_eps", 1e-6)))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache=None,
        linear_cache=None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        shortcut = x
        x = self.norm1(x)

        if self.layer_type == "full_attention":
            x, next_cache = self.token_mixer(
                x,
                mask,
                cos,
                sin,
                start_pos=start_pos,
                cache=cache,
            )
        else:
            x = self.token_mixer(
                x,
                cache_params=linear_cache,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
            next_cache = None

        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x, next_cache

from __future__ import annotations

from collections.abc import Mapping
import torch.nn as nn
from ...linear_attention import MDCGatedDeltaNet
from ..attention.grouped_query import GroupedQueryAttention


class MDCConfigAdapter:
    def __init__(self, cfg: Mapping[str, object]) -> None:
        self.hidden_size = int(cfg["emb_dim"])
        self.linear_num_value_heads = int(cfg["linear_num_value_heads"])
        self.linear_num_key_heads = int(cfg["linear_num_key_heads"])
        self.linear_key_head_dim = int(cfg["linear_key_head_dim"])
        self.linear_value_head_dim = int(cfg["linear_value_head_dim"])
        self.linear_conv_kernel_dim = int(cfg["linear_conv_kernel_dim"])
        self.hidden_act = "silu"
        self.rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.dtype = cfg.get("dtype", None)


def build_token_mixer(cfg: Mapping[str, object], layer_type: str, layer_idx: int) -> nn.Module:
    if layer_type == "full_attention":
        return GroupedQueryAttention(
            d_in=int(cfg["emb_dim"]),
            num_heads=int(cfg["n_heads"]),
            head_dim=int(cfg["head_dim"]),
            num_kv_groups=int(cfg["n_kv_groups"]),
            qk_norm=bool(cfg["qk_norm"]),
            dtype=cfg["dtype"],
        )
    if layer_type == "linear_attention":
        return MDCGatedDeltaNet(MDCConfigAdapter(cfg), layer_idx)
    raise ValueError(f"Unsupported layer type: {layer_type}")

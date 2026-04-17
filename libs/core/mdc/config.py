from __future__ import annotations
from dataclasses import dataclass, replace
import torch

DEFAULT_MDC_LAYER_TYPES = (
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
)


def build_default_mdc_layer_types(n_layers: int) -> tuple[str, ...]:
    if n_layers <= 0:
        raise ValueError("n_layers must be greater than 0.")

    base_pattern = ("linear_attention", "linear_attention", "linear_attention", "full_attention")
    repeated = (base_pattern * ((n_layers + len(base_pattern) - 1) // len(base_pattern)))[:n_layers]
    return tuple(repeated)


@dataclass(slots=True, frozen=True)
class MDCModelConfig:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    hidden_dim: int
    head_dim: int
    qk_norm: bool
    n_kv_groups: int
    rope_base: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    dtype: torch.dtype = torch.float32
    layer_types: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0.")
        if self.context_length <= 0:
            raise ValueError("context_length must be greater than 0.")
        if self.emb_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("emb_dim and hidden_dim must be greater than 0.")
        if self.n_heads <= 0 or self.n_layers <= 0:
            raise ValueError("n_heads and n_layers must be greater than 0.")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be greater than 0.")
        if self.layer_types is not None and len(self.layer_types) != self.n_layers:
            raise ValueError("len(layer_types) must equal n_layers.")

    def to_dict(self) -> dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "emb_dim": self.emb_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "head_dim": self.head_dim,
            "qk_norm": self.qk_norm,
            "n_kv_groups": self.n_kv_groups,
            "rope_base": self.rope_base,
            "partial_rotary_factor": self.partial_rotary_factor,
            "rms_norm_eps": self.rms_norm_eps,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "dtype": self.dtype,
            "layer_types": list(self.layer_types or build_default_mdc_layer_types(self.n_layers)),
        }

    def with_vocab_size(self, vocab_size: int) -> "MDCModelConfig":
        return replace(self, vocab_size=vocab_size)


def build_mdc_default_config(
    vocab_size: int,
    context_length: int,
    dtype: torch.dtype = torch.bfloat16,
) -> MDCModelConfig:
    return MDCModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        emb_dim=1_024,
        n_heads=8,
        n_layers=24,
        hidden_dim=3_584,
        head_dim=256,
        qk_norm=True,
        n_kv_groups=2,
        dtype=dtype,
    )


def build_mdc_tiny_config(
    vocab_size: int,
    context_length: int = 128,
    dtype: torch.dtype = torch.float32,
) -> MDCModelConfig:
    return MDCModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        emb_dim=64,
        n_heads=4,
        n_layers=2,
        hidden_dim=128,
        head_dim=16,
        qk_norm=False,
        n_kv_groups=2,
        rope_base=10_000.0,
        partial_rotary_factor=1.0,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        dtype=dtype,
        layer_types=("linear_attention", "full_attention"),
    )

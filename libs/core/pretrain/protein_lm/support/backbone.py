from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

import torch

from libs.core.mdc.config import MDCModelConfig, build_default_mdc_layer_types
from models.qwen3_5 import QWEN3_5_CONFIG_0_8B, QWEN3_5_CONFIG_2B


QWEN3_5_BACKBONE_FAMILY = "qwen3_5"
QWEN3_5_PROTEIN_MODEL_FAMILY = "qwen3_5_protein_lm"
LEGACY_PROTEIN_MODEL_FAMILY = "mdc_protein_lm"
DEFAULT_QWEN3_5_MODEL_NAME = "Qwen/Qwen3.5-0.8B"

QWEN3_5_MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "Qwen/Qwen3.5-0.8B": deepcopy(QWEN3_5_CONFIG_0_8B),
    "Qwen/Qwen3.5-2B": deepcopy(QWEN3_5_CONFIG_2B),
}

QWEN3_5_MODEL_ALIASES: dict[str, str] = {
    "0.8B": "Qwen/Qwen3.5-0.8B",
    "0.8b": "Qwen/Qwen3.5-0.8B",
    "2B": "Qwen/Qwen3.5-2B",
    "2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
    "Qwen3.5-2B": "Qwen/Qwen3.5-2B",
}


def build_qwen3_5_config(
    model_name: str = DEFAULT_QWEN3_5_MODEL_NAME,
    *,
    vocab_size: int,
    context_length: int | None = None,
    dtype: torch.dtype | None = None,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    resolved_name = QWEN3_5_MODEL_ALIASES.get(model_name, model_name)
    if resolved_name not in QWEN3_5_MODEL_CONFIGS:
        available = ", ".join(sorted(QWEN3_5_MODEL_CONFIGS))
        raise ValueError(f"Unknown Qwen3.5 config '{model_name}'. Available: {available}")

    resolved_overrides = dict(overrides or {})
    config = deepcopy(QWEN3_5_MODEL_CONFIGS[resolved_name])
    config["vocab_size"] = int(vocab_size)
    if context_length is not None:
        config["context_length"] = int(context_length)
    if dtype is not None:
        config["dtype"] = dtype
    config.update(resolved_overrides)

    n_layers = int(config["n_layers"])
    explicit_layer_types = "layer_types" in resolved_overrides
    config["layer_types"] = _resolve_qwen_layer_types(
        config.get("layer_types"),
        n_layers=n_layers,
        allow_resize=not explicit_layer_types,
    )
    config["model_name"] = resolved_name
    config["config_source"] = "models.qwen3_5.modeling"
    config["backbone_family"] = QWEN3_5_BACKBONE_FAMILY
    return config


def build_mdc_config_from_qwen3_5_config(
    config: Mapping[str, object],
    *,
    dtype: torch.dtype | None = None,
    attention_pattern: str | Sequence[str] = "as_config",
) -> MDCModelConfig:
    emb_dim = int(config["emb_dim"])
    n_heads = int(config["n_heads"])
    n_layers = int(config["n_layers"])
    if emb_dim % n_heads != 0:
        raise ValueError("emb_dim must be divisible by n_heads.")

    if attention_pattern == "as_config":
        layer_types = _resolve_qwen_layer_types(config.get("layer_types"), n_layers=n_layers)
    elif attention_pattern == "qwen_hybrid":
        layer_types = build_default_mdc_layer_types(n_layers)
    elif attention_pattern == "full_attention":
        layer_types = ("full_attention",) * n_layers
    elif isinstance(attention_pattern, Sequence) and not isinstance(attention_pattern, str):
        layer_types = tuple(str(layer_type) for layer_type in attention_pattern)
    else:
        raise ValueError("attention_pattern must be 'as_config', 'qwen_hybrid', 'full_attention', or a sequence.")

    resolved_dtype = dtype if dtype is not None else _coerce_torch_dtype(config.get("dtype", torch.float32))
    head_dim = int(config.get("head_dim", emb_dim // n_heads))
    return MDCModelConfig(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        hidden_dim=int(config.get("hidden_dim", 4 * emb_dim)),
        head_dim=head_dim,
        qk_norm=bool(config.get("qk_norm", True)),
        n_kv_groups=int(config.get("n_kv_groups", n_heads)),
        rope_base=float(config.get("rope_base", 10_000.0)),
        partial_rotary_factor=float(config.get("partial_rotary_factor", 1.0)),
        rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
        linear_conv_kernel_dim=int(config.get("linear_conv_kernel_dim", 4)),
        linear_key_head_dim=int(config.get("linear_key_head_dim", head_dim)),
        linear_value_head_dim=int(config.get("linear_value_head_dim", head_dim)),
        linear_num_key_heads=int(config.get("linear_num_key_heads", n_heads)),
        linear_num_value_heads=int(config.get("linear_num_value_heads", n_heads)),
        dtype=resolved_dtype,
        layer_types=layer_types,
    )


def is_supported_protein_checkpoint_family(model_family: object) -> bool:
    return model_family in {
        None,
        LEGACY_PROTEIN_MODEL_FAMILY,
        QWEN3_5_PROTEIN_MODEL_FAMILY,
    }


def extract_protein_backbone_config(checkpoint: Mapping[str, object]) -> Mapping[str, object] | None:
    qwen_config = checkpoint.get("qwen3_5_config")
    if isinstance(qwen_config, Mapping):
        return dict(qwen_config)

    legacy_config = checkpoint.get("llms_from_scratch_config")
    if isinstance(legacy_config, Mapping):
        return dict(legacy_config)

    return None


def _resolve_qwen_layer_types(
    layer_types: object,
    *,
    n_layers: int,
    allow_resize: bool = True,
) -> tuple[str, ...]:
    if layer_types is None:
        return build_default_mdc_layer_types(n_layers)

    resolved = tuple(str(layer_type) for layer_type in layer_types)
    if len(resolved) == n_layers:
        return resolved
    if allow_resize:
        return build_default_mdc_layer_types(n_layers)
    raise ValueError("len(layer_types) must equal n_layers.")


def _coerce_torch_dtype(value: object) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value

    normalized = str(value).strip()
    if normalized.startswith("torch."):
        normalized = normalized.removeprefix("torch.")

    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
        "float64": torch.float64,
        "double": torch.float64,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {value!r}")
    return dtype_map[normalized]

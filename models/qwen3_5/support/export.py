from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from ..modeling import Qwen3_5Tokenizer
from .hf_utils import DEFAULT_QWEN_REPO_ID, resolve_qwen_tokenizer_json

if TYPE_CHECKING:
    from transformers import GenerationConfig, Qwen3_5TextConfig


HFWeightsFormat = Literal["safetensors", "pytorch_bin"]

DEFAULT_OLLAMA_TEMPLATE = """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

DEFAULT_OLLAMA_STOP_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
)


def load_checkpoint(checkpoint_path: str | Path, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    return torch.load(Path(checkpoint_path), map_location=map_location)


@dataclass(frozen=True)
class ExportedQwen3_5HfArtifact:
    checkpoint_path: Path
    output_dir: Path
    model_dir: Path
    weights_path: Path
    config_path: Path
    generation_config_path: Path
    tokenizer_path: Path
    tokenizer_config_path: Path
    special_tokens_map_path: Path
    metadata_path: Path
    tokenizer_repo_id: str
    weights_format: HFWeightsFormat


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype | torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _require_qwen3_5_checkpoint(checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    if "model_config" not in checkpoint:
        raise ValueError(f"Checkpoint is missing `model_config`: {checkpoint_path}")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint is missing `model_state_dict`: {checkpoint_path}")

    model_family = checkpoint.get("model_family")
    if model_family is not None and model_family != "qwen3_5":
        raise ValueError(
            "Only qwen3_5 checkpoints can be exported for Ollama hosting in this repo. "
            f"Received: {model_family!r}"
        )

    model_config = checkpoint["model_config"]
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint `model_config` must be a dictionary.")

    layer_types = list(model_config.get("layer_types", []))
    if not layer_types:
        raise ValueError(
            "Checkpoint `model_config.layer_types` is required for Qwen3.5 export."
        )
    if "linear_attention" not in layer_types:
        raise ValueError(
            "The exported Ollama path expects the official hybrid Qwen3.5 architecture, "
            "which includes at least one `linear_attention` layer."
        )
    if not bool(model_config.get("qk_norm", True)):
        raise ValueError(
            "This exporter only supports checkpoints that keep Qwen3.5 `qk_norm=True`."
        )


def _require_transformers_export_dependencies() -> tuple[type["GenerationConfig"], type["Qwen3_5TextConfig"]]:
    try:
        from transformers import GenerationConfig, Qwen3_5TextConfig
    except ImportError as exc:
        raise RuntimeError(
            "The `transformers` package is required for `build_hf_text_config()` and "
            "strict Hugging Face verification helpers. Plain HF/Ollama artifact export "
            "can run without it."
        ) from exc
    return GenerationConfig, Qwen3_5TextConfig


def resolve_checkpoint_tokenizer(
    checkpoint: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> tuple[Path, str]:
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    repo_id = str(checkpoint.get("tokenizer_repo_id") or DEFAULT_QWEN_REPO_ID)

    if tokenizer_path is not None:
        resolved = Path(tokenizer_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {resolved}")
        return resolved, repo_id

    raw_checkpoint_tokenizer = checkpoint.get("tokenizer_file_path")
    if raw_checkpoint_tokenizer:
        resolved = Path(str(raw_checkpoint_tokenizer)).expanduser()
        if not resolved.is_absolute():
            resolved = (checkpoint_file.parent / resolved).resolve()
        else:
            resolved = resolved.resolve()
        if resolved.is_file():
            return resolved, repo_id

    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    resolved = resolve_qwen_tokenizer_json(
        root,
        repo_id=repo_id,
        local_files_only=True,
        allow_download=False,
        extra_sources=[checkpoint_file.parent],
    )
    return resolved.resolve(), repo_id


def load_export_tokenizer(tokenizer_path: str | Path, *, repo_id: str) -> Qwen3_5Tokenizer:
    return Qwen3_5Tokenizer(
        tokenizer_file_path=str(Path(tokenizer_path).resolve()),
        repo_id=repo_id,
        apply_chat_template=False,
        add_generation_prompt=False,
    )


def _build_hf_text_config_kwargs(
    model_config: dict[str, Any],
    *,
    pad_token_id: int,
    eos_token_id: int,
) -> dict[str, Any]:
    n_layers = int(model_config["n_layers"])
    layer_types = list(model_config.get("layer_types", ["full_attention"] * n_layers))
    if len(layer_types) != n_layers:
        raise ValueError("len(layer_types) must equal n_layers for Qwen3.5 export.")

    return {
        "vocab_size": int(model_config["vocab_size"]),
        "hidden_size": int(model_config["emb_dim"]),
        "intermediate_size": int(model_config["hidden_dim"]),
        "num_hidden_layers": n_layers,
        "num_attention_heads": int(model_config["n_heads"]),
        "num_key_value_heads": int(model_config["n_kv_groups"]),
        "head_dim": int(model_config["head_dim"]),
        "max_position_embeddings": int(model_config["context_length"]),
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "rms_norm_eps": float(model_config.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(model_config["rope_base"]),
        "partial_rotary_factor": float(model_config.get("partial_rotary_factor", 1.0)),
        "linear_conv_kernel_dim": int(model_config["linear_conv_kernel_dim"]),
        "linear_key_head_dim": int(model_config["linear_key_head_dim"]),
        "linear_value_head_dim": int(model_config["linear_value_head_dim"]),
        "linear_num_key_heads": int(model_config["linear_num_key_heads"]),
        "linear_num_value_heads": int(model_config["linear_num_value_heads"]),
        "layer_types": layer_types,
        "tie_word_embeddings": False,
        "use_cache": True,
        "pad_token_id": int(pad_token_id),
        "eos_token_id": int(eos_token_id),
    }


def _serialize_torch_dtype(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    normalized = str(value).strip()
    if normalized.startswith("torch."):
        normalized = normalized.removeprefix("torch.")
    return normalized or None


def build_hf_text_config_payload(
    model_config: dict[str, Any],
    *,
    pad_token_id: int,
    eos_token_id: int,
) -> dict[str, Any]:
    payload = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "bos_token_id": None,
        "model_type": "qwen3_5_text",
        **_build_hf_text_config_kwargs(
            model_config,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        ),
    }
    torch_dtype = _serialize_torch_dtype(model_config.get("dtype"))
    if torch_dtype is not None:
        payload["torch_dtype"] = torch_dtype
    return payload


def build_generation_config_payload(
    *,
    pad_token_id: int,
    eos_token_id: int,
) -> dict[str, Any]:
    return {
        "bos_token_id": None,
        "pad_token_id": int(pad_token_id),
        "eos_token_id": int(eos_token_id),
    }


def build_hf_text_config(
    model_config: dict[str, Any],
    *,
    pad_token_id: int,
    eos_token_id: int,
) -> Qwen3_5TextConfig:
    _, qwen3_5_text_config_cls = _require_transformers_export_dependencies()
    return qwen3_5_text_config_cls(
        **_build_hf_text_config_kwargs(
            model_config,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    )


def convert_repo_state_dict_to_hf(
    model_state_dict: dict[str, torch.Tensor],
    model_config: dict[str, Any],
) -> OrderedDict[str, torch.Tensor]:
    layer_types = list(model_config.get("layer_types", []))
    n_layers = int(model_config["n_layers"])
    if len(layer_types) != n_layers:
        raise ValueError("len(layer_types) must equal n_layers for Qwen3.5 export.")

    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    converted["model.embed_tokens.weight"] = model_state_dict["tok_emb.weight"].detach().cpu().contiguous()

    for layer_idx, layer_type in enumerate(layer_types):
        repo_prefix = f"trf_blocks.{layer_idx}"
        hf_prefix = f"model.layers.{layer_idx}"

        if layer_type == "full_attention":
            full_attention_map = {
                "self_attn.q_proj.weight": "token_mixer.W_query.weight",
                "self_attn.k_proj.weight": "token_mixer.W_key.weight",
                "self_attn.v_proj.weight": "token_mixer.W_value.weight",
                "self_attn.o_proj.weight": "token_mixer.out_proj.weight",
                "self_attn.q_norm.weight": "token_mixer.q_norm.weight",
                "self_attn.k_norm.weight": "token_mixer.k_norm.weight",
            }
            for hf_suffix, repo_suffix in full_attention_map.items():
                repo_name = f"{repo_prefix}.{repo_suffix}"
                if repo_name not in model_state_dict:
                    raise KeyError(f"Missing tensor for Qwen3.5 export: {repo_name}")
                converted[f"{hf_prefix}.{hf_suffix}"] = (
                    model_state_dict[repo_name].detach().cpu().contiguous()
                )
        elif layer_type == "linear_attention":
            linear_attention_suffixes = (
                "dt_bias",
                "A_log",
                "conv1d.weight",
                "norm.weight",
                "out_proj.weight",
                "in_proj_qkv.weight",
                "in_proj_z.weight",
                "in_proj_b.weight",
                "in_proj_a.weight",
            )
            for suffix in linear_attention_suffixes:
                repo_name = f"{repo_prefix}.token_mixer.{suffix}"
                if repo_name not in model_state_dict:
                    raise KeyError(f"Missing tensor for Qwen3.5 export: {repo_name}")
                converted[f"{hf_prefix}.linear_attn.{suffix}"] = (
                    model_state_dict[repo_name].detach().cpu().contiguous()
                )
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {layer_type!r}")

        feed_forward_map = {
            "mlp.gate_proj.weight": "ff.fc1.weight",
            "mlp.up_proj.weight": "ff.fc2.weight",
            "mlp.down_proj.weight": "ff.fc3.weight",
            "input_layernorm.weight": "norm1.weight",
            "post_attention_layernorm.weight": "norm2.weight",
        }
        for hf_suffix, repo_suffix in feed_forward_map.items():
            repo_name = f"{repo_prefix}.{repo_suffix}"
            if repo_name not in model_state_dict:
                raise KeyError(f"Missing tensor for Qwen3.5 export: {repo_name}")
            converted[f"{hf_prefix}.{hf_suffix}"] = (
                model_state_dict[repo_name].detach().cpu().contiguous()
            )

    converted["model.norm.weight"] = model_state_dict["final_norm.weight"].detach().cpu().contiguous()
    converted["lm_head.weight"] = model_state_dict["out_head.weight"].detach().cpu().contiguous()
    return converted


def write_hf_tokenizer_files(
    output_dir: str | Path,
    *,
    source_tokenizer_path: str | Path,
    tokenizer: Qwen3_5Tokenizer,
    context_length: int,
) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer_path = output_path / "tokenizer.json"
    shutil.copy2(Path(source_tokenizer_path).resolve(), tokenizer_path)

    special_by_id = {token_id: token for token, token_id in tokenizer._special_to_id.items()}
    pad_token = special_by_id.get(tokenizer.pad_token_id, "<|endoftext|>")
    eos_token = special_by_id.get(tokenizer.eos_token_id, pad_token)

    special_tokens_map_path = output_path / "special_tokens_map.json"
    special_tokens_map_path.write_text(
        json.dumps(
            {
                "eos_token": eos_token,
                "pad_token": pad_token,
                "additional_special_tokens": [
                    token
                    for token in Qwen3_5Tokenizer._SPECIALS
                    if token not in {pad_token, eos_token}
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tokenizer_config_path = output_path / "tokenizer_config.json"
    tokenizer_config_path.write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2TokenizerFast",
                "tokenizer_file": str(tokenizer_path),
                "model_max_length": int(context_length),
                "padding_side": "left",
                "clean_up_tokenization_spaces": False,
                "eos_token": eos_token,
                "pad_token": pad_token,
                "additional_special_tokens": [
                    token
                    for token in Qwen3_5Tokenizer._SPECIALS
                    if token not in {pad_token, eos_token}
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return tokenizer_path, tokenizer_config_path, special_tokens_map_path


def build_default_ollama_system_prompt(checkpoint: dict[str, Any]) -> str:
    settings = checkpoint.get("instruction_settings") or {}
    assistant_name = str(settings.get("assistant_name") or checkpoint.get("assistant_name") or "Ava")
    company_name = str(settings.get("company_name") or checkpoint.get("company_name") or "").strip()
    company_suffix = f" for {company_name}" if company_name else ""
    return (
        f"You are {assistant_name}, a helpful product support assistant{company_suffix}. "
        "Answer using the provided product information when it is available. "
        "If the answer is not supported by the provided information, say you do not know."
    )


def export_checkpoint_to_hf_directory(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
    project_root: str | Path | None = None,
    weights_format: HFWeightsFormat = "safetensors",
) -> ExportedQwen3_5HfArtifact:
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_file, torch.device("cpu"))
    _require_qwen3_5_checkpoint(checkpoint, checkpoint_file)

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_tokenizer_path, tokenizer_repo_id = resolve_checkpoint_tokenizer(
        checkpoint,
        checkpoint_file,
        tokenizer_path=tokenizer_path,
        project_root=project_root,
    )
    tokenizer = load_export_tokenizer(resolved_tokenizer_path, repo_id=tokenizer_repo_id)
    model_config = checkpoint["model_config"]

    config_path = output_path / "config.json"
    config_path.write_text(
        json.dumps(
            build_hf_text_config_payload(
                model_config,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    generation_config_path = output_path / "generation_config.json"
    generation_config_path.write_text(
        json.dumps(
            build_generation_config_payload(
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tokenizer_output_path, tokenizer_config_path, special_tokens_map_path = write_hf_tokenizer_files(
        output_path,
        source_tokenizer_path=resolved_tokenizer_path,
        tokenizer=tokenizer,
        context_length=int(model_config["context_length"]),
    )

    converted_state_dict = convert_repo_state_dict_to_hf(
        checkpoint["model_state_dict"],
        model_config,
    )

    if weights_format == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise RuntimeError(
                "The `safetensors` package is required for `weights_format='safetensors'`."
            ) from exc
        weights_path = output_path / "model.safetensors"
        save_file(dict(converted_state_dict), str(weights_path), metadata={"format": "pt"})
    elif weights_format == "pytorch_bin":
        weights_path = output_path / "pytorch_model.bin"
        torch.save(dict(converted_state_dict), weights_path)
    else:
        raise ValueError(f"Unsupported HF export weights format: {weights_format!r}")

    metadata_path = output_path / "export_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "artifact_format": "hf_qwen3_5",
                "source_checkpoint_path": str(checkpoint_file),
                "model_family": "qwen3_5",
                "weights_format": weights_format,
                "tokenizer_repo_id": tokenizer_repo_id,
                "tokenizer_path": str(tokenizer_output_path),
                "hf_config_path": str(config_path),
                "generation_config_path": str(generation_config_path),
                "weights_path": str(weights_path),
                "special_tokens_map_path": str(special_tokens_map_path),
                "tokenizer_config_path": str(tokenizer_config_path),
                "model_config": model_config,
                "instruction_settings": checkpoint.get("instruction_settings"),
                "chatbot_settings": checkpoint.get("chatbot_settings"),
                "inference_tokenizer_settings": checkpoint.get("inference_tokenizer_settings"),
            },
            indent=2,
            ensure_ascii=False,
            default=_json_safe,
        ),
        encoding="utf-8",
    )

    return ExportedQwen3_5HfArtifact(
        checkpoint_path=checkpoint_file,
        output_dir=output_path,
        model_dir=output_path,
        weights_path=weights_path,
        config_path=config_path,
        generation_config_path=generation_config_path,
        tokenizer_path=tokenizer_output_path,
        tokenizer_config_path=tokenizer_config_path,
        special_tokens_map_path=special_tokens_map_path,
        metadata_path=metadata_path,
        tokenizer_repo_id=tokenizer_repo_id,
        weights_format=weights_format,
    )

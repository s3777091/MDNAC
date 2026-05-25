from __future__ import annotations

import json
from pathlib import Path

import torch

from ..modeling import download_from_huggingface


def assign_parameter(left: torch.Tensor, right: torch.Tensor, tensor_name: str = "unknown") -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
        )

    with torch.no_grad():
        if isinstance(right, torch.Tensor):
            left.copy_(right.to(dtype=left.dtype, device=left.device))
        else:
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

    return left


def load_weights_into_qwen3_5(model, param_config, params) -> None:
    if "model.embed_tokens.weight" in params:
        model_prefix = "model"
    elif "model.language_model.embed_tokens.weight" in params:
        model_prefix = "model.language_model"
    else:
        raise KeyError("Could not find embed token weights in checkpoint.")

    def pkey(suffix: str) -> str:
        return f"{model_prefix}.{suffix}"

    model.tok_emb.weight = assign_parameter(
        model.tok_emb.weight,
        params[pkey("embed_tokens.weight")],
        pkey("embed_tokens.weight"),
    )

    n_layers = param_config["n_layers"]
    layer_types = param_config.get("layer_types", ["full_attention"] * n_layers)

    for layer_idx in range(n_layers):
        block = model.trf_blocks[layer_idx]
        layer_type = layer_types[layer_idx]

        if layer_type == "full_attention":
            att = block.token_mixer
            att.W_query.weight = assign_parameter(
                att.W_query.weight,
                params[pkey(f"layers.{layer_idx}.self_attn.q_proj.weight")],
                pkey(f"layers.{layer_idx}.self_attn.q_proj.weight"),
            )
            att.W_key.weight = assign_parameter(
                att.W_key.weight,
                params[pkey(f"layers.{layer_idx}.self_attn.k_proj.weight")],
                pkey(f"layers.{layer_idx}.self_attn.k_proj.weight"),
            )
            att.W_value.weight = assign_parameter(
                att.W_value.weight,
                params[pkey(f"layers.{layer_idx}.self_attn.v_proj.weight")],
                pkey(f"layers.{layer_idx}.self_attn.v_proj.weight"),
            )
            att.out_proj.weight = assign_parameter(
                att.out_proj.weight,
                params[pkey(f"layers.{layer_idx}.self_attn.o_proj.weight")],
                pkey(f"layers.{layer_idx}.self_attn.o_proj.weight"),
            )
            if hasattr(att, "q_norm") and att.q_norm is not None:
                att.q_norm.weight = assign_parameter(
                    att.q_norm.weight,
                    params[pkey(f"layers.{layer_idx}.self_attn.q_norm.weight")],
                    pkey(f"layers.{layer_idx}.self_attn.q_norm.weight"),
                )
            if hasattr(att, "k_norm") and att.k_norm is not None:
                att.k_norm.weight = assign_parameter(
                    att.k_norm.weight,
                    params[pkey(f"layers.{layer_idx}.self_attn.k_norm.weight")],
                    pkey(f"layers.{layer_idx}.self_attn.k_norm.weight"),
                )

        elif layer_type == "linear_attention":
            lat = block.token_mixer
            lat.dt_bias = assign_parameter(
                lat.dt_bias,
                params[pkey(f"layers.{layer_idx}.linear_attn.dt_bias")],
                pkey(f"layers.{layer_idx}.linear_attn.dt_bias"),
            )
            lat.A_log = assign_parameter(
                lat.A_log,
                params[pkey(f"layers.{layer_idx}.linear_attn.A_log")],
                pkey(f"layers.{layer_idx}.linear_attn.A_log"),
            )
            lat.conv1d.weight = assign_parameter(
                lat.conv1d.weight,
                params[pkey(f"layers.{layer_idx}.linear_attn.conv1d.weight")],
                pkey(f"layers.{layer_idx}.linear_attn.conv1d.weight"),
            )
            lat.norm.weight = assign_parameter(
                lat.norm.weight,
                params[pkey(f"layers.{layer_idx}.linear_attn.norm.weight")],
                pkey(f"layers.{layer_idx}.linear_attn.norm.weight"),
            )
            lat.out_proj.weight = assign_parameter(
                lat.out_proj.weight,
                params[pkey(f"layers.{layer_idx}.linear_attn.out_proj.weight")],
                pkey(f"layers.{layer_idx}.linear_attn.out_proj.weight"),
            )
            lat.in_proj_qkv.weight = assign_parameter(
                lat.in_proj_qkv.weight,
                params[pkey(f"layers.{layer_idx}.linear_attn.in_proj_qkv.weight")],
                pkey(f"layers.{layer_idx}.linear_attn.in_proj_qkv.weight"),
            )
            lat.in_proj_z.weight = assign_parameter(
                lat.in_proj_z.weight,
                params[pkey(f"layers.{layer_idx}.linear_attn.in_proj_z.weight")],
                pkey(f"layers.{layer_idx}.linear_attn.in_proj_z.weight"),
            )
            lat.in_proj_b.weight = assign_parameter(
                lat.in_proj_b.weight,
                params[pkey(f"layers.{layer_idx}.linear_attn.in_proj_b.weight")],
                pkey(f"layers.{layer_idx}.linear_attn.in_proj_b.weight"),
            )
            lat.in_proj_a.weight = assign_parameter(
                lat.in_proj_a.weight,
                params[pkey(f"layers.{layer_idx}.linear_attn.in_proj_a.weight")],
                pkey(f"layers.{layer_idx}.linear_attn.in_proj_a.weight"),
            )
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        block.norm1.weight = assign_parameter(
            block.norm1.weight,
            params[pkey(f"layers.{layer_idx}.input_layernorm.weight")],
            pkey(f"layers.{layer_idx}.input_layernorm.weight"),
        )
        block.ff.fc1.weight = assign_parameter(
            block.ff.fc1.weight,
            params[pkey(f"layers.{layer_idx}.mlp.gate_proj.weight")],
            pkey(f"layers.{layer_idx}.mlp.gate_proj.weight"),
        )
        block.ff.fc2.weight = assign_parameter(
            block.ff.fc2.weight,
            params[pkey(f"layers.{layer_idx}.mlp.up_proj.weight")],
            pkey(f"layers.{layer_idx}.mlp.up_proj.weight"),
        )
        block.ff.fc3.weight = assign_parameter(
            block.ff.fc3.weight,
            params[pkey(f"layers.{layer_idx}.mlp.down_proj.weight")],
            pkey(f"layers.{layer_idx}.mlp.down_proj.weight"),
        )
        block.norm2.weight = assign_parameter(
            block.norm2.weight,
            params[pkey(f"layers.{layer_idx}.post_attention_layernorm.weight")],
            pkey(f"layers.{layer_idx}.post_attention_layernorm.weight"),
        )

    model.final_norm.weight = assign_parameter(
        model.final_norm.weight,
        params[pkey("norm.weight")],
        pkey("norm.weight"),
    )

    if "lm_head.weight" in params:
        model.out_head.weight = assign_parameter(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    elif pkey("lm_head.weight") in params:
        model.out_head.weight = assign_parameter(
            model.out_head.weight,
            params[pkey("lm_head.weight")],
            pkey("lm_head.weight"),
        )
    else:
        model.out_head.weight = model.tok_emb.weight


def ensure_qwen3_5_repo_snapshot(
    repo_id: str,
    local_dir: str | Path,
    *,
    allow_download: bool = True,
) -> Path:
    destination = Path(local_dir).resolve()
    if (destination / "model.safetensors").exists() or (destination / "model.safetensors.index.json").exists():
        return destination

    if not allow_download:
        raise FileNotFoundError(
            f"Qwen3.5 weights not found in {destination}. Enable downloads or place the snapshot there."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the pretrained Qwen3.5 checkpoint."
        ) from exc

    repo_dir = snapshot_download(repo_id=repo_id, local_dir=str(destination))
    return Path(repo_dir).resolve()


def ensure_qwen3_5_tokenizer(
    repo_id: str,
    local_dir: str | Path,
    *,
    allow_download: bool = True,
) -> Path:
    destination_dir = Path(local_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = destination_dir / "tokenizer.json"
    if tokenizer_path.exists():
        return tokenizer_path

    if not allow_download:
        raise FileNotFoundError(
            f"tokenizer.json not found in {destination_dir}. Enable downloads or place the tokenizer there."
        )

    try:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename="tokenizer.json",
                local_dir=str(destination_dir),
            )
        ).resolve()
    except ImportError:
        downloaded = download_from_huggingface(
            repo_id=repo_id,
            filename="tokenizer.json",
            local_dir=str(destination_dir),
        )
        return Path(downloaded).resolve()


def load_qwen3_5_safetensor_weights(repo_dir: str | Path) -> dict[str, torch.Tensor]:
    repo_path = Path(repo_dir).resolve()

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to read the pretrained Qwen3.5 checkpoint.") from exc

    single_file = repo_path / "model.safetensors"
    if single_file.exists():
        return load_file(str(single_file))

    index_path = repo_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Could not find model.safetensors or model.safetensors.index.json in {repo_path}."
        )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weights_dict: dict[str, torch.Tensor] = {}
    for filename in sorted(set(index["weight_map"].values())):
        shard_path = repo_path / filename
        shard = load_file(str(shard_path))
        weights_dict.update(shard)
    return weights_dict

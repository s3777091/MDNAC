from __future__ import annotations

import json
from pathlib import Path

import torch

from .ch05 import DEFAULT_GEMMA4_E2B_REPO_ID, download_from_huggingface

DEFAULT_GEMMA4_E2B_LOCAL_DIR = DEFAULT_GEMMA4_E2B_REPO_ID.split("/")[-1]


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


def _resolve_text_prefix(params: dict[str, torch.Tensor]) -> str:
    if "model.language_model.embed_tokens.weight" in params:
        return "model.language_model"
    if "model.embed_tokens.weight" in params:
        return "model"
    raise KeyError("Could not find Gemma 4 text embedding weights in checkpoint.")


def load_weights_into_gemma4(model, param_config: dict, params: dict[str, torch.Tensor]) -> None:
    model_prefix = _resolve_text_prefix(params)

    def pkey(suffix: str) -> str:
        return f"{model_prefix}.{suffix}"

    model.tok_emb.weight = assign_parameter(
        model.tok_emb.weight,
        params[pkey("embed_tokens.weight")],
        pkey("embed_tokens.weight"),
    )

    if getattr(model, "hidden_size_per_layer_input", 0):
        model.embed_tokens_per_layer.weight = assign_parameter(
            model.embed_tokens_per_layer.weight,
            params[pkey("embed_tokens_per_layer.weight")],
            pkey("embed_tokens_per_layer.weight"),
        )
        model.per_layer_model_projection.weight = assign_parameter(
            model.per_layer_model_projection.weight,
            params[pkey("per_layer_model_projection.weight")],
            pkey("per_layer_model_projection.weight"),
        )
        model.per_layer_projection_norm.weight = assign_parameter(
            model.per_layer_projection_norm.weight,
            params[pkey("per_layer_projection_norm.weight")],
            pkey("per_layer_projection_norm.weight"),
        )

    for layer_idx in range(param_config["n_layers"]):
        block = model.blocks[layer_idx]
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
        if att.W_value is not None:
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
        att.q_norm.weight = assign_parameter(
            att.q_norm.weight,
            params[pkey(f"layers.{layer_idx}.self_attn.q_norm.weight")],
            pkey(f"layers.{layer_idx}.self_attn.q_norm.weight"),
        )
        att.k_norm.weight = assign_parameter(
            att.k_norm.weight,
            params[pkey(f"layers.{layer_idx}.self_attn.k_norm.weight")],
            pkey(f"layers.{layer_idx}.self_attn.k_norm.weight"),
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

        block.input_layernorm.weight = assign_parameter(
            block.input_layernorm.weight,
            params[pkey(f"layers.{layer_idx}.input_layernorm.weight")],
            pkey(f"layers.{layer_idx}.input_layernorm.weight"),
        )
        block.post_attention_layernorm.weight = assign_parameter(
            block.post_attention_layernorm.weight,
            params[pkey(f"layers.{layer_idx}.post_attention_layernorm.weight")],
            pkey(f"layers.{layer_idx}.post_attention_layernorm.weight"),
        )
        block.pre_feedforward_layernorm.weight = assign_parameter(
            block.pre_feedforward_layernorm.weight,
            params[pkey(f"layers.{layer_idx}.pre_feedforward_layernorm.weight")],
            pkey(f"layers.{layer_idx}.pre_feedforward_layernorm.weight"),
        )
        block.post_feedforward_layernorm.weight = assign_parameter(
            block.post_feedforward_layernorm.weight,
            params[pkey(f"layers.{layer_idx}.post_feedforward_layernorm.weight")],
            pkey(f"layers.{layer_idx}.post_feedforward_layernorm.weight"),
        )

        if getattr(block, "hidden_size_per_layer_input", 0):
            block.per_layer_input_gate.weight = assign_parameter(
                block.per_layer_input_gate.weight,
                params[pkey(f"layers.{layer_idx}.per_layer_input_gate.weight")],
                pkey(f"layers.{layer_idx}.per_layer_input_gate.weight"),
            )
            block.per_layer_projection.weight = assign_parameter(
                block.per_layer_projection.weight,
                params[pkey(f"layers.{layer_idx}.per_layer_projection.weight")],
                pkey(f"layers.{layer_idx}.per_layer_projection.weight"),
            )
            block.post_per_layer_input_norm.weight = assign_parameter(
                block.post_per_layer_input_norm.weight,
                params[pkey(f"layers.{layer_idx}.post_per_layer_input_norm.weight")],
                pkey(f"layers.{layer_idx}.post_per_layer_input_norm.weight"),
            )

    model.final_norm.weight = assign_parameter(
        model.final_norm.weight,
        params[pkey("norm.weight")],
        pkey("norm.weight"),
    )

    if "lm_head.weight" in params:
        model.out_head.weight = assign_parameter(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    elif model.cfg.get("tie_word_embeddings", True):
        model.out_head.weight = model.tok_emb.weight


def ensure_gemma4_repo_snapshot(
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
            f"Gemma 4 weights not found in {destination}. Enable downloads or place the snapshot there."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download Gemma 4 checkpoints.") from exc

    repo_dir = snapshot_download(repo_id=repo_id, local_dir=str(destination))
    return Path(repo_dir).resolve()


def ensure_gemma4_tokenizer(
    repo_id: str = DEFAULT_GEMMA4_E2B_REPO_ID,
    local_dir: str | Path | None = None,
    *,
    allow_download: bool = True,
) -> Path:
    if local_dir is None:
        local_dir = DEFAULT_GEMMA4_E2B_LOCAL_DIR
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


def load_gemma4_safetensor_weights(repo_dir: str | Path) -> dict[str, torch.Tensor]:
    repo_path = Path(repo_dir).resolve()

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to read Gemma 4 checkpoints.") from exc

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

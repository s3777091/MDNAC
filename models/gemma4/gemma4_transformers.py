from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


ACT2FN = {
    "gelu_pytorch_tanh": _gelu_pytorch_tanh,
    "silu": F.silu,
}


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_f = x.float()
        x_norm = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.with_scale:
            x_norm = x_norm * self.weight.float()
        return x_norm.to(dtype=input_dtype)


class ScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None,
        embed_scale: float,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, dtype=dtype)
        self.register_buffer("embed_scale", torch.tensor(float(embed_scale)), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


def _validate_rope_dim(head_dim: int) -> None:
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer.")


def compute_rope_params(
    *,
    head_dim: int,
    theta_base: float,
    context_length: int,
    rope_type: str = "default",
    partial_rotary_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for Gemma 4 text RoPE."""

    _validate_rope_dim(head_dim)
    if rope_type == "default":
        inv_freq = 1.0 / (
            theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype).float() / head_dim)
        )
    elif rope_type == "proportional":
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        if rope_angles <= 0:
            raise ValueError("partial_rotary_factor produces no rotary dimensions.")
        inv_freq_rotated = 1.0 / (
            theta_base
            ** (torch.arange(0, 2 * rope_angles, 2, dtype=dtype).float() / head_dim)
        )
        nope_angles = head_dim // 2 - rope_angles
        inv_freq = (
            torch.cat([inv_freq_rotated, torch.zeros(nope_angles, dtype=dtype)], dim=0)
            if nope_angles > 0
            else inv_freq_rotated
        )
    else:
        raise ValueError(f"Unsupported rope_type: {rope_type!r}")

    positions = torch.arange(context_length, dtype=dtype)
    freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=1)
    return torch.cos(emb), torch.sin(emb)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    offset: int = 0,
) -> torch.Tensor:
    seq_len = x.shape[-2]
    if offset < 0 or offset + seq_len > cos.shape[0]:
        raise ValueError(
            f"RoPE slice [{offset}, {offset + seq_len}) exceeds table length {cos.shape[0]}."
        )

    cos = cos[offset : offset + seq_len].to(device=x.device, dtype=x.dtype)
    sin = sin[offset : offset + seq_len].to(device=x.device, dtype=x.dtype)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_kv_heads,
        repeats,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch, num_kv_heads * repeats, seq_len, head_dim)


class FeedForward(nn.Module):
    def __init__(self, cfg: dict, layer_idx: int):
        super().__init__()
        first_shared_idx = cfg["n_layers"] - int(cfg.get("num_kv_shared_layers", 0))
        is_shared_layer = layer_idx >= first_shared_idx > 0
        width_multiplier = 2 if cfg.get("use_double_wide_mlp", False) and is_shared_layer else 1

        self.hidden_dim = int(cfg["hidden_dim"]) * width_multiplier
        self.gate_proj = nn.Linear(cfg["emb_dim"], self.hidden_dim, dtype=cfg["dtype"], bias=False)
        self.up_proj = nn.Linear(cfg["emb_dim"], self.hidden_dim, dtype=cfg["dtype"], bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
        self.fc1 = self.gate_proj
        self.fc2 = self.up_proj
        self.fc3 = self.down_proj
        self.act_fn = ACT2FN[cfg.get("hidden_activation", "gelu_pytorch_tanh")]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma4Attention(nn.Module):
    def __init__(self, cfg: dict, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.layer_type = cfg["layer_types"][layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = cfg.get("sliding_window") if self.is_sliding else None

        self.num_heads = int(cfg["n_heads"])
        self.head_dim = (
            int(cfg.get("global_head_dim") or cfg["head_dim"])
            if not self.is_sliding
            else int(cfg["head_dim"])
        )
        self.use_alternative_attention = bool(cfg.get("attention_k_eq_v", False)) and not self.is_sliding
        num_kv_heads = (
            cfg.get("num_global_kv_groups")
            if self.use_alternative_attention
            else cfg["n_kv_groups"]
        )
        if num_kv_heads is None:
            num_kv_heads = cfg["n_kv_groups"]
        self.num_kv_heads = int(num_kv_heads)
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("n_heads must be divisible by the selected KV head count.")
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        first_shared_idx = cfg["n_layers"] - int(cfg.get("num_kv_shared_layers", 0))
        self.is_kv_shared_layer = layer_idx >= first_shared_idx > 0
        previous_layer_types = cfg["layer_types"][:first_shared_idx]
        if self.is_kv_shared_layer:
            reversed_previous = previous_layer_types[::-1]
            if self.layer_type not in reversed_previous:
                raise ValueError(f"No non-shared source layer found for {self.layer_type!r}.")
            self.kv_shared_layer_index = (
                len(previous_layer_types) - 1 - reversed_previous.index(self.layer_type)
            )
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            self.store_full_length_kv = (
                first_shared_idx > 0
                and layer_idx
                == len(previous_layer_types) - 1 - previous_layer_types[::-1].index(self.layer_type)
            )

        d_in = int(cfg["emb_dim"])
        dtype = cfg["dtype"]
        self.q_proj = nn.Linear(d_in, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_in, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = (
            None
            if self.use_alternative_attention
            else nn.Linear(d_in, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_in, bias=False, dtype=dtype)
        self.W_query = self.q_proj
        self.W_key = self.k_proj
        self.W_value = self.v_proj
        self.out_proj = self.o_proj

        self.q_norm = RMSNorm(self.head_dim, eps=cfg.get("rms_norm_eps", 1e-6))
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.get("rms_norm_eps", 1e-6))
        self.v_norm = RMSNorm(self.head_dim, eps=cfg.get("rms_norm_eps", 1e-6), with_scale=False)

    def _compute_kv(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        kv_shape = (batch_size, seq_len, self.num_kv_heads, self.head_dim)

        key_states = self.k_proj(hidden_states).view(kv_shape)
        value_states = (
            key_states
            if self.v_proj is None
            else self.v_proj(hidden_states).view(kv_shape)
        )

        key_states = self.k_norm(key_states).transpose(1, 2)
        key_states = apply_rope(key_states, cos, sin, offset=start_pos)
        value_states = self.v_norm(value_states).transpose(1, 2)
        return key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: KVCache | None = None,
        shared_layers: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states).transpose(1, 2)
        query_states = apply_rope(query_states, cos, sin, offset=start_pos)

        if shared_layers is None:
            shared_layers = {}

        if self.is_kv_shared_layer:
            if self.kv_shared_layer_index not in shared_layers:
                raise RuntimeError(
                    "Shared KV source was not populated before a shared Gemma 4 layer. "
                    f"Missing layer index: {self.kv_shared_layer_index}"
                )
            key_states, value_states = shared_layers[self.kv_shared_layer_index]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_new, value_new = self._compute_kv(hidden_states, cos, sin, start_pos=start_pos)
            if cache is not None:
                cached = cache.get(self.layer_idx)
                if cached is None:
                    key_states, value_states = key_new, value_new
                else:
                    prev_key, prev_value = cached
                    key_states = torch.cat([prev_key, key_new], dim=2)
                    value_states = torch.cat([prev_value, value_new], dim=2)
                cache.update(self.layer_idx, (key_states, value_states))
            else:
                key_states, value_states = key_new, value_new

            if self.store_full_length_kv:
                shared_layers[self.layer_idx] = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_scores = query_states @ key_states.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(attention_mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        context = attn_weights @ value_states
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict, layer_idx: int):
        super().__init__()
        if cfg.get("enable_moe_block", False):
            raise NotImplementedError("Gemma 4 MoE blocks are not implemented in this E2B text module.")

        self.layer_idx = layer_idx
        self.layer_type = cfg["layer_types"][layer_idx]
        self.token_mixer = Gemma4Attention(cfg, layer_idx)
        self.att = self.token_mixer
        self.ff = FeedForward(cfg, layer_idx)
        self.input_layernorm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.pre_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.post_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))

        self.hidden_size_per_layer_input = int(cfg.get("hidden_size_per_layer_input", 0) or 0)
        if self.hidden_size_per_layer_input:
            self.act_fn = ACT2FN[cfg.get("hidden_activation", "gelu_pytorch_tanh")]
            self.per_layer_input_gate = nn.Linear(
                cfg["emb_dim"],
                self.hidden_size_per_layer_input,
                dtype=cfg["dtype"],
                bias=False,
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input,
                cfg["emb_dim"],
                dtype=cfg["dtype"],
                bias=False,
            )
            self.post_per_layer_input_norm = RMSNorm(
                cfg["emb_dim"],
                eps=cfg.get("rms_norm_eps", 1e-6),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        per_layer_input: torch.Tensor | None,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: KVCache | None = None,
        shared_layers: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.att(
            hidden_states,
            attention_mask=attention_mask,
            cos=cos,
            sin=sin,
            start_pos=start_pos,
            cache=cache,
            shared_layers=shared_layers,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input:
            if per_layer_input is None:
                raise ValueError("per_layer_input is required for Gemma 4 PLE layers.")
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states


class KVCache:
    def __init__(self, n_layers: int):
        self.cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n_layers
        self.shared_layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, layer_idx: int):
        return self.cache[layer_idx]

    def update(self, layer_idx: int, value: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.cache[layer_idx] = value

    def reset(self) -> None:
        for idx in range(len(self.cache)):
            self.cache[idx] = None
        self.shared_layers.clear()


Gemma4KVCache = KVCache
GroupedQueryAttention = Gemma4Attention



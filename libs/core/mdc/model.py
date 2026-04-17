from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from .config import MDCModelConfig
from .linear_attention import MDCGatedDeltaNet


def _normalize_cfg(cfg: MDCModelConfig | Mapping[str, object]) -> dict[str, object]:
    if isinstance(cfg, MDCModelConfig):
        return cfg.to_dict()
    return dict(cfg)


class FeedForward(nn.Module):
    def __init__(self, cfg: Mapping[str, object]) -> None:
        super().__init__()
        dtype = cfg["dtype"]
        emb_dim = int(cfg["emb_dim"])
        hidden_dim = int(cfg["hidden_dim"])
        self.fc1 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(emb_dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self._norm(x.float())
        x_norm = x_norm * (1.0 + self.weight.float())
        return x_norm.to(dtype=x.dtype)


def compute_rope_params(
    head_dim: int,
    theta_base: float = 10_000,
    context_length: int = 4096,
    partial_rotary_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even.")

    rotary_dim = int(head_dim * partial_rotary_factor)
    rotary_dim = max(2, rotary_dim - (rotary_dim % 2))

    inv_freq = 1.0 / (
        theta_base ** (
            torch.arange(0, rotary_dim, 2, dtype=dtype)[: (rotary_dim // 2)].float() / rotary_dim
        )
    )
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    _, _, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even.")

    rot_dim = cos.shape[-1]
    if rot_dim > head_dim:
        raise ValueError(f"RoPE dim {rot_dim} cannot exceed head_dim {head_dim}.")

    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    x1 = x_rot[..., : rot_dim // 2]
    x2 = x_rot[..., rot_dim // 2 :]

    cos = cos[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x_rot * cos) + (rotated * sin)
    return torch.cat([x_rotated, x_pass], dim=-1).to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int | None = None,
        qk_norm: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups.")

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            if d_in % num_heads != 0:
                raise ValueError("d_in must be divisible by num_heads if head_dim is not set.")
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out * 2, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache=None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, num_tokens, _ = x.shape

        q_and_gate = self.W_query(x)
        q_and_gate = q_and_gate.view(batch_size, num_tokens, self.num_heads, self.head_dim * 2)
        queries, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(batch_size, num_tokens, self.d_out)

        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.transpose(1, 2)
        keys_new = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys_new = self.k_norm(keys_new)

        prev_len = 0
        if cache is not None:
            prev_k, prev_v = cache
            if prev_k is not None:
                prev_len = prev_k.size(2)
                keys_cat_raw = torch.cat([prev_k, keys_new], dim=2)
                values_cat_raw = torch.cat([prev_v, values_new], dim=2)
            else:
                keys_cat_raw = keys_new
                values_cat_raw = values_new
        else:
            keys_cat_raw = keys_new
            values_cat_raw = values_new

        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys = apply_rope(keys_cat_raw, cos, sin, offset=start_pos - prev_len)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values_cat_raw.repeat_interleave(self.group_size, dim=1)

        if cache is not None and cache[0] is not None:
            next_cache = (
                torch.cat([cache[0], keys_new], dim=2),
                torch.cat([cache[1], values_new], dim=2),
            )
        else:
            next_cache = (keys_new, values_new)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores * (self.head_dim ** -0.5),
            dim=-1,
            dtype=torch.float32,
        ).to(queries.dtype)

        context = (attn_weights @ values).transpose(1, 2).reshape(batch_size, num_tokens, self.d_out)
        context = context * torch.sigmoid(gate)
        out = self.out_proj(context)
        return out, next_cache


class _MDCConfigAdapter:
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


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Mapping[str, object], layer_type: str, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = layer_type

        if layer_type == "full_attention":
            self.token_mixer = GroupedQueryAttention(
                d_in=int(cfg["emb_dim"]),
                num_heads=int(cfg["n_heads"]),
                head_dim=int(cfg["head_dim"]),
                num_kv_groups=int(cfg["n_kv_groups"]),
                qk_norm=bool(cfg["qk_norm"]),
                dtype=cfg["dtype"],
            )
        elif layer_type == "linear_attention":
            self.token_mixer = MDCGatedDeltaNet(_MDCConfigAdapter(cfg), layer_idx)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

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


class MDCLinearAttentionCache:
    def __init__(self, n_layers: int) -> None:
        self.conv_states = [None] * n_layers
        self.recurrent_states = [None] * n_layers
        self.has_previous_state = False

    def reset(self) -> None:
        for index in range(len(self.conv_states)):
            self.conv_states[index] = None
            self.recurrent_states[index] = None
        self.has_previous_state = False


class MDCDecoderCache:
    def __init__(self, n_layers: int) -> None:
        self.cache = [None] * n_layers
        self.linear_cache = MDCLinearAttentionCache(n_layers)

    def get(self, layer_idx: int):
        return self.cache[layer_idx]

    def update(self, layer_idx: int, value) -> None:
        self.cache[layer_idx] = value

    def reset(self) -> None:
        for index in range(len(self.cache)):
            self.cache[index] = None
        self.linear_cache.reset()


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

    def reset_kv_cache(self) -> None:
        self.current_pos = 0

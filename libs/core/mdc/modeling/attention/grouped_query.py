from __future__ import annotations

import torch
import torch.nn as nn

from ..components.normalization import RMSNorm
from ..components.rope import apply_rope


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

        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            if prev_k is not None:
                keys = torch.cat([prev_k, keys_new], dim=2)
                values = torch.cat([prev_v, values_new], dim=2)
            else:
                keys = keys_new
                values = values_new
        else:
            keys = keys_new
            values = values_new

        next_cache = (keys, values)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

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

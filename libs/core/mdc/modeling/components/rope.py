from __future__ import annotations

import torch


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

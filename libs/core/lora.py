"""Minimal, dependency-free LoRA (Low-Rank Adaptation) for the MDC backbone.

Used by the instruction/completion fine-tune stage: freeze the pretrained backbone
and train only small rank-``r`` adapters on the projection ``nn.Linear`` layers. This
shrinks the AdamW optimizer state from ~2 states per backbone weight down to a few
percent of the parameters, freeing VRAM to raise batch size / context length.

The token embedding and output head are intentionally NOT adapted here — when the
backbone is re-vocabularised for the fused profile+sequence vocabulary those rows are
freshly initialised and must stay fully trainable, otherwise the new tokens never learn.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable low-rank update ``B @ A``.

    ``y = base(x) + dropout(x) @ A^T @ B^T * (alpha / r)``. The base weight/bias are
    frozen; only ``lora_A`` and ``lora_B`` receive gradients.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be greater than 0.")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_features = base.in_features
        out_features = base.out_features
        param_dtype = base.weight.dtype
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(p=float(dropout)) if dropout > 0.0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=param_dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=param_dtype))
        # A ~ Kaiming, B = 0 -> the adapter starts as a no-op (output == frozen base).
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        update = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base_out + update * self.scaling


def _default_target_predicate(name: str) -> bool:
    # Attention (q/k/v/o, in/out proj) and FFN (fc1/fc2/fc3) projections.
    keywords = ("proj", "fc1", "fc2", "fc3", "qkv", "out")
    lowered = name.lower()
    return any(keyword in lowered for keyword in keywords)


def apply_lora_to_linears(
    root: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_predicate: Callable[[str], bool] = _default_target_predicate,
) -> int:
    """Replace matching ``nn.Linear`` children under ``root`` with ``LoRALinear``.

    Returns the number of layers adapted. Matching is by the child's qualified name.
    """
    adapted = 0
    for module_name, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            qualified = f"{module_name}.{child_name}" if module_name else child_name
            if not target_predicate(qualified):
                continue
            setattr(
                module,
                child_name,
                LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
            )
            adapted += 1
    if adapted == 0:
        raise ValueError(
            "apply_lora_to_linears matched no nn.Linear layers; check target_predicate."
        )
    return adapted


def mark_only_lora_trainable(
    model: nn.Module,
    *,
    also_trainable: Sequence[nn.Module] = (),
) -> None:
    """Freeze every parameter except LoRA adapters and the ``also_trainable`` modules.

    ``also_trainable`` should typically be the re-vocabularised token embedding and
    output head, whose new rows must keep learning.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
    for module in also_trainable:
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

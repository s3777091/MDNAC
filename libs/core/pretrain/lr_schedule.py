"""Learning-rate scheduling for protein pretraining and instruction tuning.

Both training loops historically ran AdamW at a constant learning rate. For
from-scratch language-model training that hurts convergence: a high flat LR is
unstable while weights are still random, and never decaying leaves the model
oscillating around the optimum so next-token accuracy and token coverage stall.

This module adds the standard warmup + cosine-decay schedule used by
``rasbt/LLMs-from-scratch`` and Qwen-style training. It is optimizer-agnostic
(works for a single AdamW or the muon+adamw optimizer list), resumable via an
absolute global-step index, and degrades gracefully to warmup-then-hold when the
total-step horizon is unknown (streaming datasets without ``max_steps``).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class LRScheduleConfig:
    enabled: bool = False
    warmup_steps: int = 0
    warmup_ratio: float | None = None
    min_lr_ratio: float = 0.1
    decay_steps: int | None = None

    @classmethod
    def from_optimizer_config(cls, optimizer_cfg: Mapping[str, Any]) -> "LRScheduleConfig":
        scheduler_type = str(optimizer_cfg.get("lr_scheduler") or "none").strip().lower()
        if scheduler_type not in {"none", "cosine"}:
            raise ValueError("optimizer.lr_scheduler must be one of: none, cosine")
        warmup_ratio = optimizer_cfg.get("warmup_ratio")
        return cls(
            enabled=scheduler_type == "cosine",
            warmup_steps=int(optimizer_cfg.get("warmup_steps") or 0),
            warmup_ratio=None if warmup_ratio in {None, ""} else float(warmup_ratio),
            min_lr_ratio=float(optimizer_cfg.get("min_lr_ratio", 0.1)),
            decay_steps=(
                int(optimizer_cfg["lr_decay_steps"])
                if optimizer_cfg.get("lr_decay_steps") not in {None, ""}
                else None
            ),
        )

    def resolve_warmup_steps(self, total_steps: int | None) -> int:
        if self.warmup_ratio is not None and total_steps is not None:
            return max(0, int(round(self.warmup_ratio * total_steps)))
        return max(0, int(self.warmup_steps))

    def resolve_total_steps(self, max_steps: int | None) -> int | None:
        if self.decay_steps is not None:
            return int(self.decay_steps)
        if max_steps is not None:
            return int(max_steps)
        return None


def cosine_warmup_lr_scale(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int | None,
    min_lr_ratio: float,
) -> float:
    """Return the LR multiplier in ``[min_lr_ratio, 1.0]`` for a 0-indexed step.

    Linear warmup from ~0 to 1.0 over ``warmup_steps``, then cosine decay from
    1.0 down to ``min_lr_ratio`` between ``warmup_steps`` and ``total_steps``.
    When ``total_steps`` is unknown (``None``) or not larger than the warmup,
    the multiplier holds at 1.0 after warmup (no decay).
    """
    step = max(0, int(step))
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    if total_steps is None or total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


class WarmupCosineLRScheduler:
    """Applies a warmup + cosine schedule to one or more optimizers.

    The schedule is driven by an absolute optimizer-step index so it resumes
    correctly: build with ``last_step`` set to the restored ``global_step`` and
    the upcoming step receives the right learning rate.
    """

    def __init__(
        self,
        optimizers: Sequence[torch.optim.Optimizer],
        *,
        warmup_steps: int,
        total_steps: int | None,
        min_lr_ratio: float,
        last_step: int = 0,
    ) -> None:
        self.optimizers = list(optimizers)
        if not self.optimizers:
            raise ValueError("WarmupCosineLRScheduler requires at least one optimizer.")
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = None if total_steps is None else int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        # Capture the peak LR per param group at build time; every update scales
        # these base values rather than reading back already-scaled LRs.
        self.base_lrs = [[float(group["lr"]) for group in opt.param_groups] for opt in self.optimizers]
        self._step = int(last_step)
        self._apply(self._step)

    @property
    def decays(self) -> bool:
        return self.total_steps is not None and self.total_steps > self.warmup_steps

    def scale_at(self, step: int) -> float:
        return cosine_warmup_lr_scale(
            step,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=self.min_lr_ratio,
        )

    def _apply(self, step: int) -> None:
        scale = self.scale_at(step)
        for optimizer, base_group_lrs in zip(self.optimizers, self.base_lrs):
            for group, base_lr in zip(optimizer.param_groups, base_group_lrs):
                group["lr"] = base_lr * scale

    def set_step(self, step: int) -> None:
        self._step = int(step)
        self._apply(self._step)

    def step(self) -> None:
        self.set_step(self._step + 1)

    def current_lr(self) -> float:
        return float(self.optimizers[0].param_groups[0]["lr"])


def build_warmup_cosine_scheduler(
    optimizers: Sequence[torch.optim.Optimizer],
    schedule: LRScheduleConfig,
    *,
    max_steps: int | None,
    last_step: int,
) -> WarmupCosineLRScheduler | None:
    """Construct a scheduler from config, or ``None`` when scheduling is disabled."""
    if not schedule.enabled:
        return None
    total_steps = schedule.resolve_total_steps(max_steps)
    warmup_steps = schedule.resolve_warmup_steps(total_steps)
    if warmup_steps <= 0 and total_steps is None:
        return None
    return WarmupCosineLRScheduler(
        optimizers,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=schedule.min_lr_ratio,
        last_step=last_step,
    )

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from libs.core.interfaces import CausalLMBatch


def compute_mdc_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten())


def evaluate_mdc_causal_lm_batch_loss(
    model_or_app,
    batches: Iterable[CausalLMBatch],
    *,
    device: torch.device | str,
    max_batches: int | None = None,
) -> float:
    model_or_app.eval()
    losses: list[float] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break

            resolved_batch = _move_causal_lm_batch_to_device(batch, device=device)
            logits = _forward_causal_lm_batch(model_or_app, resolved_batch)
            loss = compute_mdc_causal_lm_loss(logits, resolved_batch.labels)
            losses.append(float(loss.item()))

    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def run_mdc_causal_lm_batch_epoch(
    model_or_app,
    data_loader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    grad_clip_norm: float | None = None,
) -> float:
    model_or_app.train()
    losses: list[float] = []

    for batch in data_loader:
        resolved_batch = _move_causal_lm_batch_to_device(batch, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = _forward_causal_lm_batch(model_or_app, resolved_batch)
        loss = compute_mdc_causal_lm_loss(logits, resolved_batch.labels)
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model_or_app.parameters(), grad_clip_norm)

        optimizer.step()
        losses.append(float(loss.item()))

    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def _move_causal_lm_batch_to_device(
    batch: CausalLMBatch,
    *,
    device: torch.device | str,
) -> CausalLMBatch:
    return CausalLMBatch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        labels=batch.labels.to(device),
    )


def _forward_causal_lm_batch(
    model_or_app,
    batch: CausalLMBatch,
) -> torch.Tensor:
    if hasattr(model_or_app, "forward_causal_lm_batch"):
        return model_or_app.forward_causal_lm_batch(batch)
    return model_or_app(batch.input_ids, attn_mask=batch.attention_mask)

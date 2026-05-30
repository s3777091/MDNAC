from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F

from libs.core.interfaces import CausalLMBatch
from .distributed import set_mdc_data_loader_epoch, unwrap_mdc_training_model


def compute_mdc_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten())


def create_muon_optimizers(
    model: torch.nn.Module,
    *,
    adamw_learning_rate: float,
    muon_learning_rate: float | None = None,
    weight_decay: float = 0.1,
) -> list[torch.optim.Optimizer]:
    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_cls is None:
        raise RuntimeError(
            "torch.optim.Muon is required for optimizer.type=muon. "
            "Install a PyTorch build that includes native Muon support."
        )

    embedding_param_names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                embedding_param_names.add(full_name)

    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 2 and name not in embedding_param_names:
            muon_params.append(parameter)
        else:
            adamw_params.append(parameter)

    optimizers: list[torch.optim.Optimizer] = []
    if muon_params:
        optimizers.append(
            muon_cls(
                muon_params,
                lr=float(muon_learning_rate if muon_learning_rate is not None else adamw_learning_rate),
                weight_decay=weight_decay,
                adjust_lr_fn="match_rms_adamw",
            )
        )
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=adamw_learning_rate, weight_decay=weight_decay))
    if not optimizers:
        raise ValueError("No trainable parameters found.")
    return optimizers


def create_moon_optimizers(
    model: torch.nn.Module,
    *,
    adamw_learning_rate: float,
    muon_learning_rate: float | None = None,
    weight_decay: float = 0.1,
) -> list[torch.optim.Optimizer]:
    return create_muon_optimizers(
        model,
        adamw_learning_rate=adamw_learning_rate,
        muon_learning_rate=muon_learning_rate,
        weight_decay=weight_decay,
    )


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
    optimizer: torch.optim.Optimizer | Sequence[torch.optim.Optimizer],
    *,
    device: torch.device | str,
    grad_clip_norm: float | None = None,
    epoch: int | None = None,
) -> float:
    model_or_app.train()
    optimizers = _as_optimizer_list(optimizer)
    losses: list[float] = []

    if epoch is not None:
        set_mdc_data_loader_epoch(data_loader, epoch)

    for batch in data_loader:
        resolved_batch = _move_causal_lm_batch_to_device(batch, device=device)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        logits = _forward_causal_lm_batch(model_or_app, resolved_batch)
        loss = compute_mdc_causal_lm_loss(logits, resolved_batch.labels)
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model_or_app.parameters(), grad_clip_norm)

        for opt in optimizers:
            opt.step()
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
    if unwrap_mdc_training_model(model_or_app) is not model_or_app:
        return model_or_app(batch.input_ids, attn_mask=batch.attention_mask)
    if hasattr(model_or_app, "forward_causal_lm_batch"):
        return model_or_app.forward_causal_lm_batch(batch)
    return model_or_app(batch.input_ids, attn_mask=batch.attention_mask)


def _as_optimizer_list(
    optimizer: torch.optim.Optimizer | Sequence[torch.optim.Optimizer],
) -> list[torch.optim.Optimizer]:
    if isinstance(optimizer, torch.optim.Optimizer):
        return [optimizer]

    optimizers = list(optimizer)
    if not optimizers:
        raise ValueError("optimizer sequence must not be empty.")
    return optimizers

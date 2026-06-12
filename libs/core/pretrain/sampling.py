"""Logit-processing and next-token sampling for autoregressive generation.

Pure-tensor helpers (no model dependency) so they are cheap to unit-test on CPU and
reusable by both ``profile_generation`` and notebook inference. Supports temperature,
top-k, top-p (nucleus), repetition penalty, and an allowed-token mask that keeps the
decoder from emitting invalid tokens (e.g. profile/separator/pad ids during protein
sequence generation) — directly improving biological validity and reducing hallucination.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: Sequence[int] | torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Down-weight logits of already-generated tokens (CTRL-style penalty > 1.0)."""
    if penalty == 1.0:
        return logits
    if penalty <= 0.0:
        raise ValueError("repetition_penalty must be greater than 0.")
    ids = generated_ids.tolist() if isinstance(generated_ids, torch.Tensor) else list(generated_ids)
    if not ids:
        return logits
    unique_ids = torch.tensor(sorted(set(int(i) for i in ids)), device=logits.device, dtype=torch.long)
    selected = logits.index_select(-1, unique_ids)
    # Positive logits are divided, negative logits are multiplied (standard formulation).
    adjusted = torch.where(selected > 0, selected / penalty, selected * penalty)
    logits = logits.clone()
    logits.index_copy_(-1, unique_ids, adjusted)
    return logits


def restrict_to_allowed_tokens(
    logits: torch.Tensor,
    allowed_token_ids: Iterable[int],
) -> torch.Tensor:
    """Set every logit outside ``allowed_token_ids`` to -inf."""
    allowed = torch.tensor(sorted(set(int(i) for i in allowed_token_ids)), device=logits.device, dtype=torch.long)
    if allowed.numel() == 0:
        raise ValueError("allowed_token_ids must not be empty.")
    mask = torch.full_like(logits, float("-inf"))
    mask.index_copy_(-1, allowed, logits.index_select(-1, allowed))
    return mask


def filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only the ``top_k`` highest logits; the rest become -inf."""
    if top_k <= 0:
        return logits
    k = min(int(top_k), logits.size(-1))
    top_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_values[..., -1].unsqueeze(-1)
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose cumulative prob >= top_p."""
    if top_p >= 1.0:
        return logits
    if not 0.0 < top_p < 1.0:
        raise ValueError("top_p must be in (0, 1].")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    # Remove tokens once cumulative prob exceeds top_p, but always keep the top-1 token.
    sorted_remove = cumulative - torch.softmax(sorted_logits, dim=-1) >= top_p
    sorted_remove[..., 0] = False
    remove = sorted_remove.scatter(-1, sorted_indices, sorted_remove)
    return logits.masked_fill(remove, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    generated_ids: Sequence[int] | torch.Tensor | None = None,
    allowed_token_ids: Iterable[int] | None = None,
) -> torch.Tensor:
    """Return the next token id ``(..., 1)`` from final-position ``logits`` ``(..., vocab)``.

    Order: repetition penalty -> allowed-token mask -> temperature -> top-k -> top-p ->
    (greedy if temperature == 0 else multinomial sample).
    """
    processed = logits
    if repetition_penalty != 1.0 and generated_ids is not None:
        processed = apply_repetition_penalty(processed, generated_ids, repetition_penalty)
    if allowed_token_ids is not None:
        processed = restrict_to_allowed_tokens(processed, allowed_token_ids)

    if temperature <= 0.0:
        return torch.argmax(processed, dim=-1, keepdim=True)

    processed = processed / temperature
    if top_k is not None:
        processed = filter_top_k(processed, top_k)
    if top_p is not None:
        processed = filter_top_p(processed, top_p)
    probs = torch.softmax(processed, dim=-1)
    return torch.multinomial(probs, num_samples=1)

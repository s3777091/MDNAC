from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    full_text: str
    answer_text: str
    model_family: str


def generate_tokens_with_logits(
    next_logits: Callable[[torch.Tensor], torch.Tensor],
    idx: torch.Tensor,
    *,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    token_ids = idx.detach().cpu()
    normalized_top_k = None if top_k is None or top_k <= 0 else int(top_k)

    for _ in range(max_new_tokens):
        idx_cond = token_ids[:, -context_size:]
        logits = next_logits(idx_cond)
        logits = logits[:, -1, :]

        if normalized_top_k is not None:
            top_logits, _ = torch.topk(logits, normalized_top_k)
            min_values = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_values,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_token_id is not None and torch.all(idx_next == eos_token_id):
            break

        token_ids = torch.cat((token_ids, idx_next.to(dtype=token_ids.dtype)), dim=1)

    return token_ids

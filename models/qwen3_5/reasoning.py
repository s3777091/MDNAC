from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import torch


IGNORE_INDEX = -100
DEFAULT_PROMPT_FIELDS = ("prompt", "problem", "instruction", "user_input", "user")
DEFAULT_RESPONSE_FIELDS = ("message_content", "response", "answer", "output", "assistant")
DEFAULT_THINKING_FIELDS = ("message_thinking", "thinking", "reasoning", "cot")


def strip_think_tags(text: Any) -> str:
    return str(text or "").replace("<think>", "").replace("</think>", "").strip()


def build_reasoning_generation_tokenizer_settings(
    *,
    use_think_tokens: bool = False,
) -> dict[str, bool | str]:
    return {
        "apply_chat_template": True,
        "add_generation_prompt": True,
        "add_thinking": True,
        "thinking_template": "tagged" if use_think_tokens else "reasoning",
    }


def default_reasoning_checkpoint_metadata(*, use_think_tokens: bool = False) -> dict[str, Any]:
    return {
        "reasoning_settings": {
            "use_think_tokens": bool(use_think_tokens),
            "format": "supervised_distillation",
        },
        "inference_tokenizer_settings": build_reasoning_generation_tokenizer_settings(
            use_think_tokens=use_think_tokens
        ),
    }


def resolve_reasoning_trace(
    entry: dict[str, Any],
    *,
    thinking_field: str | None = None,
) -> str:
    return strip_think_tags(
        _resolve_field(
            entry,
            thinking_field,
            DEFAULT_THINKING_FIELDS,
            required=False,
        )
    )


def has_reasoning_trace(
    entry: dict[str, Any],
    *,
    thinking_field: str | None = None,
) -> bool:
    return bool(resolve_reasoning_trace(entry, thinking_field=thinking_field))


def _resolve_field(
    entry: dict[str, Any],
    field_name: str | None,
    candidates: Sequence[str],
    *,
    required: bool = True,
) -> str:
    if field_name:
        if field_name not in entry:
            if required:
                raise KeyError(field_name)
            return ""
        value = entry[field_name]
        if value is None and required:
            raise ValueError(f"Field '{field_name}' cannot be None.")
        return str(value or "").strip()

    for candidate in candidates:
        if candidate in entry and entry[candidate] is not None:
            value = str(entry[candidate]).strip()
            if value:
                return value

    if required:
        available = ", ".join(sorted(entry))
        raise KeyError(
            f"Could not resolve any of the fields {tuple(candidates)} from entry keys: {available}"
        )
    return ""


def format_reasoning_answer(
    entry: dict[str, Any],
    *,
    response_field: str | None = None,
    thinking_field: str | None = None,
    use_think_tokens: bool = False,
    prompt_includes_think_start: bool = False,
) -> str:
    content = strip_think_tags(
        _resolve_field(
            entry,
            response_field,
            DEFAULT_RESPONSE_FIELDS,
            required=True,
        )
    )
    if not content:
        raise ValueError("Missing non-empty response field.")

    thinking = resolve_reasoning_trace(entry, thinking_field=thinking_field)

    if use_think_tokens and prompt_includes_think_start:
        return f"{thinking}</think>\n\n{content}"
    if use_think_tokens:
        return f"<think>{thinking}</think>\n\n{content}"
    if thinking:
        return f"{thinking}\n\n{content}"
    return content


def build_supervised_examples(
    data: Iterable[dict[str, Any]],
    tokenizer,
    *,
    prompt_field: str | None = None,
    response_field: str | None = None,
    thinking_field: str | None = None,
    prompt_formatter: Callable[[str], str] | None = None,
    use_think_tokens: bool = False,
    prompt_includes_think_start: bool = False,
    chat_wrapped_prompt: bool = True,
    add_eos_token: bool = True,
) -> tuple[list[dict[str, int | list[int]]], int]:
    examples: list[dict[str, int | list[int]]] = []
    skipped = 0

    for entry in data:
        try:
            prompt_text = _resolve_field(
                entry,
                prompt_field,
                DEFAULT_PROMPT_FIELDS,
                required=True,
            )
            if prompt_formatter is not None:
                prompt_text = prompt_formatter(prompt_text)

            answer_text = format_reasoning_answer(
                entry,
                response_field=response_field,
                thinking_field=thinking_field,
                use_think_tokens=use_think_tokens,
                prompt_includes_think_start=prompt_includes_think_start,
            )

            prompt_ids = tokenizer.encode(prompt_text, chat_wrapped=chat_wrapped_prompt)
            answer_ids = tokenizer.encode(answer_text, chat_wrapped=False)
            token_ids = prompt_ids + answer_ids

            if add_eos_token and getattr(tokenizer, "eos_token_id", None) is not None:
                token_ids = [*token_ids, int(tokenizer.eos_token_id)]

            if len(token_ids) < 2:
                skipped += 1
                continue

            prompt_len = min(len(prompt_ids), len(token_ids) - 1)
            answer_token_count = len(token_ids) - prompt_len
            if answer_token_count <= 0:
                skipped += 1
                continue

            examples.append(
                {
                    "token_ids": token_ids,
                    "prompt_len": prompt_len,
                }
            )
        except (KeyError, TypeError, ValueError):
            skipped += 1

    return examples, skipped


def filter_examples_by_max_len(
    examples: Iterable[dict[str, int | list[int]]],
    *,
    max_len: int = 2048,
) -> tuple[list[dict[str, int | list[int]]], int]:
    examples = list(examples)
    filtered_examples = [
        example
        for example in examples
        if len(example["token_ids"]) <= max_len
    ]
    removed = len(examples) - len(filtered_examples)
    return filtered_examples, removed


def iter_example_batches(
    examples: Sequence[dict[str, int | list[int]]],
    batch_size: int,
) -> Iterable[Sequence[dict[str, int | list[int]]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    for start_idx in range(0, len(examples), batch_size):
        yield examples[start_idx : start_idx + batch_size]


def prepare_batch_tensors(
    batch_examples: Sequence[dict[str, int | list[int]]],
    *,
    pad_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if pad_id is None:
        raise ValueError("pad_id is required for supervised batching.")
    if not batch_examples:
        raise ValueError("batch_examples cannot be empty.")

    max_input_len = max(len(example["token_ids"]) - 1 for example in batch_examples)
    batch_size = len(batch_examples)
    device = torch.device(device)

    input_ids = torch.full(
        (batch_size, max_input_len),
        fill_value=pad_id,
        dtype=torch.long,
        device=device,
    )
    attn_mask = torch.zeros(
        (batch_size, max_input_len),
        dtype=torch.bool,
        device=device,
    )
    labels = torch.full(
        (batch_size, max_input_len),
        fill_value=IGNORE_INDEX,
        dtype=torch.long,
        device=device,
    )

    supervised_token_count = 0
    for row_idx, example in enumerate(batch_examples):
        token_ids = list(example["token_ids"])
        prompt_len = int(example["prompt_len"])

        input_seq = token_ids[:-1]
        target_seq = token_ids[1:]
        seq_len = len(input_seq)
        offset = 0

        input_ids[row_idx, offset : offset + seq_len] = torch.tensor(
            input_seq,
            dtype=torch.long,
            device=device,
        )
        attn_mask[row_idx, offset : offset + seq_len] = True
        labels[row_idx, offset : offset + seq_len] = torch.tensor(
            target_seq,
            dtype=torch.long,
            device=device,
        )

        answer_start = max(prompt_len - 1, 0)
        if answer_start > 0:
            labels[row_idx, offset : offset + answer_start] = IGNORE_INDEX

        supervised_token_count += max(0, len(token_ids) - prompt_len)

    return input_ids, attn_mask, labels, supervised_token_count


def compute_supervised_batch_loss(
    model,
    batch_examples: Sequence[dict[str, int | list[int]]],
    *,
    pad_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, int]:
    input_ids, _attn_mask, labels, supervised_token_count = prepare_batch_tensors(
        batch_examples,
        pad_id=pad_id,
        device=device,
    )

    logits = model(input_ids)
    batch_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(),
        ignore_index=IGNORE_INDEX,
    )

    return batch_loss, supervised_token_count

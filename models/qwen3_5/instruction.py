from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

DEFAULT_SYSTEM_FIELDS = ("system", "system_prompt", "developer_prompt", "persona")
DEFAULT_INSTRUCTION_FIELDS = ("instruction", "prompt", "question", "query", "user_input", "user")
DEFAULT_INPUT_FIELDS = ("input", "customer_context", "details", "metadata")
DEFAULT_CONTEXT_FIELDS = (
    "context",
    "product_context",
    "knowledge",
    "grounding",
    "retrieved_context",
    "documents",
    "sources",
)
DEFAULT_RESPONSE_FIELDS = ("response", "answer", "output", "assistant", "message_content")


def build_instruction_generation_tokenizer_settings() -> dict[str, bool | str]:
    return {
        "apply_chat_template": True,
        "add_generation_prompt": True,
    }


def _strip_legacy_think_tags(text: Any) -> str:
    return str(text or "").replace("<think>", "").replace("</think>", "").strip()


def default_instruction_checkpoint_metadata(
    *,
    task_type: str = "chatbot_qa",
    company_name: str | None = None,
    assistant_name: str = "Ava",
) -> dict[str, Any]:
    instruction_settings: dict[str, Any] = {
        "format": "supervised_instruction_tuning",
        "task_type": str(task_type).strip() or "chatbot_qa",
        "assistant_name": assistant_name,
    }
    if company_name:
        instruction_settings["company_name"] = company_name

    return {
        "instruction_settings": instruction_settings,
        "inference_tokenizer_settings": build_instruction_generation_tokenizer_settings(),
    }


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            rendered = _stringify_value(item)
            if rendered:
                parts.append(f"{key}: {rendered}")
        return "\n".join(parts).strip()
    if isinstance(value, (list, tuple, set)):
        parts = [_stringify_value(item) for item in value]
        return "\n\n".join(part for part in parts if part).strip()
    return str(value).strip()


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
        value = _stringify_value(entry[field_name])
        if not value and required:
            raise ValueError(f"Field '{field_name}' cannot be empty.")
        return value

    for candidate in candidates:
        if candidate in entry:
            value = _stringify_value(entry[candidate])
            if value:
                return value

    if required:
        available = ", ".join(sorted(entry))
        raise KeyError(
            f"Could not resolve any of the fields {tuple(candidates)} from entry keys: {available}"
        )
    return ""


def build_chatbot_qa_prompt(
    entry: dict[str, Any],
    *,
    company_name: str | None = None,
    assistant_name: str = "Ava",
    system_field: str | None = None,
    instruction_field: str | None = None,
    input_field: str | None = None,
    context_field: str | None = None,
    include_fallback_guidance: bool = True,
) -> str:
    question = _resolve_field(
        entry,
        instruction_field,
        DEFAULT_INSTRUCTION_FIELDS,
        required=True,
    )
    user_input = _resolve_field(
        entry,
        input_field,
        DEFAULT_INPUT_FIELDS,
        required=False,
    )
    context = _resolve_field(
        entry,
        context_field,
        DEFAULT_CONTEXT_FIELDS,
        required=False,
    )
    system_prompt = _resolve_field(
        entry,
        system_field,
        DEFAULT_SYSTEM_FIELDS,
        required=False,
    )

    if not system_prompt:
        company_suffix = f" for {company_name}" if company_name else ""
        system_prompt = (
            f"You are {assistant_name}, a helpful product support assistant{company_suffix}. "
            "Answer using the provided product information when it is available."
        )
        if include_fallback_guidance:
            system_prompt += (
                " If the answer is not supported by the provided information, say you do not know "
                "and ask for the missing detail."
            )

    prompt_parts = [system_prompt]
    if context:
        prompt_parts.append(f"Product information:\n{context}")
    if user_input:
        prompt_parts.append(f"Customer details:\n{user_input}")
    prompt_parts.append(f"Question:\n{question}")
    prompt_parts.append("Answer clearly, accurately, and with actionable steps when relevant.")
    return "\n\n".join(prompt_parts)


def format_instruction_response(
    entry: dict[str, Any],
    *,
    response_field: str | None = None,
    response_formatter: Callable[[str], str] | None = None,
) -> str:
    response = _strip_legacy_think_tags(
        _resolve_field(
            entry,
            response_field,
            DEFAULT_RESPONSE_FIELDS,
            required=True,
        )
    )
    if response_formatter is not None:
        response = str(response_formatter(response)).strip()
    if not response:
        raise ValueError("Missing non-empty response field.")
    return response


def build_instruction_examples(
    data: Iterable[dict[str, Any]],
    tokenizer,
    *,
    company_name: str | None = None,
    assistant_name: str = "Ava",
    system_field: str | None = None,
    instruction_field: str | None = None,
    input_field: str | None = None,
    context_field: str | None = None,
    response_field: str | None = None,
    prompt_builder: Callable[[dict[str, Any]], str] | None = None,
    response_formatter: Callable[[str], str] | None = None,
    chat_wrapped_prompt: bool = True,
    add_eos_token: bool = True,
) -> tuple[list[dict[str, int | list[int]]], int]:
    examples: list[dict[str, int | list[int]]] = []
    skipped = 0

    for entry in data:
        try:
            prompt_text = (
                prompt_builder(entry)
                if prompt_builder is not None
                else build_chatbot_qa_prompt(
                    entry,
                    company_name=company_name,
                    assistant_name=assistant_name,
                    system_field=system_field,
                    instruction_field=instruction_field,
                    input_field=input_field,
                    context_field=context_field,
                )
            )
            prompt_text = _stringify_value(prompt_text)
            if not prompt_text:
                raise ValueError("Prompt builder returned empty text.")

            answer_text = format_instruction_response(
                entry,
                response_field=response_field,
                response_formatter=response_formatter,
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


def build_chatbot_qa_examples(
    data: Iterable[dict[str, Any]],
    tokenizer,
    *,
    company_name: str | None = None,
    assistant_name: str = "Ava",
    system_field: str | None = None,
    instruction_field: str | None = None,
    input_field: str | None = None,
    context_field: str | None = None,
    response_field: str | None = None,
    prompt_builder: Callable[[dict[str, Any]], str] | None = None,
    response_formatter: Callable[[str], str] | None = None,
    chat_wrapped_prompt: bool = True,
    add_eos_token: bool = True,
) -> tuple[list[dict[str, int | list[int]]], int]:
    return build_instruction_examples(
        data,
        tokenizer,
        company_name=company_name,
        assistant_name=assistant_name,
        system_field=system_field,
        instruction_field=instruction_field,
        input_field=input_field,
        context_field=context_field,
        response_field=response_field,
        prompt_builder=prompt_builder,
        response_formatter=response_formatter,
        chat_wrapped_prompt=chat_wrapped_prompt,
        add_eos_token=add_eos_token,
    )

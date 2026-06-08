from __future__ import annotations

ENDOFTEXT_TOKEN = "<|endoftext|>"
CHAT_STOP_MARKERS = (
    ENDOFTEXT_TOKEN,
    "\nQuestion:",
    "\nPrompt>",
    "\nAnswer>",
    "\n## ",
    "\n### ",
    "\n#### ",
)
QWEN_STOP_MARKERS = (
    "<|im_end|>",
    ENDOFTEXT_TOKEN,
)


def resolve_eos_token_id(tokenizer) -> int | None:
    if hasattr(tokenizer, "eos_token_id") and getattr(tokenizer, "eos_token_id") is not None:
        return int(tokenizer.eos_token_id)

    if hasattr(tokenizer, "eot_token"):
        return int(tokenizer.eot_token)

    token_ids = tokenizer.encode(ENDOFTEXT_TOKEN, allowed_special={ENDOFTEXT_TOKEN})
    if not token_ids:
        return None
    return int(token_ids[0])


def build_chat_prompt(prompt: str) -> str:
    return f"Question: {prompt}\nAnswer:"


def trim_at_markers(text: str, stop_markers: tuple[str, ...]) -> str:
    cut_index = len(text)
    for marker in stop_markers:
        marker_index = text.find(marker)
        if marker_index != -1:
            cut_index = min(cut_index, marker_index)
    return text[:cut_index]


def strip_qa_echo(text: str) -> str:
    stripped = text.lstrip()
    qa_markers = (
        ("\nA:", "Q:"),
        ("\nAnswer:", "Question:"),
    )
    for answer_marker, question_prefix in qa_markers:
        if stripped.startswith(question_prefix):
            answer_index = stripped.find(answer_marker)
            if answer_index != -1:
                return stripped[answer_index + len(answer_marker) :].lstrip()
    return text


def strip_qwen_echo(text: str) -> str:
    stripped = text.lstrip()
    prefixes = (
        "<|im_start|>assistant",
        "assistant\n",
    )
    for prefix in prefixes:
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].lstrip()
    return text


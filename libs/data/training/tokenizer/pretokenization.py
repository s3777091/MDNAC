from __future__ import annotations


def pretokenize_text(
    text: str,
    special_tokens: tuple[str, ...],
    allowed_special: set[str],
) -> list[str]:
    tokens: list[str] = []
    ordered_special_tokens = sorted(special_tokens, key=len, reverse=True)
    index = 0
    buffer: list[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            tokens.append("".join(buffer))
            buffer = []

    while index < len(text):
        matched_special = None
        for special_token in ordered_special_tokens:
            if text.startswith(special_token, index):
                matched_special = special_token
                break

        if matched_special is not None:
            if matched_special not in allowed_special:
                raise ValueError(f"Disallowed special token encountered in text: {matched_special}")
            flush_buffer()
            tokens.append(matched_special)
            index += len(matched_special)
            continue

        character = text[index]
        if character == "\r":
            index += 1
            continue

        buffer.append(character)
        index += 1

    flush_buffer()
    return tokens

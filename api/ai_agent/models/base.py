from __future__ import annotations

from typing import Protocol


class ChatModel(Protocol):
    def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """Generate a response from chat-style messages."""

    def ensure_ready(self) -> None:
        """Load or validate expensive resources before serving traffic."""

"""Model adapters for the AI agent."""

from .base import ChatModel
from .openai_model import OpenAIChatModel

__all__ = ["ChatModel", "OpenAIChatModel"]

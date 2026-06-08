"""Inference runtime helpers for Ava checkpoints and exported artifacts."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "GenerateOptions",
    "GenerationResult",
    "InferenceAPI",
    "InferenceSession",
    "chat",
    "generate_text",
    "load_api",
]


def __getattr__(name: str):
    if name in {"GenerateOptions", "InferenceAPI", "chat", "generate_text", "load_api"}:
        module = import_module("api.controllers.inference")
        return getattr(module, name)
    if name == "DEFAULT_CHECKPOINT_PATH":
        module = import_module("interfere.artifacts")
        return getattr(module, name)
    if name == "GenerationResult":
        module = import_module("interfere.backends.base")
        return getattr(module, name)
    if name == "InferenceSession":
        module = import_module("interfere.session")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

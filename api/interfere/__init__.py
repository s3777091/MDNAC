"""ONNX-only protein inference runtime for the standalone api project."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "APISettings",
    "DEFAULT_MODEL_DIR",
    "MODEL_PATH_ENV_VAR",
    "GenerateOptions",
    "GenerationResult",
    "InferenceAPI",
    "InferenceSession",
    "chat",
    "generate_text",
    "load_api",
    "load_api_from_config",
    "load_config",
    "load_inference_artifact",
]


def __getattr__(name: str):
    if name in {"DEFAULT_MODEL_DIR", "MODEL_PATH_ENV_VAR", "load_inference_artifact"}:
        module = import_module("interfere.artifacts")
        return getattr(module, name)
    if name in {"APISettings", "load_config"}:
        module = import_module("interfere.config")
        return getattr(module, name)
    if name in {
        "InferenceAPI",
        "chat",
        "generate_text",
        "load_api",
        "load_api_from_config",
    }:
        module = import_module("interfere.inference")
        return getattr(module, name)
    if name in {"GenerateOptions", "GenerationResult", "InferenceSession"}:
        module = import_module("interfere.session")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

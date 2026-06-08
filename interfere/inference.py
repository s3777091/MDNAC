from __future__ import annotations

from api.controllers.inference import (
    GenerateOptions,
    InferenceAPI,
    chat,
    generate_text,
    load_api,
)
from .artifacts import DEFAULT_CHECKPOINT_PATH, detect_model_family
from .backends.base import GenerationResult
from .backends.qwen3_5 import (
    load_qwen_tokenizer as _load_qwen_tokenizer,
    resolve_qwen_tokenizer_path as _resolve_qwen_tokenizer_path,
    resolve_qwen_tokenizer_settings as _resolve_qwen_tokenizer_settings,
)
from .session import InferenceSession

_detect_model_family = detect_model_family

__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "GenerateOptions",
    "GenerationResult",
    "InferenceAPI",
    "InferenceSession",
    "chat",
    "generate_text",
    "load_api",
    "_detect_model_family",
    "_load_qwen_tokenizer",
    "_resolve_qwen_tokenizer_path",
    "_resolve_qwen_tokenizer_settings",
]

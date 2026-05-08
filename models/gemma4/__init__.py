from importlib import import_module

__all__ = [
    "DEFAULT_GEMMA4_E2B_IT_REPO_ID",
    "DEFAULT_GEMMA4_E2B_REPO_ID",
    "GEMMA4_CONFIG_E2B",
    "GEMMA4_E2B_LAYER_TYPES",
    "Gemma4KVCache",
    "Gemma4Model",
    "Gemma4Tokenizer",
    "GroupedQueryAttention",
    "KVCache",
    "assign_parameter",
    "build_gemma4_e2b_config",
    "build_model",
    "compute_rope_params",
    "copy_model_config",
    "download_from_huggingface",
    "ensure_gemma4_repo_snapshot",
    "ensure_gemma4_tokenizer",
    "generate_text_simple",
    "generate_text_simple_stream",
    "load_gemma4_safetensor_weights",
    "load_weights_into_gemma4",
    "text_to_token_ids",
    "token_ids_to_text",
]

_SYMBOL_TO_MODULE = {
    "DEFAULT_GEMMA4_E2B_IT_REPO_ID": "ch05",
    "DEFAULT_GEMMA4_E2B_REPO_ID": "ch05",
    "GEMMA4_CONFIG_E2B": "ch05",
    "GEMMA4_E2B_LAYER_TYPES": "ch05",
    "Gemma4KVCache": "gemma4_transformers",
    "Gemma4Model": "ch05",
    "Gemma4Tokenizer": "ch05",
    "GroupedQueryAttention": "gemma4_transformers",
    "KVCache": "gemma4_transformers",
    "assign_parameter": "pretrained",
    "build_gemma4_e2b_config": "ch05",
    "build_model": "ch05",
    "compute_rope_params": "gemma4_transformers",
    "copy_model_config": "ch05",
    "download_from_huggingface": "ch05",
    "ensure_gemma4_repo_snapshot": "pretrained",
    "ensure_gemma4_tokenizer": "pretrained",
    "generate_text_simple": "ch05",
    "generate_text_simple_stream": "ch05",
    "load_gemma4_safetensor_weights": "pretrained",
    "load_weights_into_gemma4": "pretrained",
    "text_to_token_ids": "ch05",
    "token_ids_to_text": "ch05",
}


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f".{_SYMBOL_TO_MODULE[name]}", __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)

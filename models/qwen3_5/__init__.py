from importlib import import_module

__all__ = [
    "KVCache",
    "QWEN3_5_CONFIG_0_8B",
    "QWEN3_5_CONFIG_2B",
    "Qwen3_5Model",
    "Qwen3_5Tokenizer",
    "Qwen3_5DatasetV1",
    "build_model",
    "build_chatbot_qa_examples",
    "build_chatbot_qa_prompt",
    "build_instruction_examples",
    "build_instruction_generation_tokenizer_settings",
    "calc_loss_batch",
    "calc_loss_loader",
    "create_moon_optimizers",
    "create_dataloader_v1",
    "create_muon_optimizers",
    "default_instruction_checkpoint_metadata",
    "download_from_huggingface",
    "evaluate_model",
    "export_checkpoint_to_hf_directory",
    "format_instruction_response",
    "generate_and_print_sample",
    "generate_text_simple",
    "generate_text_simple_stream",
    "DEFAULT_OLLAMA_STOP_TOKENS",
    "DEFAULT_OLLAMA_TEMPLATE",
    "build_default_ollama_system_prompt",
    "build_hf_text_config",
    "convert_repo_state_dict_to_hf",
    "ExportedQwen3_5HfArtifact",
    "load_export_tokenizer",
    "resolve_checkpoint_tokenizer",
    "text_to_token_ids",
    "token_ids_to_text",
    "train_model_simple",
]

_SYMBOL_TO_MODULE = {
    "KVCache": "modeling",
    "QWEN3_5_CONFIG_0_8B": "modeling",
    "QWEN3_5_CONFIG_2B": "modeling",
    "Qwen3_5Model": "modeling",
    "Qwen3_5Tokenizer": "modeling",
    "Qwen3_5DatasetV1": "data",
    "build_model": "modeling",
    "build_chatbot_qa_examples": "instruction",
    "build_chatbot_qa_prompt": "instruction",
    "build_instruction_examples": "instruction",
    "build_instruction_generation_tokenizer_settings": "instruction",
    "calc_loss_batch": "modeling",
    "calc_loss_loader": "modeling",
    "create_moon_optimizers": "modeling",
    "create_dataloader_v1": "data",
    "create_muon_optimizers": "modeling",
    "default_instruction_checkpoint_metadata": "instruction",
    "download_from_huggingface": "modeling",
    "evaluate_model": "modeling",
    "export_checkpoint_to_hf_directory": "support.export",
    "format_instruction_response": "instruction",
    "generate_and_print_sample": "modeling",
    "generate_text_simple": "modeling",
    "generate_text_simple_stream": "modeling",
    "DEFAULT_OLLAMA_STOP_TOKENS": "support.export",
    "DEFAULT_OLLAMA_TEMPLATE": "support.export",
    "build_default_ollama_system_prompt": "support.export",
    "build_hf_text_config": "support.export",
    "convert_repo_state_dict_to_hf": "support.export",
    "ExportedQwen3_5HfArtifact": "support.export",
    "load_export_tokenizer": "support.export",
    "resolve_checkpoint_tokenizer": "support.export",
    "text_to_token_ids": "modeling",
    "token_ids_to_text": "modeling",
    "train_model_simple": "modeling",
}


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f".{_SYMBOL_TO_MODULE[name]}", __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import DEFAULT_MODEL_DIR
from .config import generation_kwargs, load_config
from .session import GenerateOptions, GenerationResult, InferenceSession


class InferenceAPI:
    def __init__(self, session: InferenceSession) -> None:
        self.session = session

    @classmethod
    def load(
        cls,
        *,
        model_path: str | Path | None = None,
        device_name: str = "auto",
    ) -> "InferenceAPI":
        return cls(InferenceSession.load(model_path=model_path, device_name=device_name))

    def generate_protein(
        self,
        prompt: str = "",
        *,
        options: GenerateOptions | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        return self.session.generate_protein(prompt, options=options, **kwargs)

    def generate_text(self, prompt: str = "", **kwargs: Any) -> GenerationResult:
        return self.generate_protein(prompt, **kwargs)


def load_api(
    *,
    model_path: str | Path | None = None,
    device_name: str = "auto",
) -> InferenceAPI:
    return InferenceAPI.load(model_path=model_path, device_name=device_name)


def load_api_from_config(
    *,
    config_path: str | Path | None = None,
    environment: str | None = None,
) -> InferenceAPI:
    settings = load_config(config_path=config_path, environment=environment)
    return load_api(model_path=settings.model.path, device_name=settings.model.device)


def generate_text(
    prompt: str = "",
    *,
    model_path: str | Path | None = None,
    device_name: str = "auto",
    **kwargs: Any,
) -> GenerationResult:
    api = load_api(model_path=model_path, device_name=device_name)
    return api.generate_text(prompt, **kwargs)


def chat(prompt: str = "", **kwargs: Any) -> GenerationResult:
    return generate_text(prompt, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate a protein sequence with the ONNX model stored under api/data/model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Exact `.onnx` or sidecar `.json` path. Defaults to {DEFAULT_MODEL_DIR}.",
    )
    parser.add_argument("--config", default=None, help="Path to api/config.yaml.")
    parser.add_argument("--env", default=None, help="Environment name from config.yaml.")
    parser.add_argument("--prompt", default="", help="Protein prefix or full prompt.")
    parser.add_argument("--device", default=None, choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    settings = load_config(config_path=args.config, environment=args.env)
    kwargs = generation_kwargs(settings.generation)
    for key, value in {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
    }.items():
        if value is not None:
            kwargs[key] = value

    result = generate_text(
        args.prompt,
        model_path=args.model or settings.model.path,
        device_name=args.device or settings.model.device,
        **kwargs,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


__all__ = [
    "DEFAULT_MODEL_DIR",
    "GenerateOptions",
    "GenerationResult",
    "InferenceAPI",
    "InferenceSession",
    "chat",
    "generate_text",
    "load_api",
    "load_api_from_config",
]


if __name__ == "__main__":
    main()

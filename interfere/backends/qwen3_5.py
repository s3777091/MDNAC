from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from train.models.qwen3_5.ch05 import (
    Qwen3_5Tokenizer,
    build_model as build_qwen3_5_model,
    text_to_token_ids as qwen3_5_text_to_token_ids,
    token_ids_to_text as qwen3_5_token_ids_to_text,
)
from train.models.qwen3_5.ch07 import build_instruction_generation_tokenizer_settings
from train.models.qwen3_5.reasoning import build_reasoning_generation_tokenizer_settings
from train.pipeline.runtime.device import resolve_device

from ..artifacts import ResolvedArtifact
from ..onnx_runtime import create_onnx_session
from .base import GenerationResult, generate_tokens_with_logits
from .shared import QWEN_STOP_MARKERS, resolve_eos_token_id, strip_qwen_echo, trim_at_markers


def resolve_qwen_tokenizer_path(
    metadata: dict[str, Any],
    reference_path: Path,
) -> Path:
    explicit_path = metadata.get("tokenizer_file_path")
    candidates: list[Path] = []

    if explicit_path:
        explicit = Path(str(explicit_path)).expanduser()
        candidates.append(explicit)

        tokenizer_name = explicit.name or "tokenizer.json"
        tokenizer_parent_name = explicit.parent.name
        if tokenizer_parent_name:
            candidates.extend(
                [
                    Path("data") / "pretrained" / tokenizer_parent_name / tokenizer_name,
                    Path("data") / "tokenizers" / tokenizer_parent_name / tokenizer_name,
                ]
            )

    candidates.extend(
        [
            reference_path.parent / "tokenizer.json",
            reference_path.parent / "data" / "pretrained" / "Qwen3.5-0.8B" / "tokenizer.json",
            reference_path.parent / "data" / "tokenizers" / "Qwen3.5-0.8B" / "tokenizer.json",
            Path("data") / "pretrained" / "Qwen3.5-0.8B" / "tokenizer.json",
            Path("data") / "tokenizers" / "Qwen3.5-0.8B" / "tokenizer.json",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.expanduser()
        try:
            resolved = normalized.resolve()
        except OSError:
            resolved = normalized
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    searched_paths = "\n".join(f"- {path}" for path in seen)
    raise FileNotFoundError(
        "Unable to resolve a local tokenizer.json for the Qwen artifact.\n"
        f"Searched:\n{searched_paths}"
    )


def resolve_qwen_tokenizer_settings(metadata: dict[str, Any]) -> dict[str, Any]:
    explicit_settings = metadata.get("inference_tokenizer_settings") or metadata.get(
        "tokenizer_settings"
    )
    if isinstance(explicit_settings, dict):
        settings = dict(explicit_settings)
    elif metadata.get("reasoning_settings") or metadata.get("reasoning_mode"):
        reasoning_settings = metadata.get("reasoning_settings")
        settings = build_reasoning_generation_tokenizer_settings(
            use_think_tokens=bool(
                isinstance(reasoning_settings, dict)
                and reasoning_settings.get("use_think_tokens")
            )
        )
    elif (
        metadata.get("instruction_settings")
        or metadata.get("instruction_mode")
        or metadata.get("chatbot_settings")
    ):
        settings = build_instruction_generation_tokenizer_settings()
    else:
        settings = {
            "apply_chat_template": False,
            "add_generation_prompt": False,
            "add_thinking": False,
            "thinking_template": "tagged",
        }

    settings.setdefault("apply_chat_template", False)
    settings.setdefault("add_generation_prompt", False)
    settings.setdefault("add_thinking", False)
    settings.setdefault("thinking_template", "tagged")
    return settings


def load_qwen_tokenizer(
    metadata: dict[str, Any],
    reference_path: Path,
    ) -> Qwen3_5Tokenizer:
    tokenizer_path = resolve_qwen_tokenizer_path(metadata, reference_path)
    return Qwen3_5Tokenizer(
        tokenizer_file_path=tokenizer_path,
        **resolve_qwen_tokenizer_settings(metadata),
    )


class Qwen3_5InferenceBackend:
    def __init__(
        self,
        *,
        checkpoint: dict[str, Any],
        source_path: Path,
        artifact_format: str,
        tokenizer: Qwen3_5Tokenizer,
        device: torch.device,
        model: torch.nn.Module | None = None,
        onnx_session=None,
    ) -> None:
        self.checkpoint = checkpoint
        self.source_path = source_path
        self.artifact_format = artifact_format
        self.tokenizer = tokenizer
        self.device = device
        self.model = model
        self.onnx_session = onnx_session
        self.model_family = "qwen3_5"
        self.eos_token_id = resolve_eos_token_id(tokenizer)

    @classmethod
    def load(
        cls,
        artifact: ResolvedArtifact,
        *,
        device_name: str = "auto",
    ) -> "Qwen3_5InferenceBackend":
        if artifact.metadata.get("classifier_settings"):
            raise ValueError(
                "Classifier checkpoints are no longer supported in this repo. "
                "Load a generation checkpoint instead."
            )
        tokenizer = load_qwen_tokenizer(artifact.metadata, artifact.reference_path)
        if artifact.artifact_format == "pytorch":
            device = resolve_device(device_name)
            if artifact.checkpoint is None:
                raise ValueError("PyTorch inference requires checkpoint data.")
            model = build_qwen3_5_model(artifact.model_config)
            model.load_state_dict(artifact.checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            return cls(
                checkpoint=artifact.checkpoint,
                source_path=artifact.source_path,
                artifact_format=artifact.artifact_format,
                tokenizer=tokenizer,
                device=device,
                model=model,
            )

        if artifact.onnx_path is None:
            raise ValueError("ONNX inference requires an `.onnx` artifact path.")
        onnx_session, device = create_onnx_session(artifact.onnx_path, device_name=device_name)
        return cls(
            checkpoint=artifact.metadata,
            source_path=artifact.source_path,
            artifact_format=artifact.artifact_format,
            tokenizer=tokenizer,
            device=device,
            onnx_session=onnx_session,
        )

    def _generate_with_onnx(
        self,
        encoded: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
    ) -> torch.Tensor:
        if self.onnx_session is None:
            raise RuntimeError("ONNX session is not initialized.")

        def _next_logits(idx_cond: torch.Tensor) -> torch.Tensor:
            outputs = self.onnx_session.run(
                ["logits"],
                {"input_ids": idx_cond.detach().cpu().numpy()},
            )
            return torch.from_numpy(outputs[0])

        return generate_tokens_with_logits(
            _next_logits,
            encoded,
            max_new_tokens=max_new_tokens,
            context_size=int(self.checkpoint["model_config"]["context_length"]),
            temperature=temperature,
            top_k=top_k,
            eos_token_id=self.eos_token_id,
        )

    def _generate_with_pytorch(
        self,
        encoded: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("PyTorch model is not initialized.")

        def _next_logits(idx_cond: torch.Tensor) -> torch.Tensor:
            if self.model is None:
                raise RuntimeError("PyTorch model is not initialized.")
            with torch.no_grad():
                return self.model(idx_cond.to(self.device)).detach().cpu()

        return generate_tokens_with_logits(
            _next_logits,
            encoded.detach().cpu(),
            max_new_tokens=max_new_tokens,
            context_size=int(self.checkpoint["model_config"]["context_length"]),
            temperature=temperature,
            top_k=top_k,
            eos_token_id=self.eos_token_id,
        )

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 120,
        temperature: float = 0.0,
        top_k: int | None = None,
        qa_mode: bool = False,
    ) -> GenerationResult:
        raw_prompt = prompt.strip() or prompt
        chat_wrapped = bool(qa_mode and getattr(self.tokenizer, "apply_chat_template", False))
        encoded = qwen3_5_text_to_token_ids(
            raw_prompt,
            self.tokenizer,
            chat_wrapped=chat_wrapped,
        )

        if self.model is not None:
            token_ids = self._generate_with_pytorch(
                encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        else:
            token_ids = self._generate_with_onnx(
                encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        generated_text = qwen3_5_token_ids_to_text(token_ids, self.tokenizer)
        full_text = trim_at_markers(generated_text, QWEN_STOP_MARKERS)

        answer_text = ""
        if token_ids.shape[1] > encoded.shape[1]:
            answer_text = qwen3_5_token_ids_to_text(token_ids[:, encoded.shape[1] :], self.tokenizer)
        answer_text = trim_at_markers(answer_text, QWEN_STOP_MARKERS)
        answer_text = strip_qwen_echo(answer_text)

        return GenerationResult(
            prompt=raw_prompt,
            full_text=full_text,
            answer_text=answer_text.lstrip(" \r\n").rstrip(),
            model_family=self.model_family,
        )

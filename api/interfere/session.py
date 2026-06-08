from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import DEFAULT_MODEL_DIR, ResolvedArtifact, load_inference_artifact
from .onnx_runtime import create_onnx_session
from .tokenizer import (
    ProteinTokenizer,
    extract_protein_sequence,
    normalize_protein_prompt,
)


@dataclass(frozen=True)
class GenerateOptions:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_k: int | None = None
    seed: int | None = None
    stop_at_endoftext: bool = True
    ensure_protein_prompt: bool = True


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    full_text: str
    generated_text: str
    protein_sequence: str
    model_family: str
    model_path: str

    @property
    def answer_text(self) -> str:
        return self.protein_sequence

    def to_dict(self) -> dict[str, str]:
        return {
            "prompt": self.prompt,
            "full_text": self.full_text,
            "generated_text": self.generated_text,
            "protein_sequence": self.protein_sequence,
            "answer_text": self.answer_text,
            "model_family": self.model_family,
            "model_path": self.model_path,
        }


class InferenceSession:
    def __init__(
        self,
        *,
        artifact: ResolvedArtifact,
        onnx_session,
        providers: tuple[str, ...],
        tokenizer: ProteinTokenizer,
    ) -> None:
        self.artifact = artifact
        self.onnx_session = onnx_session
        self.providers = providers
        self.tokenizer = tokenizer
        self.model_family = artifact.model_family
        self.model_config = artifact.model_config
        self.context_length = artifact.context_length
        self.source_path = artifact.source_path
        self.onnx_path = artifact.onnx_path
        self.metadata_path = artifact.metadata_path
        self.input_name = _resolve_io_name(
            artifact.input_names,
            [item.name for item in onnx_session.get_inputs()],
            kind="input",
        )
        self.output_name = _resolve_io_name(
            artifact.output_names,
            [item.name for item in onnx_session.get_outputs()],
            kind="output",
        )

    @classmethod
    def load(
        cls,
        *,
        model_path: str | Path | None = None,
        device_name: str = "auto",
    ) -> "InferenceSession":
        artifact = load_inference_artifact(model_path)
        onnx_session, providers = create_onnx_session(
            artifact.onnx_path,
            device_name=device_name,
        )
        tokenizer = ProteinTokenizer.from_payload(artifact.tokenizer_payload)
        return cls(
            artifact=artifact,
            onnx_session=onnx_session,
            providers=providers,
            tokenizer=tokenizer,
        )

    def generate_protein(
        self,
        prompt: str = "",
        *,
        options: GenerateOptions | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        stop_at_endoftext: bool | None = None,
        ensure_protein_prompt: bool | None = None,
    ) -> GenerationResult:
        resolved_options = _merge_options(
            options,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            stop_at_endoftext=stop_at_endoftext,
            ensure_protein_prompt=ensure_protein_prompt,
        )
        normalized_prompt = normalize_protein_prompt(
            prompt,
            ensure_start_token=resolved_options.ensure_protein_prompt,
        )
        encoded = self.tokenizer.encode(normalized_prompt)
        if not encoded:
            raise ValueError("Prompt encoded to zero tokens.")

        token_ids = np.asarray([encoded], dtype=np.int64)
        generated_start = token_ids.shape[1]
        rng = np.random.default_rng(resolved_options.seed)

        for _ in range(resolved_options.max_new_tokens):
            input_ids = token_ids[:, -self.context_length :].astype(np.int64, copy=False)
            outputs = self.onnx_session.run(
                [self.output_name],
                {self.input_name: input_ids},
            )
            logits = np.asarray(outputs[0])
            if logits.ndim != 3:
                raise RuntimeError(
                    f"Expected ONNX logits with shape [batch, sequence, vocab], got {logits.shape}."
                )
            next_token_id = _select_next_token(
                logits[:, -1, :],
                temperature=resolved_options.temperature,
                top_k=resolved_options.top_k,
                rng=rng,
            )
            token_ids = np.concatenate([token_ids, next_token_id], axis=1)
            if (
                resolved_options.stop_at_endoftext
                and int(next_token_id[0, 0]) == self.tokenizer.eos_token_id
            ):
                break

        full_ids = token_ids[0].tolist()
        generated_ids = token_ids[0, generated_start:].tolist()
        full_text = self.tokenizer.decode(full_ids)
        generated_text = self.tokenizer.decode(generated_ids) if generated_ids else ""
        return GenerationResult(
            prompt=normalized_prompt,
            full_text=full_text,
            generated_text=generated_text,
            protein_sequence=extract_protein_sequence(full_text),
            model_family=self.model_family,
            model_path=str(self.onnx_path),
        )

    def generate_text(self, prompt: str, **kwargs: Any) -> GenerationResult:
        return self.generate_protein(prompt, **kwargs)


def _merge_options(
    options: GenerateOptions | None,
    *,
    max_new_tokens: int | None,
    temperature: float | None,
    top_k: int | None,
    seed: int | None,
    stop_at_endoftext: bool | None,
    ensure_protein_prompt: bool | None,
) -> GenerateOptions:
    base = options or GenerateOptions()
    resolved = GenerateOptions(
        max_new_tokens=base.max_new_tokens if max_new_tokens is None else int(max_new_tokens),
        temperature=base.temperature if temperature is None else float(temperature),
        top_k=base.top_k if top_k is None else top_k,
        seed=base.seed if seed is None else int(seed),
        stop_at_endoftext=(
            base.stop_at_endoftext if stop_at_endoftext is None else bool(stop_at_endoftext)
        ),
        ensure_protein_prompt=(
            base.ensure_protein_prompt
            if ensure_protein_prompt is None
            else bool(ensure_protein_prompt)
        ),
    )
    if resolved.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be greater than or equal to 0.")
    if resolved.temperature < 0.0:
        raise ValueError("temperature must be greater than or equal to 0.")
    if resolved.top_k is not None and resolved.top_k <= 0:
        raise ValueError("top_k must be positive when provided.")
    return resolved


def _select_next_token(
    next_logits: np.ndarray,
    *,
    temperature: float,
    top_k: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    logits = np.asarray(next_logits, dtype=np.float64)
    if logits.shape[0] != 1:
        raise RuntimeError("This API currently supports batch size 1 generation only.")

    if top_k is not None and top_k < logits.shape[-1]:
        top_indices = np.argpartition(logits, -top_k, axis=-1)[:, -top_k:]
        masked = np.full_like(logits, -np.inf)
        np.put_along_axis(masked, top_indices, np.take_along_axis(logits, top_indices, axis=-1), axis=-1)
        logits = masked

    if temperature > 0.0:
        logits = logits / temperature
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities = probabilities / np.sum(probabilities, axis=-1, keepdims=True)
        next_id = int(rng.choice(logits.shape[-1], p=probabilities[0]))
    else:
        next_id = int(np.argmax(logits, axis=-1)[0])
    return np.asarray([[next_id]], dtype=np.int64)


def _resolve_io_name(preferred_names: tuple[str, ...], available_names: list[str], *, kind: str) -> str:
    for name in preferred_names:
        if name in available_names:
            return name
    if len(available_names) == 1:
        return available_names[0]
    available = ", ".join(available_names)
    preferred = ", ".join(preferred_names)
    raise ValueError(
        f"Unable to choose ONNX {kind}. Preferred {{{preferred}}}; available {{{available}}}."
    )


__all__ = [
    "DEFAULT_MODEL_DIR",
    "GenerateOptions",
    "GenerationResult",
    "InferenceSession",
]

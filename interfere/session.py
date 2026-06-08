from __future__ import annotations

from pathlib import Path

from .artifacts import DEFAULT_CHECKPOINT_PATH, load_inference_artifact
from .backends.qwen3_5 import Qwen3_5InferenceBackend


class InferenceSession:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.checkpoint = backend.checkpoint
        self.model = getattr(backend, "model", None)
        self.tokenizer = backend.tokenizer
        self.device = backend.device
        self.model_family = backend.model_family
        self.artifact_format = backend.artifact_format
        self.source_path = backend.source_path

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        device_name: str = "auto",
    ) -> "InferenceSession":
        artifact = load_inference_artifact(checkpoint_path)
        if artifact.model_family != "qwen3_5":
            raise ValueError(
                "Only Qwen3.5 inference artifacts are supported in this repo. "
                f"Received: {artifact.model_family}"
            )
        backend = Qwen3_5InferenceBackend.load(artifact, device_name=device_name)
        return cls(backend)

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 120,
        temperature: float = 0.0,
        top_k: int | None = None,
        qa_mode: bool = False,
    ):
        return self.backend.generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            qa_mode=qa_mode,
        )

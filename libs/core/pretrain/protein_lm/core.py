from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from libs.core.interfaces import CausalLMBatch, IGNORE_INDEX
from libs.core.mdc.config import MDCModelConfig
from libs.data.training.tokenizer import SequenceTokenizer
from libs.data.training.tokenizer.constants import DEFAULT_VOCAB_SIZES
from .support.backbone import (
    QWEN3_5_BACKBONE_FAMILY,
    QWEN3_5_PROTEIN_MODEL_FAMILY,
    is_supported_protein_checkpoint_family,
)


PROTEIN_START_TOKEN = "<|protein|>"
PROTEIN_END_TOKEN = "<|endoftext|>"


@dataclass(slots=True, frozen=True)
class ProteinTokenizerArtifact:
    tokenizer: SequenceTokenizer
    tokenizer_map_path: Path
    rebuilt: bool

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


@dataclass(slots=True, frozen=True)
class ProteinCausalLMExample:
    input_ids: torch.Tensor
    labels: torch.Tensor


class ProteinCausalLMTextDataset(Dataset[ProteinCausalLMExample]):
    def __init__(
        self,
        text: str,
        tokenizer: SequenceTokenizer,
        *,
        context_length: int,
        stride: int | None = None,
    ) -> None:
        if context_length <= 1:
            raise ValueError("context_length must be greater than 1.")

        self.context_length = int(context_length)
        self.stride = int(stride or context_length)
        if self.stride <= 0:
            raise ValueError("stride must be greater than 0.")

        token_ids = tokenizer.encode(text)
        if len(token_ids) < 2:
            raise ValueError("Protein corpus must encode to at least 2 tokens.")

        self.examples = _build_causal_lm_examples(
            token_ids,
            context_length=self.context_length,
            stride=self.stride,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ProteinCausalLMExample:
        return self.examples[index]


class ProteinCausalLMBatchCollator:
    def __init__(
        self,
        *,
        pad_token_id: int,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.ignore_index = int(ignore_index)

    def __call__(self, batch: Sequence[ProteinCausalLMExample]) -> CausalLMBatch:
        if not batch:
            raise ValueError("batch must not be empty.")

        max_length = max(int(example.input_ids.numel()) for example in batch)
        input_ids = torch.full((len(batch), max_length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_length), self.ignore_index, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)

        for row_index, example in enumerate(batch):
            length = int(example.input_ids.numel())
            input_ids[row_index, :length] = example.input_ids
            labels[row_index, :length] = example.labels
            attention_mask[row_index, :length] = 1

        return CausalLMBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


def load_protein_corpus_text(train_text_path: Path | str) -> str:
    resolved_path = Path(train_text_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Protein train.txt not found: {resolved_path}")

    text = resolved_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Protein corpus is empty: {resolved_path}")
    if PROTEIN_START_TOKEN not in text:
        raise ValueError(f"Protein corpus must contain {PROTEIN_START_TOKEN}.")
    if PROTEIN_END_TOKEN not in text:
        raise ValueError(f"Protein corpus must contain {PROTEIN_END_TOKEN}.")
    return text


def split_protein_corpus_text(
    text: str,
    *,
    train_ratio: float = 0.9,
) -> tuple[str, str]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        normalized = text if text.endswith("\n") else text + "\n"
        return normalized, normalized

    split_index = int(len(lines) * train_ratio)
    split_index = max(1, min(len(lines) - 1, split_index))
    train_text = "\n".join(lines[:split_index]) + "\n"
    val_text = "\n".join(lines[split_index:]) + "\n"
    return train_text, val_text


def build_or_load_protein_tokenizer(
    train_text_path: Path | str,
    *,
    tokenizer_map_path: Path | str | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZES["protein"],
    rebuild: bool = False,
) -> ProteinTokenizerArtifact:
    resolved_train_path = Path(train_text_path)
    resolved_tokenizer_path = (
        Path(tokenizer_map_path)
        if tokenizer_map_path is not None
        else resolved_train_path.with_name("tokenizer_map.json")
    )

    if resolved_tokenizer_path.exists() and not rebuild:
        tokenizer = SequenceTokenizer.load_map(resolved_tokenizer_path)
        if tokenizer.sequence_type != "protein":
            raise ValueError("Loaded tokenizer_map.json is not a protein tokenizer.")
        return ProteinTokenizerArtifact(
            tokenizer=tokenizer,
            tokenizer_map_path=resolved_tokenizer_path,
            rebuilt=False,
        )

    text = load_protein_corpus_text(resolved_train_path)
    tokenizer = SequenceTokenizer.from_text(
        text,
        sequence_type="protein",
        vocab_size=vocab_size,
    )
    resolved_tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save_map(resolved_tokenizer_path)
    return ProteinTokenizerArtifact(
        tokenizer=tokenizer,
        tokenizer_map_path=resolved_tokenizer_path,
        rebuilt=True,
    )


def create_protein_lm_dataloader(
    text: str,
    tokenizer: SequenceTokenizer,
    *,
    context_length: int,
    stride: int | None = None,
    batch_size: int = 4,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[CausalLMBatch]:
    dataset = ProteinCausalLMTextDataset(
        text,
        tokenizer,
        context_length=context_length,
        stride=stride,
    )
    collator = ProteinCausalLMBatchCollator(
        pad_token_id=tokenizer.str_to_int["<|pad|>"],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )


def save_protein_pretrain_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | Sequence[torch.optim.Optimizer] | None,
    model_config: MDCModelConfig | Mapping[str, object],
    tokenizer: SequenceTokenizer,
    tokenizer_map_path: Path | str,
    epoch: int,
    global_step: int,
    tokens_seen: int,
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    training_args: Mapping[str, object] | None = None,
    best_val_loss: float | None = None,
    extra: Mapping[str, object] | None = None,
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model_config, MDCModelConfig):
        serialized_model_config: Mapping[str, object] = model_config.to_dict()
    else:
        serialized_model_config = dict(model_config)

    checkpoint: dict[str, Any] = {
        "model_family": QWEN3_5_PROTEIN_MODEL_FAMILY,
        "backbone_family": QWEN3_5_BACKBONE_FAMILY,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": _optimizer_state_dict(optimizer),
        "model_config": dict(serialized_model_config),
        "tokenizer_map": json.loads(tokenizer.to_json()),
        "tokenizer_map_path": str(Path(tokenizer_map_path).resolve()),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "tokens_seen": int(tokens_seen),
        "train_losses": list(train_losses),
        "val_losses": list(val_losses),
        "training_args": dict(training_args or {}),
        "best_val_loss": best_val_loss,
    }
    if extra:
        checkpoint.update(dict(extra))

    torch.save(checkpoint, resolved_path)
    return resolved_path


def load_protein_pretrain_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | Sequence[torch.optim.Optimizer] | None = None,
    map_location: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    if not is_supported_protein_checkpoint_family(checkpoint.get("model_family")):
        raise ValueError(
            "Unsupported protein checkpoint family: "
            f"{checkpoint.get('model_family')!r}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        _load_optimizer_state_dict(optimizer, checkpoint["optimizer_state_dict"])
    return checkpoint


def _optimizer_state_dict(
    optimizer: torch.optim.Optimizer | Sequence[torch.optim.Optimizer] | None,
):
    if optimizer is None:
        return None
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer.state_dict()
    return [opt.state_dict() for opt in optimizer]


def _load_optimizer_state_dict(
    optimizer: torch.optim.Optimizer | Sequence[torch.optim.Optimizer],
    state_dict,
) -> None:
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.load_state_dict(state_dict)
        return

    optimizers = list(optimizer)
    if len(optimizers) != len(state_dict):
        raise ValueError("Optimizer count does not match checkpoint optimizer_state_dict.")
    for opt, state in zip(optimizers, state_dict, strict=True):
        opt.load_state_dict(state)


def generate_protein_text(
    model: torch.nn.Module,
    tokenizer: SequenceTokenizer,
    prompt: str = PROTEIN_START_TOKEN,
    *,
    device: torch.device | str,
    max_new_tokens: int = 64,
    context_length: int | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    use_cache: bool = True,
    stop_at_endoftext: bool = True,
) -> str:
    model.eval()
    token_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    eos_token_id = tokenizer.str_to_int[PROTEIN_END_TOKEN]
    context_size = int(context_length or getattr(model, "cfg", {}).get("context_length", token_ids.size(1)))
    cache = (
        _create_generation_cache(model)
        if use_cache and token_ids.size(1) + max_new_tokens <= context_size
        else None
    )

    with torch.no_grad():
        if cache is not None:
            _reset_generation_cache(model, cache)
            logits = model(token_ids[:, -context_size:], cache=cache)

            for _ in range(max_new_tokens):
                next_token = _select_next_token(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                )
                token_ids = torch.cat([token_ids, next_token], dim=1)
                if stop_at_endoftext and int(next_token.item()) == eos_token_id:
                    break
                logits = model(next_token, cache=cache)
        else:
            for _ in range(max_new_tokens):
                logits = model(token_ids[:, -context_size:])
                next_token = _select_next_token(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                )

                token_ids = torch.cat([token_ids, next_token], dim=1)
                if stop_at_endoftext and int(next_token.item()) == eos_token_id:
                    break

    return tokenizer.decode(token_ids.squeeze(0).tolist())


def _create_generation_cache(model: torch.nn.Module):
    create_cache = getattr(model, "create_kv_cache", None)
    if callable(create_cache):
        return create_cache()
    return None


def _reset_generation_cache(model: torch.nn.Module, cache) -> None:
    reset_cache = getattr(model, "reset_kv_cache", None)
    if callable(reset_cache):
        try:
            reset_cache(cache)
            return
        except TypeError:
            reset_cache()

    reset = getattr(cache, "reset", None)
    if callable(reset):
        reset()


def _select_next_token(
    next_logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    if top_k is not None:
        top_logits, _ = torch.topk(next_logits, top_k)
        min_value = top_logits[:, -1].unsqueeze(-1)
        next_logits = torch.where(
            next_logits < min_value,
            next_logits.new_full((), float("-inf")),
            next_logits,
        )

    if temperature > 0.0:
        probs = torch.softmax(next_logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    return torch.argmax(next_logits, dim=-1, keepdim=True)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _build_causal_lm_examples(
    token_ids: Sequence[int],
    *,
    context_length: int,
    stride: int,
) -> list[ProteinCausalLMExample]:
    if len(token_ids) <= context_length + 1:
        return [
            ProteinCausalLMExample(
                input_ids=torch.tensor(token_ids[:-1], dtype=torch.long),
                labels=torch.tensor(token_ids[1:], dtype=torch.long),
            )
        ]

    last_start = len(token_ids) - context_length - 1
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)

    examples: list[ProteinCausalLMExample] = []
    for start in starts:
        end = start + context_length
        examples.append(
            ProteinCausalLMExample(
                input_ids=torch.tensor(token_ids[start:end], dtype=torch.long),
                labels=torch.tensor(token_ids[start + 1 : end + 1], dtype=torch.long),
            )
        )
    return examples

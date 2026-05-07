from __future__ import annotations

import json
from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from libs.core.interfaces import CausalLMBatch, IGNORE_INDEX
from libs.core.mdc import MDCModelConfig, build_default_mdc_layer_types
from libs.data.training.tokenizer import SequenceTokenizer
from libs.data.training.tokenizer.constants import DEFAULT_VOCAB_SIZES
from models.qwen3_5 import QWEN3_5_CONFIG_0_8B, QWEN3_5_CONFIG_2B


PROTEIN_START_TOKEN = "<|protein|>"
PROTEIN_END_TOKEN = "<|endoftext|>"
QWEN3_5_BACKBONE_FAMILY = "qwen3_5"
QWEN3_5_PROTEIN_MODEL_FAMILY = "qwen3_5_protein_lm"
LEGACY_PROTEIN_MODEL_FAMILY = "mdc_protein_lm"
DEFAULT_QWEN3_5_MODEL_NAME = "Qwen/Qwen3.5-0.8B"

QWEN3_5_MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "Qwen/Qwen3.5-0.8B": deepcopy(QWEN3_5_CONFIG_0_8B),
    "Qwen/Qwen3.5-2B": deepcopy(QWEN3_5_CONFIG_2B),
}

QWEN3_5_MODEL_ALIASES: dict[str, str] = {
    "0.8B": "Qwen/Qwen3.5-0.8B",
    "0.8b": "Qwen/Qwen3.5-0.8B",
    "2B": "Qwen/Qwen3.5-2B",
    "2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
    "Qwen3.5-2B": "Qwen/Qwen3.5-2B",
}


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


def build_qwen3_5_config(
    model_name: str = DEFAULT_QWEN3_5_MODEL_NAME,
    *,
    vocab_size: int,
    context_length: int | None = None,
    dtype: torch.dtype | None = None,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    resolved_name = QWEN3_5_MODEL_ALIASES.get(model_name, model_name)
    if resolved_name not in QWEN3_5_MODEL_CONFIGS:
        available = ", ".join(sorted(QWEN3_5_MODEL_CONFIGS))
        raise ValueError(f"Unknown Qwen3.5 config '{model_name}'. Available: {available}")

    resolved_overrides = dict(overrides or {})
    config = deepcopy(QWEN3_5_MODEL_CONFIGS[resolved_name])
    config["vocab_size"] = int(vocab_size)
    if context_length is not None:
        config["context_length"] = int(context_length)
    if dtype is not None:
        config["dtype"] = dtype
    config.update(resolved_overrides)

    n_layers = int(config["n_layers"])
    explicit_layer_types = "layer_types" in resolved_overrides
    config["layer_types"] = _resolve_qwen_layer_types(
        config.get("layer_types"),
        n_layers=n_layers,
        allow_resize=not explicit_layer_types,
    )
    config["model_name"] = resolved_name
    config["config_source"] = "models.qwen3_5.ch05"
    config["backbone_family"] = QWEN3_5_BACKBONE_FAMILY
    return config


def build_mdc_config_from_qwen3_5_config(
    config: Mapping[str, object],
    *,
    dtype: torch.dtype | None = None,
    attention_pattern: str | Sequence[str] = "as_config",
) -> MDCModelConfig:
    emb_dim = int(config["emb_dim"])
    n_heads = int(config["n_heads"])
    n_layers = int(config["n_layers"])
    if emb_dim % n_heads != 0:
        raise ValueError("emb_dim must be divisible by n_heads.")

    if attention_pattern == "as_config":
        layer_types = _resolve_qwen_layer_types(config.get("layer_types"), n_layers=n_layers)
    elif attention_pattern == "qwen_hybrid":
        layer_types = build_default_mdc_layer_types(n_layers)
    elif attention_pattern == "full_attention":
        layer_types = ("full_attention",) * n_layers
    elif isinstance(attention_pattern, Sequence) and not isinstance(attention_pattern, str):
        layer_types = tuple(str(layer_type) for layer_type in attention_pattern)
    else:
        raise ValueError("attention_pattern must be 'as_config', 'qwen_hybrid', 'full_attention', or a sequence.")

    resolved_dtype = dtype if dtype is not None else _coerce_torch_dtype(config.get("dtype", torch.float32))
    head_dim = int(config.get("head_dim", emb_dim // n_heads))
    return MDCModelConfig(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        hidden_dim=int(config.get("hidden_dim", 4 * emb_dim)),
        head_dim=head_dim,
        qk_norm=bool(config.get("qk_norm", True)),
        n_kv_groups=int(config.get("n_kv_groups", n_heads)),
        rope_base=float(config.get("rope_base", 10_000.0)),
        partial_rotary_factor=float(config.get("partial_rotary_factor", 1.0)),
        rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
        linear_conv_kernel_dim=int(config.get("linear_conv_kernel_dim", 4)),
        linear_key_head_dim=int(config.get("linear_key_head_dim", head_dim)),
        linear_value_head_dim=int(config.get("linear_value_head_dim", head_dim)),
        linear_num_key_heads=int(config.get("linear_num_key_heads", n_heads)),
        linear_num_value_heads=int(config.get("linear_num_value_heads", n_heads)),
        dtype=resolved_dtype,
        layer_types=layer_types,
    )


def _resolve_qwen_layer_types(
    layer_types: object,
    *,
    n_layers: int,
    allow_resize: bool = True,
) -> tuple[str, ...]:
    if layer_types is None:
        return build_default_mdc_layer_types(n_layers)

    resolved = tuple(str(layer_type) for layer_type in layer_types)
    if len(resolved) == n_layers:
        return resolved
    if allow_resize:
        return build_default_mdc_layer_types(n_layers)
    raise ValueError("len(layer_types) must equal n_layers.")


def _coerce_torch_dtype(value: object) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value

    normalized = str(value).strip()
    if normalized.startswith("torch."):
        normalized = normalized.removeprefix("torch.")

    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
        "float64": torch.float64,
        "double": torch.float64,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {value!r}")
    return dtype_map[normalized]


def is_supported_protein_checkpoint_family(model_family: object) -> bool:
    return model_family in {
        None,
        LEGACY_PROTEIN_MODEL_FAMILY,
        QWEN3_5_PROTEIN_MODEL_FAMILY,
    }


def extract_protein_backbone_config(checkpoint: Mapping[str, object]) -> Mapping[str, object] | None:
    qwen_config = checkpoint.get("qwen3_5_config")
    if isinstance(qwen_config, Mapping):
        return dict(qwen_config)

    legacy_config = checkpoint.get("llms_from_scratch_config")
    if isinstance(legacy_config, Mapping):
        return dict(legacy_config)

    return None


def save_protein_pretrain_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
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
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
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
    optimizer: torch.optim.Optimizer | None = None,
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
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


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
    stop_at_endoftext: bool = True,
) -> str:
    model.eval()
    token_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    eos_token_id = tokenizer.str_to_int[PROTEIN_END_TOKEN]
    context_size = int(context_length or getattr(model, "cfg", {}).get("context_length", token_ids.size(1)))

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(token_ids[:, -context_size:])
            next_logits = logits[:, -1, :]

            if top_k is not None:
                top_logits, _ = torch.topk(next_logits, top_k)
                min_value = top_logits[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < min_value,
                    torch.tensor(float("-inf"), device=next_logits.device),
                    next_logits,
                )

            if temperature > 0.0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            token_ids = torch.cat([token_ids, next_token], dim=1)
            if stop_at_endoftext and int(next_token.item()) == eos_token_id:
                break

    return tokenizer.decode(token_ids.squeeze(0).tolist())


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

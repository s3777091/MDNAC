from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from libs.core.interfaces import CausalLMBatch, IGNORE_INDEX
from libs.core.mdc.config import MDCModelConfig
from libs.core.pretrain.distributed import (
    create_mdc_distributed_sampler,
    normalize_parallel_state_dict,
    partition_items_for_worker,
    resolve_mdc_distributed_context,
    unwrap_mdc_training_model,
)
from libs.data.config import DataConfig
from libs.data.training.streaming import S3TextPart, downloaded_minio_text_part, list_minio_text_parts
from libs.data.training.tokenizer import SequenceTokenizer
from libs.data.training.tokenizer.constants import DEFAULT_VOCAB_SIZES
from .support.backbone import (
    PROGEN_BACKBONE_FAMILY,
    PROGEN_PROTEIN_MODEL_FAMILY,
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


class ProteinCausalLMStreamingTextDataset(IterableDataset[ProteinCausalLMExample]):
    def __init__(
        self,
        tokenizer: SequenceTokenizer,
        *,
        context_length: int,
        prefix_uri: str | None = None,
        part_uris: Sequence[str] | None = None,
        part_paths: Sequence[Path | str] | None = None,
        s3_client=None,
        config: DataConfig | None = None,
        cache_dir: Path | str | None = None,
        keep_downloaded_parts: bool = False,
        stride: int | None = None,
        shuffle_parts: bool = False,
        shuffle_examples: bool = False,
        shuffle_buffer_size: int = 8192,
        seed: int = 0,
        split: str | None = None,
        train_ratio: float = 0.9,
        split_seed: int = 42,
        distributed: bool | None = None,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        if context_length <= 1:
            raise ValueError("context_length must be greater than 1.")

        self.tokenizer = tokenizer
        self.context_length = int(context_length)
        self.stride = int(stride or context_length)
        if self.stride <= 0:
            raise ValueError("stride must be greater than 0.")

        provided_sources = sum(
            source is not None
            for source in (prefix_uri, part_uris, part_paths)
        )
        if provided_sources != 1:
            raise ValueError("Provide exactly one of prefix_uri, part_uris, or part_paths.")

        if part_paths is not None:
            self.parts = [Path(path) for path in part_paths]
            missing_paths = [path for path in self.parts if not path.exists()]
            if missing_paths:
                raise FileNotFoundError(f"Protein training part was not found: {missing_paths[0]}")
        else:
            self.parts = list(
                list_minio_text_parts(
                    prefix_uri=prefix_uri,
                    part_uris=part_uris,
                    s3_client=s3_client,
                    config=config,
                    suffixes=(".txt",),
                )
            )
        if not self.parts:
            source = prefix_uri or ", ".join(part_uris or ())
            raise FileNotFoundError(f"No protein training parts found in {source!r}.")

        self._s3_client = s3_client
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.keep_downloaded_parts = bool(keep_downloaded_parts)
        self.shuffle_parts = bool(shuffle_parts)
        self.shuffle_examples = bool(shuffle_examples)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        if self.shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be greater than 0.")
        self.seed = int(seed)
        self.split = split
        self.train_ratio = float(train_ratio)
        self.split_seed = int(split_seed)
        self.epoch = 0
        resolved_rank, _, resolved_world_size = resolve_mdc_distributed_context(
            rank=rank,
            world_size=world_size,
        )
        self.use_distributed = bool(distributed) if distributed is not None else resolved_world_size > 1
        self.rank = resolved_rank if self.use_distributed else 0
        self.world_size = resolved_world_size if self.use_distributed else 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        parts, partition_index = _parts_for_current_worker(
            self.parts,
            rank=self.rank,
            world_size=self.world_size,
        )
        rng = random.Random(self.seed + self.epoch * 1000003 + partition_index)
        if self.shuffle_parts:
            rng.shuffle(parts)

        for part in parts:
            if isinstance(part, S3TextPart):
                source = part.uri
                with downloaded_minio_text_part(
                    part,
                    s3_client=self._s3_client,
                    config=self.config,
                    cache_dir=self.cache_dir,
                    keep_downloaded_parts=self.keep_downloaded_parts,
                ) as part_path:
                    examples = _iter_causal_lm_examples_from_text_path(
                        part_path,
                        self.tokenizer,
                        context_length=self.context_length,
                        stride=self.stride,
                        source=source,
                        split=self.split,
                        train_ratio=self.train_ratio,
                        split_seed=self.split_seed,
                    )
                    if self.shuffle_examples:
                        examples = _iter_bounded_shuffled_examples(
                            examples,
                            rng=rng,
                            buffer_size=self.shuffle_buffer_size,
                        )
                    yield from examples
            else:
                source = str(part)
                examples = _iter_causal_lm_examples_from_text_path(
                    part,
                    self.tokenizer,
                    context_length=self.context_length,
                    stride=self.stride,
                    source=source,
                    split=self.split,
                    train_ratio=self.train_ratio,
                    split_seed=self.split_seed,
                )
                if self.shuffle_examples:
                    examples = _iter_bounded_shuffled_examples(
                        examples,
                        rng=rng,
                        buffer_size=self.shuffle_buffer_size,
                    )
                yield from examples


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


def discover_protein_train_text_paths(
    train_text_path: Path | str,
    *,
    part_glob: str = "train_part_*.txt",
    prefer_parts: bool = True,
) -> tuple[Path, ...]:
    resolved_train_path = Path(train_text_path)
    part_paths = tuple(sorted(resolved_train_path.parent.glob(part_glob), key=_natural_path_sort_key))

    if prefer_parts and part_paths:
        return part_paths
    if resolved_train_path.exists():
        return (resolved_train_path,)
    if part_paths:
        return part_paths

    raise FileNotFoundError(
        f"Protein corpus was not found. Expected {resolved_train_path} or parts matching "
        f"{resolved_train_path.parent / part_glob}."
    )


def load_protein_corpus_text_parts(train_text_paths: Sequence[Path | str]) -> str:
    resolved_paths = tuple(Path(path) for path in train_text_paths)
    if not resolved_paths:
        raise ValueError("train_text_paths must not be empty.")

    texts = []
    for path in resolved_paths:
        texts.append(load_protein_corpus_text(path).rstrip("\r\n"))

    corpus_text = "\n".join(texts) + "\n"
    if not corpus_text.strip():
        raise ValueError("Protein corpus parts are empty.")
    return corpus_text


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


def build_or_load_protein_tokenizer_from_text_paths(
    train_text_paths: Sequence[Path | str],
    *,
    tokenizer_map_path: Path | str | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZES["protein"],
    rebuild: bool = False,
) -> ProteinTokenizerArtifact:
    resolved_train_paths = tuple(Path(path) for path in train_text_paths)
    if not resolved_train_paths:
        raise ValueError("train_text_paths must not be empty.")
    resolved_tokenizer_path = (
        Path(tokenizer_map_path)
        if tokenizer_map_path is not None
        else resolved_train_paths[0].with_name("tokenizer_map.json")
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

    text = load_protein_corpus_text_parts(resolved_train_paths)
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
    distributed: bool | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    sampler_seed: int = 0,
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
    sampler = create_mdc_distributed_sampler(
        dataset,
        shuffle=shuffle,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        seed=sampler_seed,
        drop_last=drop_last,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        collate_fn=collator,
    )


def create_streaming_protein_lm_dataloader(
    tokenizer: SequenceTokenizer,
    *,
    context_length: int,
    prefix_uri: str | None = None,
    part_uris: Sequence[str] | None = None,
    part_paths: Sequence[Path | str] | None = None,
    s3_client=None,
    config: DataConfig | None = None,
    cache_dir: Path | str | None = None,
    keep_downloaded_parts: bool = False,
    stride: int | None = None,
    batch_size: int = 4,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_parts: bool = False,
    shuffle_examples: bool = False,
    shuffle_buffer_size: int = 8192,
    seed: int = 0,
    split: str | None = None,
    train_ratio: float = 0.9,
    split_seed: int = 42,
    distributed: bool | None = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> DataLoader[CausalLMBatch]:
    dataset = ProteinCausalLMStreamingTextDataset(
        tokenizer,
        context_length=context_length,
        prefix_uri=prefix_uri,
        part_uris=part_uris,
        part_paths=part_paths,
        s3_client=s3_client,
        config=config,
        cache_dir=cache_dir,
        keep_downloaded_parts=keep_downloaded_parts,
        stride=stride,
        shuffle_parts=shuffle_parts,
        shuffle_examples=shuffle_examples,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        split=split,
        train_ratio=train_ratio,
        split_seed=split_seed,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    collator = ProteinCausalLMBatchCollator(
        pad_token_id=tokenizer.str_to_int["<|pad|>"],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
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
    best_metric_name: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model_config, MDCModelConfig):
        serialized_model_config: Mapping[str, object] = model_config.to_dict()
    else:
        serialized_model_config = dict(model_config)

    checkpoint: dict[str, Any] = {
        "model_family": PROGEN_PROTEIN_MODEL_FAMILY,
        "backbone_family": PROGEN_BACKBONE_FAMILY,
        "model_state_dict": unwrap_mdc_training_model(model).state_dict(),
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
        "best_metric_name": best_metric_name,
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
    unwrap_mdc_training_model(model).load_state_dict(
        normalize_parallel_state_dict(checkpoint["model_state_dict"]),
        strict=strict,
    )
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        _load_optimizer_state_dict(optimizer, checkpoint["optimizer_state_dict"])
    return checkpoint


def load_protein_pretrain_checkpoint_for_profile_tuning(
    path: Path | str,
    *,
    model: torch.nn.Module,
    map_location: torch.device | str = "cpu",
    strict_backbone: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    if not is_supported_protein_checkpoint_family(checkpoint.get("model_family")):
        raise ValueError(
            "Unsupported protein checkpoint family: "
            f"{checkpoint.get('model_family')!r}"
        )

    checkpoint_state = normalize_parallel_state_dict(checkpoint["model_state_dict"])
    target_model = unwrap_mdc_training_model(model)
    target_state = target_model.state_dict()
    adapted_state: dict[str, torch.Tensor] = {}
    skipped_keys: dict[str, str] = {}
    copied_vocab_rows = 0

    for name, source_tensor in checkpoint_state.items():
        target_tensor = target_state.get(name)
        if target_tensor is None:
            skipped_keys[name] = "missing_in_target"
            continue

        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            adapted_state[name] = source_tensor
            continue

        if name in {"tok_emb.weight", "out_head.weight"} and _can_copy_vocab_prefix(
            source_tensor,
            target_tensor,
        ):
            expanded_tensor = target_tensor.detach().clone()
            rows = int(source_tensor.shape[0])
            expanded_tensor[:rows] = source_tensor.to(
                device=expanded_tensor.device,
                dtype=expanded_tensor.dtype,
            )
            adapted_state[name] = expanded_tensor
            copied_vocab_rows = max(copied_vocab_rows, rows)
            continue

        skipped_keys[name] = f"shape_mismatch:{tuple(source_tensor.shape)}->{tuple(target_tensor.shape)}"

    incompatible = target_model.load_state_dict(adapted_state, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    if strict_backbone:
        non_vocab_skipped = {
            key: reason
            for key, reason in skipped_keys.items()
            if key not in {"tok_emb.weight", "out_head.weight"}
        }
        non_vocab_missing = [
            key
            for key in missing_keys
            if key not in {"tok_emb.weight", "out_head.weight"}
        ]
        if non_vocab_skipped or non_vocab_missing or unexpected_keys:
            raise ValueError(
                "Protein checkpoint is not compatible with the target profile-tuning model. "
                f"skipped={non_vocab_skipped}, missing={non_vocab_missing}, unexpected={unexpected_keys}"
            )

    return {
        "checkpoint": checkpoint,
        "copied_vocab_rows": copied_vocab_rows,
        "loaded_keys": sorted(adapted_state),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "skipped_keys": skipped_keys,
    }


def _can_copy_vocab_prefix(source_tensor: torch.Tensor, target_tensor: torch.Tensor) -> bool:
    return (
        source_tensor.ndim == 2
        and target_tensor.ndim == 2
        and int(source_tensor.shape[0]) <= int(target_tensor.shape[0])
        and int(source_tensor.shape[1]) == int(target_tensor.shape[1])
    )


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
    base_model = unwrap_mdc_training_model(model)
    base_model.eval()
    token_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    eos_token_id = tokenizer.str_to_int[PROTEIN_END_TOKEN]
    context_size = int(context_length or getattr(base_model, "cfg", {}).get("context_length", token_ids.size(1)))
    cache = (
        _create_generation_cache(base_model)
        if use_cache and token_ids.size(1) + max_new_tokens <= context_size
        else None
    )

    with torch.no_grad():
        if cache is not None:
            _reset_generation_cache(base_model, cache)
            logits = base_model(token_ids[:, -context_size:], cache=cache)

            for _ in range(max_new_tokens):
                next_token = _select_next_token(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                )
                token_ids = torch.cat([token_ids, next_token], dim=1)
                if stop_at_endoftext and int(next_token.item()) == eos_token_id:
                    break
                logits = base_model(next_token, cache=cache)
        else:
            for _ in range(max_new_tokens):
                logits = base_model(token_ids[:, -context_size:])
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
    return list(
        _iter_causal_lm_examples(
            token_ids,
            context_length=context_length,
            stride=stride,
        )
    )


def _iter_causal_lm_examples(
    token_ids: Sequence[int],
    *,
    context_length: int,
    stride: int,
):
    if len(token_ids) <= context_length + 1:
        yield ProteinCausalLMExample(
            input_ids=torch.tensor(token_ids[:-1], dtype=torch.long),
            labels=torch.tensor(token_ids[1:], dtype=torch.long),
        )
        return

    last_start = len(token_ids) - context_length - 1
    start = 0
    while start <= last_start:
        end = start + context_length
        yield ProteinCausalLMExample(
            input_ids=torch.tensor(token_ids[start:end], dtype=torch.long),
            labels=torch.tensor(token_ids[start + 1 : end + 1], dtype=torch.long),
        )
        start += stride

    final_aligned_start = ((last_start // stride) * stride)
    if final_aligned_start != last_start:
        end = last_start + context_length
        yield ProteinCausalLMExample(
            input_ids=torch.tensor(token_ids[last_start:end], dtype=torch.long),
            labels=torch.tensor(token_ids[last_start + 1 : end + 1], dtype=torch.long),
        )


def _line_belongs_to_split(
    text: str,
    *,
    split: str | None,
    train_ratio: float,
    split_seed: int,
) -> bool:
    if split is None:
        return True

    if split not in {"train", "val"}:
        raise ValueError("split must be one of: None, 'train', 'val'")

    key = f"{split_seed}:{text.strip()}".encode("utf-8")
    digest = hashlib.sha1(key).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)

    if split == "train":
        return bucket < train_ratio
    return bucket >= train_ratio


def _iter_causal_lm_examples_from_text_path(
    path: Path,
    tokenizer: SequenceTokenizer,
    *,
    context_length: int,
    stride: int,
    source: str,
    split: str | None = None,
    train_ratio: float = 0.9,
    split_seed: int = 42,
):
    saw_text = False
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            saw_text = True
            text = line if line.endswith("\n") else line + "\n"
            if not _line_belongs_to_split(
                text,
                split=split,
                train_ratio=train_ratio,
                split_seed=split_seed,
            ):
                continue

            line_source = f"{source}:{line_number}"
            _validate_protein_corpus_text(text, source=line_source)
            token_ids = tokenizer.encode(text)
            if len(token_ids) < 2:
                raise ValueError(f"Protein corpus line must encode to at least 2 tokens: {line_source}")

            yield from _iter_causal_lm_examples(
                token_ids,
                context_length=context_length,
                stride=stride,
            )

    if not saw_text:
        return


def _iter_bounded_shuffled_examples(
    examples,
    *,
    rng: random.Random,
    buffer_size: int,
):
    buffer: list[ProteinCausalLMExample] = []
    for example in examples:
        buffer.append(example)
        if len(buffer) >= buffer_size:
            index = rng.randrange(len(buffer))
            buffer[index], buffer[-1] = buffer[-1], buffer[index]
            yield buffer.pop()

    while buffer:
        index = rng.randrange(len(buffer))
        buffer[index], buffer[-1] = buffer[-1], buffer[index]
        yield buffer.pop()


def _validate_protein_corpus_text(text: str, *, source: str) -> None:
    if PROTEIN_START_TOKEN not in text:
        raise ValueError(f"Protein corpus part must contain {PROTEIN_START_TOKEN}: {source}")
    if PROTEIN_END_TOKEN not in text:
        raise ValueError(f"Protein corpus part must contain {PROTEIN_END_TOKEN}: {source}")


def _parts_for_current_worker(
    parts: Sequence[S3TextPart | Path],
    *,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[list[S3TextPart | Path], int]:
    worker_info = get_worker_info()
    worker_id = 0 if worker_info is None else int(worker_info.id)
    num_workers = 1 if worker_info is None else int(worker_info.num_workers)
    return partition_items_for_worker(
        parts,
        rank=rank,
        world_size=world_size,
        worker_id=worker_id,
        num_workers=num_workers,
    )


def _natural_path_sort_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)

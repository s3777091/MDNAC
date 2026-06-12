"""Small runtime services used by ProteinPretrainTrainer."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import nullcontext
import fnmatch
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import torch
from torch.utils.data import DataLoader

from libs.core.interfaces import CausalLMBatch
from libs.core.mdc.config import MDCModelConfig
from libs.core.pretrain.distributed import MDCTrainingRuntime
from libs.core.pretrain.training import evaluate_mdc_causal_lm_batch_loss
from libs.core.pretrain.protein_lm.core import (
    create_protein_lm_dataloader,
    create_streaming_protein_lm_dataloader,
    load_protein_corpus_text_parts,
    split_protein_corpus_text,
)
from libs.data.config import DataConfig
from libs.data.training.tokenizer import SequenceTokenizer


ConfigSection: TypeAlias = Mapping[str, Any]
PathsConfig: TypeAlias = ConfigSection
DataTrainingConfig: TypeAlias = ConfigSection
ModelTrainingConfig: TypeAlias = ConfigSection
OptimizerTrainingConfig: TypeAlias = ConfigSection
RuntimeTrainingConfig: TypeAlias = ConfigSection
ResumeTrainingConfig: TypeAlias = ConfigSection
MinioTrainingConfig: TypeAlias = ConfigSection
OptimizerBundle: TypeAlias = torch.optim.Optimizer | Sequence[torch.optim.Optimizer]
ProteinBatchLoader: TypeAlias = DataLoader[CausalLMBatch]


@dataclass(slots=True)
class TrainerState:
    global_step: int = 0
    tokens_seen: int = 0
    epoch: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_val_loss: float = math.inf
    best_metric_name: str = "val_loss"

    @property
    def best_loss_or_none(self) -> float | None:
        return None if math.isinf(self.best_val_loss) else self.best_val_loss


@dataclass(slots=True)
class TrainerComponents:
    model: torch.nn.Module | None = None
    optimizer: OptimizerBundle | None = None
    tokenizer: SequenceTokenizer | None = None
    model_config: MDCModelConfig | None = None
    runtime: MDCTrainingRuntime | None = None


@dataclass(slots=True)
class LoaderBundle:
    train_loader: ProteinBatchLoader
    train_eval_loader: ProteinBatchLoader | None
    val_loader: ProteinBatchLoader | None

    def __iter__(self) -> Iterator[ProteinBatchLoader | None]:
        yield self.train_loader
        yield self.train_eval_loader
        yield self.val_loader


@dataclass(slots=True)
class PrecisionContext:
    autocast_dtype: torch.dtype | None
    use_autocast: bool
    grad_scaler: torch.amp.GradScaler | None

    def autocast(self):
        if self.use_autocast:
            return torch.amp.autocast("cuda", dtype=self.autocast_dtype)
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            return
        loss.backward()

    def unscale_(self, optimizers: Sequence[torch.optim.Optimizer]) -> None:
        if self.grad_scaler is None:
            return
        for optimizer in optimizers:
            self.grad_scaler.unscale_(optimizer)

    def step(self, optimizers: Sequence[torch.optim.Optimizer]) -> None:
        if self.grad_scaler is not None:
            for optimizer in optimizers:
                self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
            return
        for optimizer in optimizers:
            optimizer.step()


@dataclass(frozen=True, slots=True)
class TrainingLoopSettings:
    num_epochs: int
    max_steps: int | None
    eval_freq: int
    eval_batches: int
    grad_clip_norm: float | None
    save_every_steps: int | None
    save_best: bool
    save_last: bool
    save_final: bool
    gradient_accumulation_steps: int
    log_every_steps: int

    @classmethod
    def from_config(cls, training_cfg: ConfigSection) -> "TrainingLoopSettings":
        eval_freq = int(training_cfg["eval_freq"])
        save_last = bool(training_cfg.get("save_last", False))
        if save_last:
            raise ValueError("training.save_last must be false; use checkpoint_best.pt as the model artifact")
        save_best = bool(training_cfg.get("save_best", True))
        if not save_best:
            raise ValueError("training.save_best must be true so checkpoint_best.pt is available")
        return cls(
            num_epochs=int(training_cfg["num_epochs"]),
            max_steps=training_cfg.get("max_steps"),
            eval_freq=eval_freq,
            eval_batches=int(training_cfg["eval_batches"]),
            grad_clip_norm=training_cfg["grad_clip_norm"],
            save_every_steps=training_cfg.get("save_every_steps"),
            save_best=save_best,
            save_last=save_last,
            save_final=bool(training_cfg.get("save_final", True)),
            gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
            log_every_steps=max(1, eval_freq // 2) if eval_freq > 0 else 50,
        )


@dataclass(slots=True)
class GradientAccumulator:
    steps: int
    micro_step: int = 0

    def next_micro_step(self) -> int:
        self.micro_step += 1
        return self.micro_step

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss / self.steps

    @property
    def at_boundary(self) -> bool:
        return self.micro_step % self.steps == 0

    @property
    def has_leftover(self) -> bool:
        return self.micro_step % self.steps != 0


def resolve_precision_context(mixed_precision: str, device: torch.device) -> PrecisionContext:
    autocast_dtype = _resolve_autocast_dtype(mixed_precision, device)
    use_autocast = autocast_dtype is not None and device.type == "cuda"
    use_grad_scaler = use_autocast and autocast_dtype == torch.float16
    grad_scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None
    return PrecisionContext(
        autocast_dtype=autocast_dtype,
        use_autocast=use_autocast,
        grad_scaler=grad_scaler,
    )


def _resolve_autocast_dtype(mixed_precision: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return None


def optimizer_list(optimizer: OptimizerBundle) -> list[torch.optim.Optimizer]:
    if isinstance(optimizer, torch.optim.Optimizer):
        return [optimizer]
    optimizers = list(optimizer)
    if not optimizers:
        raise ValueError("optimizer sequence must not be empty.")
    return optimizers


def zero_grad(optimizers: Sequence[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def step_optimizers(
    *,
    model: torch.nn.Module,
    optimizers: Sequence[torch.optim.Optimizer],
    precision: PrecisionContext,
    grad_clip_norm: float | None,
) -> None:
    if grad_clip_norm is not None:
        precision.unscale_(optimizers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    precision.step(optimizers)
    zero_grad(optimizers)


class DataLoaderFactory:
    """Builds protein LM loaders without owning trainer state."""

    def __init__(
        self,
        *,
        tokenizer: SequenceTokenizer,
        model_config: MDCModelConfig,
        runtime: MDCTrainingRuntime,
        paths: PathsConfig,
        data_cfg: DataTrainingConfig,
        model_cfg: ModelTrainingConfig,
        minio_cfg: MinioTrainingConfig,
        minio_data_config: DataConfig | None,
        local_paths_provider: Callable[[], tuple[Path, ...]],
        is_main_process: bool,
        merged_train_path: Path | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._model_config = model_config
        self._runtime = runtime
        self._paths = paths
        self._data_cfg = data_cfg
        self._model_cfg = model_cfg
        self._minio_cfg = minio_cfg
        self._minio_data_config = minio_data_config
        self._local_paths_provider = local_paths_provider
        self._is_main_process = is_main_process
        self._merged_train_path = merged_train_path

    def build(self) -> LoaderBundle:
        local_paths = self._local_paths_provider()
        if self._use_minio:
            return self.build_minio_streaming_loaders()
        if self._merged_train_path is not None:
            # Single merged corpus -> stream it with LINE-level sharding so every
            # worker stays busy (file-level sharding would idle all but one worker).
            return self.build_local_streaming_loaders(
                (self._merged_train_path,), shard_by="line"
            )
        if self._use_local_streaming(local_paths):
            return self.build_local_streaming_loaders(local_paths)
        return self.build_in_memory_loaders(local_paths)

    def build_minio_streaming_loaders(self) -> LoaderBundle:
        train_loader = self._streaming_loader(
            split="train",
            shuffle_parts=bool(self._data_cfg["shuffle_parts"]),
            shuffle_examples=bool(self._data_cfg["shuffle_examples"]),
            shuffle_buffer_size=int(self._data_cfg["shuffle_buffer_size"]),
            seed=self._runtime.rank,
            distributed=self._runtime.distributed,
            rank=self._runtime.rank,
            world_size=self._runtime.world_size,
            include_minio=True,
        )
        train_eval_loader = (
            self._eval_streaming_loader(split="train", include_minio=True)
            if self._is_main_process
            else None
        )
        val_loader = (
            self._eval_streaming_loader(split="val", include_minio=True)
            if self._is_main_process
            else None
        )
        return LoaderBundle(train_loader, train_eval_loader, val_loader)

    def build_local_streaming_loaders(
        self, local_paths: tuple[Path, ...], *, shard_by: str = "file"
    ) -> LoaderBundle:
        train_loader = self._streaming_loader(
            split="train",
            part_paths=local_paths,
            shuffle_parts=bool(self._data_cfg["shuffle_parts"]),
            shuffle_examples=bool(self._data_cfg["shuffle_examples"]),
            shuffle_buffer_size=int(self._data_cfg["shuffle_buffer_size"]),
            seed=self._runtime.rank,
            distributed=self._runtime.distributed,
            rank=self._runtime.rank,
            world_size=self._runtime.world_size,
            shard_by=shard_by,
        )
        train_eval_loader = (
            self._eval_streaming_loader(split="train", part_paths=local_paths, shard_by=shard_by)
            if self._is_main_process
            else None
        )
        val_loader = (
            self._eval_streaming_loader(split="val", part_paths=local_paths, shard_by=shard_by)
            if self._is_main_process
            else None
        )
        return LoaderBundle(train_loader, train_eval_loader, val_loader)

    def build_in_memory_loaders(self, local_paths: tuple[Path, ...]) -> LoaderBundle:
        corpus_text = load_protein_corpus_text_parts(local_paths) if local_paths else ""
        if not corpus_text:
            raise ValueError("No local corpus or MinIO parts configured.")
        train_text, val_text = split_protein_corpus_text(
            corpus_text,
            train_ratio=self._data_cfg["train_ratio"],
        )
        loader_kwargs = self._loader_kwargs
        train_loader = create_protein_lm_dataloader(
            train_text,
            self._tokenizer,
            shuffle=True,
            sampler_seed=0,
            distributed=self._runtime.distributed,
            rank=self._runtime.rank,
            world_size=self._runtime.world_size,
            **loader_kwargs,
        )
        train_eval_loader = (
            create_protein_lm_dataloader(
                train_text,
                self._tokenizer,
                shuffle=False,
                distributed=False,
                **loader_kwargs,
            )
            if self._is_main_process
            else None
        )
        val_loader = (
            create_protein_lm_dataloader(
                val_text,
                self._tokenizer,
                shuffle=False,
                distributed=False,
                **loader_kwargs,
            )
            if val_text and self._is_main_process
            else None
        )
        return LoaderBundle(train_loader, train_eval_loader, val_loader)

    @property
    def _loader_kwargs(self) -> dict[str, Any]:
        context_length = int(self._model_config.context_length)
        return {
            "context_length": context_length,
            "stride": self._model_cfg["stride"] or max(1, context_length // 2),
            "batch_size": self._data_cfg["batch_size"],
            "num_workers": self._data_cfg["num_workers"],
            "pin_memory": self._data_cfg["pin_memory"],
        }

    @property
    def _use_minio(self) -> bool:
        return bool(self._minio_cfg["train_parts_prefix_uri"] or self._minio_cfg["train_part_uris"])

    def _use_local_streaming(self, local_paths: tuple[Path, ...]) -> bool:
        if not self._data_cfg["stream_local_train_parts"]:
            return False
        part_glob = str(self._data_cfg.get("train_part_glob") or "train_part_*.txt")
        return any(fnmatch.fnmatch(path.name, part_glob) for path in local_paths)

    def _eval_streaming_loader(
        self,
        *,
        split: str,
        include_minio: bool = False,
        part_paths: tuple[Path, ...] | None = None,
        shard_by: str = "file",
    ) -> ProteinBatchLoader:
        return self._streaming_loader(
            split=split,
            part_paths=part_paths,
            shuffle_parts=False,
            shuffle_examples=False,
            seed=0,
            distributed=False,
            include_minio=include_minio,
            shard_by=shard_by,
        )

    def _streaming_loader(
        self,
        *,
        split: str,
        shuffle_parts: bool,
        shuffle_examples: bool,
        seed: int,
        distributed: bool,
        include_minio: bool = False,
        part_paths: tuple[Path, ...] | None = None,
        shuffle_buffer_size: int | None = None,
        rank: int | None = None,
        world_size: int | None = None,
        shard_by: str = "file",
    ) -> ProteinBatchLoader:
        kwargs: dict[str, Any] = {
            "shuffle_parts": shuffle_parts,
            "shuffle_examples": shuffle_examples,
            "seed": seed,
            "distributed": distributed,
            "split": split,
            "train_ratio": float(self._data_cfg["train_ratio"]),
            "split_seed": int(self._data_cfg.get("split_seed", 42)),
            "shard_by": shard_by,
            **self._loader_kwargs,
        }
        if shuffle_buffer_size is not None:
            kwargs["shuffle_buffer_size"] = shuffle_buffer_size
        if rank is not None:
            kwargs["rank"] = rank
        if world_size is not None:
            kwargs["world_size"] = world_size
        if include_minio:
            kwargs.update(
                {
                    "prefix_uri": self._minio_cfg["train_parts_prefix_uri"] or None,
                    "part_uris": self._minio_cfg["train_part_uris"] or None,
                    "config": self._minio_data_config,
                    "cache_dir": self._paths["train_part_cache_dir"],
                    "keep_downloaded_parts": self._data_cfg["keep_downloaded_train_parts"],
                }
            )
        else:
            kwargs["part_paths"] = part_paths
        return create_streaming_protein_lm_dataloader(self._tokenizer, **kwargs)


class CheckpointService:
    """Handles saving and loading training checkpoints."""

    def __init__(
        self,
        paths: PathsConfig,
        resume_cfg: ResumeTrainingConfig,
        data_cfg: DataTrainingConfig,
        model_cfg: ModelTrainingConfig,
        optimizer_cfg: OptimizerTrainingConfig,
        runtime_cfg: RuntimeTrainingConfig,
        minio_cfg: MinioTrainingConfig,
    ) -> None:
        self._paths = paths
        self._resume_cfg = resume_cfg
        self._data_cfg = data_cfg
        self._model_cfg = model_cfg
        self._optimizer_cfg = optimizer_cfg
        self._runtime_cfg = runtime_cfg
        self._minio_cfg = minio_cfg

    def save_checkpoint(
        self,
        path: Path,
        model: torch.nn.Module,
        optimizer: OptimizerBundle | None,
        model_config: MDCModelConfig,
        tokenizer: SequenceTokenizer,
        epoch: int,
        global_step: int,
        tokens_seen: int,
        train_losses: list[float],
        val_losses: list[float],
        best_val_loss: float | None,
        best_metric_name: str | None,
        local_paths: tuple[Path, ...],
    ) -> Path:
        from libs.core.pretrain.protein_lm.core import save_protein_pretrain_checkpoint
        from libs.core.pretrain.training_config import describe_protein_training_optimizers

        return save_protein_pretrain_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            tokenizer=tokenizer,
            tokenizer_map_path=self._paths["tokenizer_map_path"],
            epoch=epoch,
            global_step=global_step,
            tokens_seen=tokens_seen,
            train_losses=train_losses,
            val_losses=val_losses,
            best_val_loss=best_val_loss,
            best_metric_name=best_metric_name,
            training_args={
                "batch_size": self._data_cfg["batch_size"],
                "context_length": int(model_config.context_length),
                "stride": self._model_cfg["stride"],
                "learning_rate": self._optimizer_cfg["learning_rate"],
                "weight_decay": self._optimizer_cfg["weight_decay"],
                "optimizer_type": self._optimizer_cfg["type"],
                "optimizer_types": describe_protein_training_optimizers(optimizer),
                "multi_gpu_mode": self._runtime_cfg["multi_gpu_mode"],
                "num_workers": self._data_cfg["num_workers"],
                "pin_memory": self._data_cfg["pin_memory"],
                "train_config_path": str(self._paths.get("config_path", "")),
            },
            extra={
                "corpus_files": [str(p.resolve()) for p in local_paths],
                "minio_train_parts_prefix_uri": self._minio_cfg["train_parts_prefix_uri"],
                "minio_train_part_uris": list(self._minio_cfg["train_part_uris"]),
            },
        )

    def save_last(self, **kwargs) -> Path:
        return self.save_checkpoint(self._resume_cfg["output_checkpoint_path"], **kwargs)

    def save_best(self, **kwargs) -> Path:
        return self.save_checkpoint(self._resume_cfg["best_checkpoint_path"], **kwargs)

    def save_final(self, **kwargs) -> Path:
        return self.save_checkpoint(self._resume_cfg["final_checkpoint_path"], **kwargs)


class MetricsWriter:
    """Handles metrics persistence to JSONL file."""

    def __init__(self, metrics_path: Path | str | None) -> None:
        self._metrics_path = Path(metrics_path) if metrics_path else None

    def append(self, epoch: int, global_step: int, tokens_seen: int, train_loss: float, val_loss: float) -> None:
        if self._metrics_path is None:
            return
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "train_loss": _json_loss(train_loss),
            "val_loss": _json_loss(val_loss),
        }
        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _json_loss(value: float) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if resolved != resolved or resolved in {float("inf"), float("-inf")}:
        return None
    return resolved


class Evaluator:
    """Handles model evaluation during training."""

    def evaluate(
        self,
        model: torch.nn.Module,
        train_eval_loader: ProteinBatchLoader | None,
        val_loader: ProteinBatchLoader | None,
        device: torch.device,
        max_batches: int,
        autocast_dtype: torch.dtype | None = None,
    ) -> tuple[float, float]:
        """Returns (train_eval_loss, val_loss). NaN if loader is None."""
        train_eval_loss = (
            evaluate_mdc_causal_lm_batch_loss(
                model, train_eval_loader, device=device, max_batches=max_batches, autocast_dtype=autocast_dtype,
            )
            if train_eval_loader is not None
            else float("nan")
        )
        val_loss = (
            evaluate_mdc_causal_lm_batch_loss(
                model, val_loader, device=device, max_batches=max_batches, autocast_dtype=autocast_dtype,
            )
            if val_loader is not None
            else float("nan")
        )
        return train_eval_loss, val_loss

"""Extracted services for ProteinPretrainTrainer to reduce class size.

These services handle:
- Checkpoint save/load operations
- Metrics persistence
- DataLoader construction strategy selection
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from libs.core.interfaces import CausalLMBatch
from libs.core.pretrain.training import evaluate_mdc_causal_lm_batch_loss


class CheckpointService:
    """Handles saving and loading training checkpoints."""

    def __init__(
        self,
        paths: dict[str, Any],
        resume_cfg: dict[str, Any],
        data_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        optimizer_cfg: dict[str, Any],
        runtime_cfg: dict[str, Any],
        minio_cfg: dict[str, Any],
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
        optimizer: Any,
        model_config: Any,
        tokenizer: Any,
        epoch: int,
        global_step: int,
        tokens_seen: int,
        train_losses: list[float],
        val_losses: list[float],
        best_val_loss: float | None,
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
            training_args={
                "batch_size": self._data_cfg["batch_size"],
                "context_length": int(model_config.context_length),
                "stride": self._model_cfg["stride"],
                "learning_rate": self._optimizer_cfg["learning_rate"],
                "muon_learning_rate": self._optimizer_cfg.get("muon_learning_rate"),
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
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class Evaluator:
    """Handles model evaluation during training."""

    def evaluate(
        self,
        model: torch.nn.Module,
        train_eval_loader: Any | None,
        val_loader: Any | None,
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

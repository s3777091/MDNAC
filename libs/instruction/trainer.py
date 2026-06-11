from __future__ import annotations

import json
import math
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from libs.core.app import MicrobialDecoderCoreApp
from libs.core.interfaces import CausalLMBatch, IGNORE_INDEX
from libs.core.mdc.config import MDCModelConfig
from libs.core.pretrain.distributed import (
    MDCTrainingRuntime,
    cleanup_mdc_distributed_training,
    prepare_mdc_training_runtime,
    set_mdc_data_loader_epoch,
    unwrap_mdc_training_model,
)
from libs.core.pretrain.profiled import (
    MDCProfileSequencePretrainArtifacts,
    save_mdc_profile_sequence_pretrain_artifacts,
)
from libs.core.pretrain.protein_lm.core import (
    _natural_path_sort_key,
    load_protein_pretrain_checkpoint_for_profile_tuning,
)
from libs.core.pretrain.lr_schedule import (
    LRScheduleConfig,
    build_warmup_cosine_scheduler,
)
from libs.core.pretrain.training import compute_mdc_causal_lm_loss
from libs.core.pretrain.training_config import (
    _as_project_path,
    _bool_value,
    _int_sequence_or_none,
    _nested_get,
    _normalize_device,
    _normalize_optimizer_type,
    _optional_float,
    _optional_int,
    _resolve_auto_bool,
    _resolve_mixed_precision,
    create_protein_training_optimizer,
    describe_protein_training_optimizers,
)
from libs.data.training.profile_tokenizer import DEFAULT_PROFILE_BASE_CHARSET

from .data import (
    count_instruction_split_records_by_split,
    create_instruction_dataloader,
)
from .schema import audit_instruction_jsonl, iter_instruction_records, resolve_instruction_paths


INSTRUCTION_CHECKPOINT_FAMILY = "mdc_profile_sequence_instruction"


@dataclass(frozen=True)
class InstructionTrainingConfig:
    instruction_jsonl: str | Path | Sequence[str | Path]
    base_checkpoint_path: str | Path
    output_dir: str | Path
    artifact_dir: str | Path | None = None
    artifact_source_jsonl: str | Path | None = None
    sequence_tokenizer_map_path: str | Path | None = None
    auto_detect_sequence_tokenizer_map: bool = True
    rebuild_artifacts: bool = False
    artifact_profile_sample_size: int = 100_000
    default_sequence_type: str = "protein"
    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"
    prompt_format: str = "alpaca"
    audit_before_training: bool = True
    reuse_existing_audit: bool = False
    profile_vocab_size: int = 256
    kmer_size: int = 3
    train_ratio: float = 0.95
    split_seed: int = 42
    count_splits_before_training: bool | str = "auto"
    count_progress_every: int | None = 250_000
    batch_size: int = 2
    num_workers: int = 0
    pin_memory: bool = True
    shuffle_files: bool = True
    shuffle_records: bool = True
    shuffle_buffer_size: int = 2048
    num_epochs: int = 1
    max_steps: int | None = None
    gradient_accumulation_steps: int = 8
    eval_freq: int = 100
    eval_batches: int = 16
    log_every_steps: int | None = None
    save_every_steps: int | None = 100
    save_last: bool = False
    save_best: bool = True
    save_final: bool = True
    grad_clip_norm: float | None = 1.0
    optimizer_type: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    fused: bool | str = "auto"
    lr_scheduler: str = "none"
    warmup_steps: int = 0
    warmup_ratio: float | None = None
    min_lr_ratio: float = 0.1
    lr_decay_steps: int | None = None
    device: str = "auto"
    multi_gpu_mode: str = "auto"
    ddp_find_unused_parameters: bool = False
    data_parallel_device_ids: Sequence[int] | None = None
    mixed_precision: str = "auto"
    resume_if_available: bool = True
    restore_optimizer_state: bool = True
    seed: int = 123
    train_on_prompt: bool = False
    include_separator_in_loss: bool = False
    include_eos_in_loss: bool = False
    strict_backbone: bool = True


def discover_instruction_jsonl_training_paths(
    project_root: str | Path,
    config_mapping: Mapping[str, Any],
) -> tuple[Path, ...]:
    resolved_project_root = Path(project_root).expanduser().resolve()
    instruction_jsonl_path = _as_project_path(
        _config_value(config_mapping, ("paths", "instruction_jsonl_path"), ("paths", "instruction_jsonl"))
        or Path("data/compiled/refseq_bacteria_protein/instruction.jsonl"),
        project_root=resolved_project_root,
    )
    instruction_part_dir_value = _config_value(config_mapping, ("paths", "instruction_part_dir"))
    instruction_part_dir = (
        _as_project_path(instruction_part_dir_value, project_root=resolved_project_root)
        if instruction_part_dir_value is not None
        else instruction_jsonl_path.parent
    )
    instruction_part_glob = str(
        _config_value(config_mapping, ("data", "instruction_part_glob"), ("paths", "instruction_part_glob"))
        or "instruction_part_*.jsonl"
    )
    prefer_instruction_parts = _bool_value(
        _config_value(config_mapping, ("data", "prefer_instruction_parts")),
        True,
    )
    part_paths = (
        tuple(sorted(instruction_part_dir.glob(instruction_part_glob), key=_natural_path_sort_key))
        if instruction_part_dir.exists()
        else ()
    )
    if prefer_instruction_parts and part_paths:
        return part_paths
    if instruction_jsonl_path.exists():
        return (instruction_jsonl_path,)
    if part_paths:
        return part_paths
    return (instruction_jsonl_path,)


def build_instruction_training_config(
    project_root: str | Path,
    config_mapping: Mapping[str, Any],
) -> InstructionTrainingConfig:
    if not isinstance(config_mapping, Mapping):
        raise TypeError("config_mapping must be a mapping loaded by the shared training config loader.")

    resolved_project_root = Path(project_root).expanduser().resolve()
    instruction_paths = discover_instruction_jsonl_training_paths(resolved_project_root, config_mapping)
    instruction_jsonl_path = _as_project_path(
        _config_value(config_mapping, ("paths", "instruction_jsonl_path"), ("paths", "instruction_jsonl"))
        or Path("data/compiled/refseq_bacteria_protein/instruction.jsonl"),
        project_root=resolved_project_root,
    )
    base_checkpoint_path = _as_project_path(
        _config_value(config_mapping, ("paths", "base_checkpoint_path"))
        or Path("data/checkpoints/protein_from_scratch/checkpoint_best.pt"),
        project_root=resolved_project_root,
    )
    if base_checkpoint_path.name != "checkpoint_best.pt":
        raise ValueError("paths.base_checkpoint_path must point to checkpoint_best.pt")

    output_dir = _as_project_path(
        _config_value(config_mapping, ("paths", "checkpoint_dir"), ("paths", "output_dir"))
        or Path("data/checkpoints/protein_instruction"),
        project_root=resolved_project_root,
    )
    artifact_dir = _as_project_path(
        _config_value(config_mapping, ("paths", "artifact_dir"))
        or Path("data/compiled/refseq_bacteria_instruction_profile"),
        project_root=resolved_project_root,
    )
    artifact_source_jsonl = _as_project_path(
        _config_value(config_mapping, ("paths", "artifact_source_jsonl")) or instruction_jsonl_path,
        project_root=resolved_project_root,
    )
    sequence_tokenizer_map_path = _resolve_optional_project_path(
        _config_value(config_mapping, ("paths", "sequence_tokenizer_map_path"), ("paths", "tokenizer_map_path")),
        project_root=resolved_project_root,
    )
    if sequence_tokenizer_map_path is None:
        default_sequence_tokenizer_map_path = instruction_jsonl_path.with_name("tokenizer_map.json")
        if default_sequence_tokenizer_map_path.exists():
            sequence_tokenizer_map_path = default_sequence_tokenizer_map_path

    rebuild_artifacts_value = _config_value(
        config_mapping,
        ("artifacts", "rebuild_artifacts"),
        ("artifacts", "rebuild"),
    )
    count_splits_before_training_value = _config_value(
        config_mapping,
        ("data", "count_splits_before_training"),
        ("training", "count_splits_before_training"),
    )
    return InstructionTrainingConfig(
        instruction_jsonl=instruction_paths,
        base_checkpoint_path=base_checkpoint_path,
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        artifact_source_jsonl=artifact_source_jsonl,
        sequence_tokenizer_map_path=sequence_tokenizer_map_path,
        auto_detect_sequence_tokenizer_map=_bool_value(
            _config_value(config_mapping, ("artifacts", "auto_detect_sequence_tokenizer_map")),
            True,
        ),
        rebuild_artifacts=_bool_value(rebuild_artifacts_value, False),
        artifact_profile_sample_size=int(
            _config_value(config_mapping, ("artifacts", "profile_sample_size")) or 100_000
        ),
        default_sequence_type=str(
            _config_value(config_mapping, ("schema", "default_sequence_type")) or "protein"
        ),
        instruction_field=str(_config_value(config_mapping, ("schema", "instruction_field")) or "instruction"),
        input_field=str(_config_value(config_mapping, ("schema", "input_field")) or "input"),
        output_field=str(_config_value(config_mapping, ("schema", "output_field")) or "output"),
        prompt_format=str(_config_value(config_mapping, ("schema", "prompt_format")) or "alpaca"),
        audit_before_training=_bool_value(
            _config_value(config_mapping, ("data", "audit_before_training"), ("training", "audit_before_training")),
            True,
        ),
        reuse_existing_audit=_bool_value(
            _config_value(config_mapping, ("data", "reuse_existing_audit"), ("training", "reuse_existing_audit")),
            False,
        ),
        profile_vocab_size=int(_config_value(config_mapping, ("artifacts", "profile_vocab_size")) or 256),
        kmer_size=int(_config_value(config_mapping, ("artifacts", "kmer_size")) or 3),
        train_ratio=float(_config_value(config_mapping, ("data", "train_ratio")) or 0.95),
        split_seed=int(_config_value(config_mapping, ("data", "split_seed")) or 42),
        count_splits_before_training=(
            "auto" if count_splits_before_training_value is None else count_splits_before_training_value
        ),
        count_progress_every=_optional_int(
            _config_value(config_mapping, ("data", "count_progress_every"), ("training", "count_progress_every")),
            default=250_000,
        ),
        batch_size=int(_config_value(config_mapping, ("data", "batch_size")) or 2),
        num_workers=int(_config_value(config_mapping, ("data", "num_workers")) or 0),
        pin_memory=_resolve_auto_bool(
            _config_value(config_mapping, ("data", "pin_memory")),
            default=torch.cuda.is_available(),
        ),
        shuffle_files=_bool_value(
            _config_value(config_mapping, ("data", "shuffle_files"), ("data", "shuffle_parts")),
            True,
        ),
        shuffle_records=_bool_value(
            _config_value(config_mapping, ("data", "shuffle_records"), ("data", "shuffle_examples")),
            True,
        ),
        shuffle_buffer_size=int(_config_value(config_mapping, ("data", "shuffle_buffer_size")) or 4096),
        num_epochs=int(_config_value(config_mapping, ("training", "num_epochs")) or 1),
        max_steps=_optional_int(_config_value(config_mapping, ("training", "max_steps"))),
        gradient_accumulation_steps=int(
            _config_value(config_mapping, ("training", "gradient_accumulation_steps")) or 8
        ),
        eval_freq=int(_config_value(config_mapping, ("training", "eval_freq")) or 100),
        eval_batches=int(_config_value(config_mapping, ("training", "eval_batches")) or 16),
        log_every_steps=_optional_int(_config_value(config_mapping, ("training", "log_every_steps"))),
        save_every_steps=_optional_int(
            _config_value(config_mapping, ("training", "save_every_steps")),
            default=100,
        ),
        save_last=_bool_value(_config_value(config_mapping, ("training", "save_last")), False),
        save_best=_bool_value(_config_value(config_mapping, ("training", "save_best")), True),
        save_final=_bool_value(_config_value(config_mapping, ("training", "save_final")), True),
        grad_clip_norm=_optional_float(
            _config_value(config_mapping, ("training", "grad_clip_norm")),
            default=1.0,
        ),
        optimizer_type=_normalize_optimizer_type(
            _config_value(config_mapping, ("optimizer", "type")) or "adamw"
        ),
        learning_rate=float(_config_value(config_mapping, ("optimizer", "learning_rate")) or 2e-4),
        weight_decay=float(_config_value(config_mapping, ("optimizer", "weight_decay")) or 0.1),
        fused=_resolve_auto_bool(
            _config_value(config_mapping, ("optimizer", "fused")),
            default=True,
        ),
        lr_scheduler=_normalize_instruction_lr_scheduler(
            _config_value(config_mapping, ("optimizer", "lr_scheduler"))
        ),
        warmup_steps=int(_config_value(config_mapping, ("optimizer", "warmup_steps")) or 0),
        warmup_ratio=_optional_float(_config_value(config_mapping, ("optimizer", "warmup_ratio"))),
        min_lr_ratio=float(
            _optional_float(_config_value(config_mapping, ("optimizer", "min_lr_ratio")), default=0.1)
        ),
        lr_decay_steps=_optional_int(_config_value(config_mapping, ("optimizer", "lr_decay_steps"))),
        device=_normalize_device(_config_value(config_mapping, ("runtime", "device")) or "auto"),
        multi_gpu_mode=str(_config_value(config_mapping, ("runtime", "multi_gpu_mode")) or "auto"),
        ddp_find_unused_parameters=_bool_value(
            _config_value(config_mapping, ("runtime", "ddp_find_unused_parameters")),
            False,
        ),
        data_parallel_device_ids=_int_sequence_or_none(
            _config_value(config_mapping, ("runtime", "data_parallel_device_ids"))
        ),
        mixed_precision=_resolve_mixed_precision(_config_value(config_mapping, ("runtime", "mixed_precision"))),
        resume_if_available=_bool_value(
            _config_value(config_mapping, ("resume", "resume_if_available"), ("mode", "resume_if_available")),
            True,
        ),
        restore_optimizer_state=_bool_value(
            _config_value(config_mapping, ("resume", "restore_optimizer_state")),
            True,
        ),
        seed=int(_config_value(config_mapping, ("runtime", "seed"), ("training", "seed")) or 123),
        train_on_prompt=_bool_value(_config_value(config_mapping, ("training", "train_on_prompt")), False),
        include_separator_in_loss=_bool_value(
            _config_value(config_mapping, ("training", "include_separator_in_loss")),
            False,
        ),
        include_eos_in_loss=_bool_value(
            _config_value(config_mapping, ("training", "include_eos_in_loss")),
            False,
        ),
        strict_backbone=_bool_value(_config_value(config_mapping, ("resume", "strict_backbone")), True),
    )


def _normalize_instruction_lr_scheduler(value: Any) -> str:
    if value is None:
        return "none"
    normalized = str(value).strip().lower()
    if normalized not in {"none", "cosine"}:
        raise ValueError("optimizer.lr_scheduler must be one of: none, cosine")
    return normalized


def _config_value(config_mapping: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = _nested_get(config_mapping, *path)
        if value is not None:
            return value
    return None


def _resolve_optional_project_path(value: Any, *, project_root: Path) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _as_project_path(value, project_root=project_root)


def _sample_instruction_records_for_artifacts(
    paths: Path | Sequence[Path],
    *,
    default_sequence_type: str,
    instruction_field: str,
    input_field: str,
    output_field: str,
    prompt_format: str,
    sample_size: int,
):
    if sample_size <= 0:
        raise ValueError("artifact_profile_sample_size must be positive.")
    records = []
    for index, record in enumerate(
        iter_instruction_records(
            paths,
            default_sequence_type=default_sequence_type,
            instruction_field=instruction_field,
            input_field=input_field,
            output_field=output_field,
            prompt_format=prompt_format,
        )
    ):
        if index >= sample_size:
            break
        records.append(record)
    if not records:
        raise ValueError(f"Instruction JSONL does not contain any valid records: {paths}")
    return tuple(records)


@dataclass(frozen=True)
class InstructionTrainingResult:
    output_dir: Path
    checkpoint_last_path: Path | None
    checkpoint_best_path: Path | None
    checkpoint_final_path: Path | None
    metrics_history_path: Path
    training_summary_path: Path
    loss_plot_path: Path
    global_step: int
    tokens_seen: int
    epochs_completed: int
    best_val_loss: float | None
    final_train_loss: float | None
    final_val_loss: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "checkpoint_last_path": None if self.checkpoint_last_path is None else str(self.checkpoint_last_path),
            "checkpoint_best_path": None if self.checkpoint_best_path is None else str(self.checkpoint_best_path),
            "checkpoint_final_path": None if self.checkpoint_final_path is None else str(self.checkpoint_final_path),
            "metrics_history_path": str(self.metrics_history_path),
            "training_summary_path": str(self.training_summary_path),
            "loss_plot_path": str(self.loss_plot_path),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "epochs_completed": self.epochs_completed,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
        }


class InstructionTrainer:
    def __init__(self, config: InstructionTrainingConfig) -> None:
        self.config = config
        if config.save_last:
            raise ValueError("save_last must be false; use checkpoint_best.pt as the model artifact")
        if not config.save_best:
            raise ValueError("save_best must be true so checkpoint_best.pt is available")
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.artifact_dir = (
            self.output_dir / "profile_sequence_artifacts"
            if config.artifact_dir is None
            else Path(config.artifact_dir).expanduser().resolve()
        )
        self.base_checkpoint_path = Path(config.base_checkpoint_path).expanduser().resolve()
        if self.base_checkpoint_path.name != "checkpoint_best.pt":
            raise ValueError("base_checkpoint_path must point to checkpoint_best.pt")
        self.instruction_paths = resolve_instruction_paths(config.instruction_jsonl)
        self.checkpoint_last_path = self.output_dir / "checkpoint_last.pt"
        self.checkpoint_best_path = self.output_dir / "checkpoint_best.pt"
        self.checkpoint_final_path = self.output_dir / "checkpoint_final.pt"
        self.metrics_history_path = self.output_dir / "metrics_history.jsonl"
        self.training_summary_path = self.output_dir / "training_summary.json"
        self.loss_plot_path = self.output_dir / "loss_curve.png"

        self.artifacts: MDCProfileSequencePretrainArtifacts | None = None
        self.runtime: MDCTrainingRuntime | None = None
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer] | None = None
        self.model_config: MDCModelConfig | None = None
        self.base_checkpoint: dict[str, Any] | None = None
        self._loaded_from_resume = False
        self._global_step = 0
        self._tokens_seen = 0
        self._epoch = 0
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._tokens_seen_history: list[int] = []
        self._best_val_loss = math.inf
        self._final_train_loss: float | None = None
        self._final_val_loss: float | None = None

    @property
    def is_main_process(self) -> bool:
        runtime = getattr(self, "runtime", None)
        return runtime is None or runtime.is_main_process

    def train(self) -> InstructionTrainingResult:
        self._validate_config()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self._log("🚀 Starting profile-to-protein instruction training")
        self._log(f"🧾 Instruction JSONL files: {len(self.instruction_paths)}")
        for index, path in enumerate(self.instruction_paths, start=1):
            self._log(f"   📄 [{index}/{len(self.instruction_paths)}] {path}")
        self._log(f"📥 Base checkpoint: {self.base_checkpoint_path}")
        self._log(f"📁 Output dir: {self.output_dir}")
        self._log(f"🧬 Artifact dir: {self.artifact_dir}")
        self._log(
            "⚙️  Training config: "
            f"epochs={self.config.num_epochs} | batch_size={self.config.batch_size} | "
            f"grad_accum={self.config.gradient_accumulation_steps} | eval_freq={self.config.eval_freq} | "
            f"eval_batches={self.config.eval_batches} | max_steps={self.config.max_steps}"
        )

        self._log("📦 [1/7] Preparing profile-aware instruction artifacts...")
        self.artifacts = self._load_or_build_artifacts()
        self._log(
            "✅ Artifacts ready | "
            f"vocab_size={self.artifacts.layout.vocab_size} | "
            f"records={self.artifacts.record_count:,} | sequence_type={self.artifacts.sequence_type}"
        )

        self._log("🧠 [2/7] Building model from base checkpoint...")
        self._setup_model()
        param_count = sum(p.numel() for p in unwrap_mdc_training_model(self.model).parameters())
        self._log(
            f"✅ Model ready | params={param_count:,} | context_length={self.model_config.context_length} | "
            f"vocab_size={self.model_config.vocab_size}"
        )

        self._log("⚙️  [3/7] Preparing runtime...")
        self._setup_runtime()
        self._log("🔧 [4/7] Creating optimizer...")
        self._setup_optimizer()
        self._maybe_restore_resume_state()

        audit_path = self.output_dir / "instruction_audit.json"
        if self.config.reuse_existing_audit and audit_path.exists():
            self._log(f"🔎 [5/7] Reusing existing instruction audit: {audit_path}")
            if self.is_main_process:
                self._save_config_snapshot()
        elif self.config.audit_before_training:
            self._log("🔎 [5/7] Auditing instruction JSONL...")
            audit = audit_instruction_jsonl(
                self.instruction_paths,
                default_sequence_type=self.config.default_sequence_type,
                instruction_field=self.config.instruction_field,
                input_field=self.config.input_field,
                output_field=self.config.output_field,
                prompt_format=self.config.prompt_format,
            )
            if self.is_main_process:
                _write_json(audit_path, audit.to_dict())
                self._log(f"📝 Instruction audit written: {audit_path}")
                self._save_config_snapshot()
        else:
            self._log("🔎 [5/7] Instruction audit skipped by config.")
            if self.is_main_process:
                self._save_config_snapshot()

        train_count: int | None = None
        per_epoch_step_limit: int | None = None
        if self._should_count_splits_before_training():
            self._log("📊 [6/7] Counting train/validation rows in one pass...")
            counts = count_instruction_split_records_by_split(
                self.instruction_paths,
                train_ratio=self.config.train_ratio,
                split_seed=self.config.split_seed,
                artifacts=self.artifacts,
                default_sequence_type=self.config.default_sequence_type,
                instruction_field=self.config.instruction_field,
                input_field=self.config.input_field,
                output_field=self.config.output_field,
                prompt_format=self.config.prompt_format,
                max_sequence_length=int(self.model_config.context_length),
                progress_every=self.config.count_progress_every,
                progress_callback=self._log_split_count_progress,
            )
            train_count = counts["train"]
            self._log(
                "✅ Instruction rows counted | "
                f"train={train_count:,} | val={counts['val']:,} | "
                f"skipped_long={counts['skipped_for_length']:,} | train_ratio={self.config.train_ratio:.3f}"
            )
            per_epoch_step_limit = self._per_epoch_step_limit(train_count)
            if per_epoch_step_limit is not None:
                self._log(f"🔢 Distributed per-epoch optimizer step limit: {per_epoch_step_limit:,}")
        else:
            self._log("📊 [6/7] Streaming train/validation rows directly; pre-count skipped.")

        self._log("📂 Building train loader...")
        train_loader = self._build_loader("train", drop_last=self.runtime.distributed)
        self._log("📂 Building validation loader...")
        val_loader = self._build_loader("val", drop_last=False)
        started_at = time.time()

        try:
            self._log("🏋️ [7/7] Starting instruction training loop...")
            self._training_loop(train_loader, val_loader, per_epoch_step_limit=per_epoch_step_limit)
        finally:
            if self.runtime is not None and self.runtime.distributed:
                cleanup_mdc_distributed_training()

        elapsed_minutes = (time.time() - started_at) / 60.0
        if self.is_main_process:
            if self.config.save_final:
                self._log(f"💾 Saving final checkpoint: {self.checkpoint_final_path}")
                self._save_checkpoint(self.checkpoint_final_path)
            self._save_loss_plot()
            summary = self._build_summary(elapsed_minutes=elapsed_minutes)
            _write_json(self.training_summary_path, summary)
            self._log(f"📝 Training summary written: {self.training_summary_path}")
            self._log(f"🎉 Instruction training complete in {elapsed_minutes:.2f} min")

        return self._build_result()

    def _validate_config(self) -> None:
        if not self.base_checkpoint_path.is_file():
            raise FileNotFoundError(f"Base checkpoint not found: {self.base_checkpoint_path}")
        if not 0.0 < self.config.train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1.")
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.config.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.config.num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        if self.config.eval_batches <= 0:
            raise ValueError("eval_batches must be positive.")
        if self.config.optimizer_type not in {"muon", "adamw"}:
            raise ValueError("optimizer_type must be one of: 'muon', 'adamw'.")
        if self.config.mixed_precision not in {"auto", "no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of: 'auto', 'no', 'fp16', 'bf16'.")

    def _load_or_build_artifacts(self) -> MDCProfileSequencePretrainArtifacts:
        tokenizer_map_path = self.artifact_dir / "tokenizer_map.json"
        if tokenizer_map_path.exists() and not self.config.rebuild_artifacts:
            artifacts = MDCProfileSequencePretrainArtifacts.from_tokenizer_map_file(tokenizer_map_path)
            missing_profile_chars = _missing_profile_base_charset(artifacts)
            if not missing_profile_chars:
                self._log(f"   ✅ Reusing tokenizer map: {tokenizer_map_path}")
                return artifacts
            self._log(
                "   Rebuilding tokenizer map because profile tokenizer is missing "
                f"instruction-safe characters: {_format_missing_profile_chars(missing_profile_chars)}"
            )

        artifact_source_jsonl = (
            Path(self.config.artifact_source_jsonl).expanduser().resolve()
            if self.config.artifact_source_jsonl is not None
            else None
        )
        artifact_record_paths: Path | Sequence[Path]
        adjacent_tokenizer_map_base: Path | None = None
        if artifact_source_jsonl is not None and artifact_source_jsonl.is_file():
            artifact_record_paths = artifact_source_jsonl
            adjacent_tokenizer_map_base = artifact_source_jsonl
        elif self.instruction_paths:
            artifact_record_paths = self.instruction_paths
            if artifact_source_jsonl is not None and len(self.instruction_paths) == 1:
                adjacent_tokenizer_map_base = self.instruction_paths[0]
        else:
            raise ValueError("At least one instruction JSONL path is required to build tokenizer artifacts.")

        self._log(f"   🧾 Artifact source: {_format_paths_for_log(artifact_record_paths)}")
        self._log(f"   🔬 Sampling up to {self.config.artifact_profile_sample_size:,} records for artifacts...")
        records = _sample_instruction_records_for_artifacts(
            artifact_record_paths,
            default_sequence_type=self.config.default_sequence_type,
            instruction_field=self.config.instruction_field,
            input_field=self.config.input_field,
            output_field=self.config.output_field,
            prompt_format=self.config.prompt_format,
            sample_size=self.config.artifact_profile_sample_size,
        )
        sequence_types = {record.sequence_type for record in records}
        if len(sequence_types) != 1:
            raise ValueError(
                "The current MDC profile-aware instruction format requires a single sequence_type "
                "across instruction records."
            )

        sequence_tokenizer_map_path = self.config.sequence_tokenizer_map_path
        if sequence_tokenizer_map_path is None and self.config.auto_detect_sequence_tokenizer_map:
            if adjacent_tokenizer_map_base is not None:
                adjacent_tokenizer_map_path = adjacent_tokenizer_map_base.with_name("tokenizer_map.json")
                if adjacent_tokenizer_map_path.exists():
                    sequence_tokenizer_map_path = adjacent_tokenizer_map_path

        if sequence_tokenizer_map_path is not None:
            self._log(f"   🧬 Sequence tokenizer map: {sequence_tokenizer_map_path}")
        self._log(
            "   🛠️  Building artifacts | "
            f"records={len(records):,} | sequence_type={next(iter(sequence_types))} | "
            f"profile_vocab_size={self.config.profile_vocab_size} | kmer_size={self.config.kmer_size}"
        )
        save_mdc_profile_sequence_pretrain_artifacts(
            records,
            self.artifact_dir,
            sequence_type=next(iter(sequence_types)),
            kmer_size=self.config.kmer_size,
            profile_vocab_size=self.config.profile_vocab_size,
            sequence_tokenizer_map_path=sequence_tokenizer_map_path,
        )
        return MDCProfileSequencePretrainArtifacts.from_tokenizer_map_file(tokenizer_map_path)

    def _setup_model(self) -> None:
        self._log(f"   📥 Loading checkpoint payload: {self.base_checkpoint_path}")
        self.base_checkpoint = torch.load(self.base_checkpoint_path, map_location="cpu")
        base_config = _load_mdc_model_config(self.base_checkpoint)
        self.model_config = base_config.with_vocab_size(self.artifacts.layout.vocab_size)
        app = MicrobialDecoderCoreApp(self.model_config, self.artifacts.layout)

        if self._is_instruction_checkpoint(self.base_checkpoint):
            self._log("   🔁 Base checkpoint is already an instruction checkpoint; restoring full app state.")
            app.load_state_dict(
                _normalize_app_state_dict(self.base_checkpoint["model_state_dict"]),
                strict=True,
            )
        else:
            self._log("   🧩 Loading protein backbone for instruction tuning.")
            load_protein_pretrain_checkpoint_for_profile_tuning(
                self.base_checkpoint_path,
                model=app.model,
                strict_backbone=self.config.strict_backbone,
            )
        self.model = app

    def _setup_runtime(self) -> None:
        requested_device = self.config.device
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.runtime = prepare_mdc_training_runtime(
            self.model,
            device=requested_device,
            multi_gpu=self.config.multi_gpu_mode,
            find_unused_parameters=self.config.ddp_find_unused_parameters,
            data_parallel_device_ids=self.config.data_parallel_device_ids,
        )
        self.model = self.runtime.model
        self._log(
            f"✅ Runtime ready | device={self.runtime.device} | distributed={self.runtime.distributed} "
            f"| data_parallel={self.runtime.data_parallel} | rank={self.runtime.rank}/{self.runtime.world_size}"
        )

    def _setup_optimizer(self) -> None:
        optimizer_config = {
            "type": self.config.optimizer_type,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "fused": self.config.fused,
        }
        self.optimizer = create_protein_training_optimizer(
            self.model,
            optimizer_config,
            device=self.runtime.device,
        )
        if self.is_main_process:
            self._log(f"✅ Optimizer ready | {describe_protein_training_optimizers(self.optimizer)}")

    def _maybe_restore_resume_state(self) -> None:
        if not self.config.resume_if_available:
            self._log("⏭️  Resume disabled; starting from base checkpoint.")
            return
        if not self.checkpoint_best_path.exists():
            self._log(f"🆕 No instruction best resume checkpoint found: {self.checkpoint_best_path}")
            return
        self._log(f"📥 Loading instruction best resume checkpoint: {self.checkpoint_best_path}")
        checkpoint = torch.load(self.checkpoint_best_path, map_location=self.runtime.device)
        if not self._is_instruction_checkpoint(checkpoint):
            raise ValueError(f"Resume checkpoint is not an instruction checkpoint: {self.checkpoint_best_path}")

        unwrap_mdc_training_model(self.model).load_state_dict(
            _normalize_app_state_dict(checkpoint["model_state_dict"]),
            strict=True,
        )
        if self.config.restore_optimizer_state and checkpoint.get("optimizer_state_dict") is not None:
            _load_optimizer_state_dict(self.optimizer, checkpoint["optimizer_state_dict"])

        self._epoch = int(checkpoint.get("epoch", 0))
        self._global_step = int(checkpoint.get("global_step", 0))
        self._tokens_seen = int(checkpoint.get("tokens_seen", 0))
        self._train_losses = list(checkpoint.get("train_losses", []))
        self._val_losses = list(checkpoint.get("val_losses", []))
        self._tokens_seen_history = list(checkpoint.get("tokens_seen_history", []))
        best_val_loss = checkpoint.get("best_val_loss")
        if _is_finite(best_val_loss):
            self._best_val_loss = float(best_val_loss)
        self._loaded_from_resume = True
        self._log(
            f"✅ Resumed instruction training | epoch={self._epoch} "
            f"| step={self._global_step} | tokens={self._tokens_seen:,}"
        )

    def _build_loader(self, split: str, *, drop_last: bool):
        return create_instruction_dataloader(
            self.artifacts,
            self.instruction_paths,
            split=split,
            train_ratio=self.config.train_ratio,
            split_seed=self.config.split_seed,
            default_sequence_type=self.config.default_sequence_type,
            instruction_field=self.config.instruction_field,
            input_field=self.config.input_field,
            output_field=self.config.output_field,
            prompt_format=self.config.prompt_format,
            max_sequence_length=int(self.model_config.context_length),
            batch_size=self.config.batch_size,
            drop_last=drop_last,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle_files=self.config.shuffle_files if split == "train" else False,
            shuffle_records=self.config.shuffle_records if split == "train" else False,
            shuffle_buffer_size=self.config.shuffle_buffer_size,
            seed=self.config.seed,
            distributed=self.runtime.distributed,
            rank=self.runtime.rank,
            world_size=self.runtime.world_size,
            train_on_prompt=self.config.train_on_prompt,
            include_separator_in_loss=self.config.include_separator_in_loss,
            include_eos_in_loss=self.config.include_eos_in_loss,
        )

    def _per_epoch_step_limit(self, train_count: int) -> int | None:
        if not self.runtime.distributed:
            return None
        steps = train_count // (self.config.batch_size * self.runtime.world_size)
        if steps <= 0:
            raise ValueError(
                "Not enough training rows for distributed training with drop_last=true. "
                "Reduce batch_size or world_size."
            )
        return steps

    def _should_count_splits_before_training(self) -> bool:
        value = self.config.count_splits_before_training
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "auto":
                should_count = bool(self.runtime.distributed)
            elif normalized in {"1", "true", "yes", "on"}:
                should_count = True
            elif normalized in {"0", "false", "no", "off"}:
                should_count = False
            else:
                raise ValueError(
                    "count_splits_before_training must be true, false, or 'auto'."
                )
        else:
            should_count = bool(value)
        if self.runtime.distributed and not should_count:
            raise ValueError(
                "Distributed instruction training requires count_splits_before_training=true or 'auto' "
                "so all ranks stop at the same per-epoch step."
            )
        return should_count

    def _log_split_count_progress(
        self,
        rows_seen: int,
        train_count: int,
        val_count: int,
        skipped_for_length: int,
    ) -> None:
        self._log(
            "   📊 Count progress | "
            f"rows={rows_seen:,} | train={train_count:,} | val={val_count:,} | "
            f"skipped_long={skipped_for_length:,}"
        )

    def _training_loop(self, train_loader, val_loader, *, per_epoch_step_limit: int | None) -> None:
        amp_dtype = self._resolve_autocast_dtype()
        use_autocast = amp_dtype is not None and self.runtime.device.type == "cuda"
        use_grad_scaler = use_autocast and amp_dtype == torch.float16
        grad_scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None
        optimizers = _optimizer_list(self.optimizer)
        scheduler = build_warmup_cosine_scheduler(
            optimizers,
            LRScheduleConfig(
                enabled=self.config.lr_scheduler == "cosine",
                warmup_steps=self.config.warmup_steps,
                warmup_ratio=self.config.warmup_ratio,
                min_lr_ratio=self.config.min_lr_ratio,
                decay_steps=self.config.lr_decay_steps,
            ),
            max_steps=self.config.max_steps,
            last_step=self._global_step,
        )
        if scheduler is not None:
            horizon = "cosine decay" if scheduler.decays else "warmup then hold"
            self._log(
                f"📉 LR schedule: warmup={scheduler.warmup_steps} steps | {horizon} | "
                f"min_lr_ratio={scheduler.min_lr_ratio} | start_lr={scheduler.current_lr():.2e}"
            )
        micro_step = 0
        start_epoch = self._epoch
        log_every_steps = self._log_every_steps()

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        for epoch_offset in range(1, self.config.num_epochs + 1):
            epoch = start_epoch + epoch_offset
            self._epoch = epoch
            set_mdc_data_loader_epoch(train_loader, epoch - 1)
            unwrap_mdc_training_model(self.model).train()
            max_steps_str = f"/{self.config.max_steps}" if self.config.max_steps else ""
            self._log(
                f"📈 Epoch {epoch}/{start_epoch + self.config.num_epochs} | "
                f"step={self._global_step}{max_steps_str} | tokens={self._tokens_seen:,}"
            )

            local_batches = 0
            for batch in train_loader:
                if per_epoch_step_limit is not None and local_batches >= per_epoch_step_limit:
                    break
                batch = _move_batch(batch, self.runtime.device)
                micro_step += 1
                local_batches += 1

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_autocast):
                    logits = self.model(batch.input_ids, attention_mask=batch.attention_mask)
                    loss = compute_mdc_causal_lm_loss(logits, batch.labels)
                    scaled_loss = loss / self.config.gradient_accumulation_steps

                if grad_scaler is not None:
                    grad_scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                self._tokens_seen += self._count_supervised_tokens(batch)

                if micro_step % self.config.gradient_accumulation_steps == 0:
                    if self.config.grad_clip_norm is not None:
                        if grad_scaler is not None:
                            for opt in optimizers:
                                grad_scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

                    if grad_scaler is not None:
                        for opt in optimizers:
                            grad_scaler.step(opt)
                        grad_scaler.update()
                    else:
                        for opt in optimizers:
                            opt.step()

                    for opt in optimizers:
                        opt.zero_grad(set_to_none=True)

                    self._global_step += 1
                    if scheduler is not None:
                        scheduler.step()
                    if self.is_main_process and self._global_step % log_every_steps == 0:
                        lr_text = _format_optimizer_lrs(optimizers)
                        self._log(
                            f"  🔄 step={self._global_step} | epoch={epoch} | micro_step={micro_step} | "
                            f"loss={float(loss.item()):.4f} | tokens={self._tokens_seen:,} | lr={lr_text}"
                        )

                    if self.config.eval_freq > 0 and self._global_step % self.config.eval_freq == 0:
                        self._run_eval(val_loader, train_loader=train_loader)

                    if (
                        self.is_main_process
                        and self.config.save_last
                        and self.config.save_every_steps
                        and self._global_step % self.config.save_every_steps == 0
                    ):
                        self._log(f"  💾 Saving checkpoint at step {self._global_step}: {self.checkpoint_last_path}")
                        self._save_checkpoint(self.checkpoint_last_path)

                    if self.config.max_steps is not None and self._global_step >= self.config.max_steps:
                        break

            if micro_step % self.config.gradient_accumulation_steps != 0:
                if self.config.grad_clip_norm is not None:
                    if grad_scaler is not None:
                        for opt in optimizers:
                            grad_scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                if grad_scaler is not None:
                    for opt in optimizers:
                        grad_scaler.step(opt)
                    grad_scaler.update()
                else:
                    for opt in optimizers:
                        opt.step()
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                self._global_step += 1
                if scheduler is not None:
                    scheduler.step()
                if self.is_main_process:
                    self._log(
                        f"  🔄 step={self._global_step} | epoch={epoch} | leftover_micro_steps="
                        f"{micro_step % self.config.gradient_accumulation_steps} | tokens={self._tokens_seen:,}"
                    )

            self._barrier()
            if self.is_main_process:
                self._log(f"  📊 Epoch {epoch} end — evaluating...")
            self._run_eval(val_loader, train_loader=train_loader)
            if self.is_main_process and self.config.save_last:
                self._log(f"  💾 Saving epoch checkpoint: {self.checkpoint_last_path}")
                self._save_checkpoint(self.checkpoint_last_path)
            self._barrier()

            if self.config.max_steps is not None and self._global_step >= self.config.max_steps:
                self._log(f"⏹️  Reached max_steps={self.config.max_steps}")
                break

    def _run_eval(self, val_loader, *, train_loader) -> None:
        if self.is_main_process:
            self._log(
                f"  🔎 Running eval | step={self._global_step} | "
                f"max_batches={self.config.eval_batches}"
            )
        train_loss = self._evaluate(train_loader, max_batches=self.config.eval_batches)
        val_loss = self._evaluate(val_loader, max_batches=self.config.eval_batches)
        if self.is_main_process:
            self._train_losses.append(train_loss)
            self._val_losses.append(val_loss)
            self._tokens_seen_history.append(self._tokens_seen)
            self._final_train_loss = train_loss
            self._final_val_loss = val_loss
            improved = self.config.save_best and _is_finite(val_loss) and val_loss < self._best_val_loss
            if improved:
                self._best_val_loss = float(val_loss)
            self._append_metrics(train_loss, val_loss)
            best_marker = " 🏆 new best val_loss!" if improved else ""
            self._log(
                f"  📊 eval step={self._global_step} | train={_format_loss(train_loss)} "
                f"| val={_format_loss(val_loss)} | tokens={self._tokens_seen:,}{best_marker}"
            )
            if improved:
                self._log(f"  💾 Saving best checkpoint: {self.checkpoint_best_path}")
                self._save_checkpoint(self.checkpoint_best_path)

    def _evaluate(self, loader, *, max_batches: int) -> float:
        model_was_training = unwrap_mdc_training_model(self.model).training
        unwrap_mdc_training_model(self.model).eval()
        amp_dtype = self._resolve_autocast_dtype()
        use_autocast = amp_dtype is not None and self.runtime.device.type == "cuda"
        loss_sum = torch.zeros((), dtype=torch.float64, device=self.runtime.device)
        batch_count = torch.zeros((), dtype=torch.float64, device=self.runtime.device)
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if batch_index >= max_batches:
                    break
                batch = _move_batch(batch, self.runtime.device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_autocast):
                    logits = self.model(batch.input_ids, attention_mask=batch.attention_mask)
                    loss = compute_mdc_causal_lm_loss(logits, batch.labels)
                if torch.isfinite(loss):
                    loss_sum += loss.detach().double()
                    batch_count += 1

        if self.runtime.distributed:
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(batch_count, op=torch.distributed.ReduceOp.SUM)

        if model_was_training:
            unwrap_mdc_training_model(self.model).train()
        if float(batch_count.item()) == 0.0:
            return float("nan")
        return float((loss_sum / batch_count).item())

    def _save_checkpoint(self, path: Path) -> Path:
        app = unwrap_mdc_training_model(self.model)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_family": INSTRUCTION_CHECKPOINT_FAMILY,
            "base_checkpoint_path": str(self.base_checkpoint_path),
            "instruction_jsonl_paths": [str(path) for path in self.instruction_paths],
            "artifact_dir": str(self.artifact_dir),
            "model_state_dict": app.state_dict(),
            "optimizer_state_dict": _optimizer_state_dict(self.optimizer),
            "model_config": dict(app.model.cfg),
            "layout": app.layout.to_config_dict(),
            "epoch": self._epoch,
            "global_step": self._global_step,
            "tokens_seen": self._tokens_seen,
            "train_losses": list(self._train_losses),
            "val_losses": list(self._val_losses),
            "tokens_seen_history": list(self._tokens_seen_history),
            "best_val_loss": None if math.isinf(self._best_val_loss) else self._best_val_loss,
            "training_args": _json_safe(self.config.__dict__),
            "optimizer_types": describe_protein_training_optimizers(self.optimizer),
        }
        torch.save(payload, path)
        return path

    def _resolve_autocast_dtype(self) -> torch.dtype | None:
        if self.config.mixed_precision == "no" or self.runtime.device.type != "cuda":
            return None
        if self.config.mixed_precision == "bf16":
            return torch.bfloat16
        if self.config.mixed_precision == "fp16":
            return torch.float16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _count_supervised_tokens(self, batch: CausalLMBatch) -> int:
        count = torch.tensor(
            int((batch.labels != IGNORE_INDEX).sum().item()),
            dtype=torch.long,
            device=self.runtime.device,
        )
        if self.runtime.distributed:
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        return int(count.item())

    def _append_metrics(self, train_loss: float, val_loss: float) -> None:
        payload = {
            "epoch": self._epoch,
            "global_step": self._global_step,
            "tokens_seen": self._tokens_seen,
            "train_loss": _json_loss(train_loss),
            "val_loss": _json_loss(val_loss),
            "train_perplexity": _json_perplexity(train_loss),
            "val_perplexity": _json_perplexity(val_loss),
            "learning_rate": self.config.learning_rate,
            "best_val_loss": None if math.isinf(self._best_val_loss) else self._best_val_loss,
            "checkpoint_last_path": str(self.checkpoint_last_path),
            "checkpoint_best_path": str(self.checkpoint_best_path),
        }
        self.metrics_history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        self._log(f"  🧾 Metrics appended: {self.metrics_history_path}")

    def _save_config_snapshot(self) -> None:
        snapshot_path = self.output_dir / "training_config.snapshot.json"
        _write_json(snapshot_path, _json_safe(self.config.__dict__))
        self._log(f"📝 Config snapshot written: {snapshot_path}")

    def _save_loss_plot(self) -> None:
        if not self._train_losses:
            self._log("📉 Loss plot skipped: no eval losses recorded.")
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(7, 4))
        x_points = list(range(1, len(self._train_losses) + 1))
        ax1.plot(x_points, self._train_losses, label="Train loss")
        ax1.plot(x_points, self._val_losses, label="Val loss", linestyle="-.")
        ax1.set_xlabel("Evaluation step")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax2 = ax1.twiny()
        ax2.plot(self._tokens_seen_history, self._train_losses, alpha=0)
        ax2.set_xlabel("Tokens seen")
        fig.tight_layout()
        self.loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.loss_plot_path, dpi=150)
        plt.close(fig)
        self._log(f"📉 Loss plot written: {self.loss_plot_path}")

    def _build_summary(self, *, elapsed_minutes: float) -> dict[str, Any]:
        return {
            **self._build_result().to_dict(),
            "elapsed_minutes": elapsed_minutes,
            "instruction_jsonl_paths": [str(path) for path in self.instruction_paths],
            "base_checkpoint_path": str(self.base_checkpoint_path),
            "artifact_dir": str(self.artifact_dir),
            "loaded_from_resume": self._loaded_from_resume,
            "training_args": _json_safe(self.config.__dict__),
        }

    def _build_result(self) -> InstructionTrainingResult:
        return InstructionTrainingResult(
            output_dir=self.output_dir,
            checkpoint_last_path=self.checkpoint_last_path if self.checkpoint_last_path.exists() else None,
            checkpoint_best_path=self.checkpoint_best_path if self.checkpoint_best_path.exists() else None,
            checkpoint_final_path=self.checkpoint_final_path if self.checkpoint_final_path.exists() else None,
            metrics_history_path=self.metrics_history_path,
            training_summary_path=self.training_summary_path,
            loss_plot_path=self.loss_plot_path,
            global_step=self._global_step,
            tokens_seen=self._tokens_seen,
            epochs_completed=self._epoch,
            best_val_loss=None if math.isinf(self._best_val_loss) else self._best_val_loss,
            final_train_loss=self._final_train_loss,
            final_val_loss=self._final_val_loss,
        )

    def _barrier(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _log(self, message: str) -> None:
        if self.is_main_process:
            _print_log(message)

    def _log_every_steps(self) -> int:
        if self.config.log_every_steps is not None:
            return max(1, int(self.config.log_every_steps))
        if self.config.eval_freq > 0:
            return max(1, self.config.eval_freq // 2)
        return 50

    @staticmethod
    def _is_instruction_checkpoint(checkpoint: Mapping[str, Any]) -> bool:
        return checkpoint.get("model_family") == INSTRUCTION_CHECKPOINT_FAMILY


def _load_mdc_model_config(checkpoint: Mapping[str, Any]) -> MDCModelConfig:
    payload = dict(checkpoint.get("model_config") or {})
    if not payload:
        raise ValueError("Checkpoint is missing model_config.")
    if payload.get("layer_types") is not None:
        payload["layer_types"] = tuple(payload["layer_types"])
    payload["dtype"] = _coerce_dtype(payload.get("dtype", torch.float32))
    return MDCModelConfig(**payload)


def _coerce_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    text = str(value).replace("torch.", "")
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(text, torch.float32)


def _move_batch(batch: CausalLMBatch, device: torch.device) -> CausalLMBatch:
    return CausalLMBatch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        labels=batch.labels.to(device),
    )


def _optimizer_list(
    optimizer: torch.optim.Optimizer | Sequence[torch.optim.Optimizer] | None,
) -> list[torch.optim.Optimizer]:
    if optimizer is None:
        raise ValueError("optimizer has not been initialized.")
    if isinstance(optimizer, torch.optim.Optimizer):
        return [optimizer]
    return list(optimizer)


def _optimizer_state_dict(
    optimizer: torch.optim.Optimizer | Sequence[torch.optim.Optimizer] | None,
):
    optimizers = _optimizer_list(optimizer)
    if len(optimizers) == 1:
        return optimizers[0].state_dict()
    return [opt.state_dict() for opt in optimizers]


def _load_optimizer_state_dict(
    optimizer: torch.optim.Optimizer | Sequence[torch.optim.Optimizer] | None,
    state_dict,
) -> None:
    optimizers = _optimizer_list(optimizer)
    if isinstance(state_dict, list):
        if len(state_dict) != len(optimizers):
            raise ValueError("Optimizer state count does not match optimizer count.")
        for opt, state in zip(optimizers, state_dict, strict=True):
            opt.load_state_dict(state)
        return
    if len(optimizers) != 1:
        raise ValueError("Checkpoint has one optimizer state but the run uses multiple optimizers.")
    optimizers[0].load_state_dict(state_dict)


def _normalize_app_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(state_dict)
    for prefix in ("module.", "_orig_mod."):
        while normalized and all(key.startswith(prefix) for key in normalized):
            normalized = {key[len(prefix):]: value for key, value in normalized.items()}
    return normalized


def _json_loss(value: float) -> float | None:
    return float(value) if _is_finite(value) else None


def _json_perplexity(value: float) -> float | None:
    if not _is_finite(value):
        return None
    return math.exp(min(float(value), 50.0))


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _format_loss(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.4f}"


def _format_paths_for_log(paths: Path | Sequence[Path]) -> str:
    if isinstance(paths, Path):
        return str(paths)
    return ", ".join(str(path) for path in paths)


def _missing_profile_base_charset(artifacts: MDCProfileSequencePretrainArtifacts) -> tuple[str, ...]:
    return tuple(
        character
        for character in DEFAULT_PROFILE_BASE_CHARSET
        if character not in artifacts.profile_tokenizer.str_to_int
    )


def _format_missing_profile_chars(characters: Sequence[str], *, limit: int = 12) -> str:
    preview = ", ".join(repr(character) for character in characters[:limit])
    if len(characters) > limit:
        preview += f", ... (+{len(characters) - limit} more)"
    return preview or "none"


def _format_optimizer_lrs(optimizers: Sequence[torch.optim.Optimizer]) -> str:
    values: list[str] = []
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            lr = group.get("lr")
            if lr is not None:
                values.append(f"{float(lr):.2e}")
    if not values:
        return "n/a"
    unique_values = sorted(set(values))
    return ",".join(unique_values)


def _print_log(message: str) -> None:
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe_message = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe_message, flush=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value

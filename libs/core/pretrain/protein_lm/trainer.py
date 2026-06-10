from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from libs.core.interfaces import CausalLMBatch
from libs.core.mdc import MDCDecoderModel
from libs.core.mdc.config import MDCModelConfig
from libs.core.pretrain.distributed import (
    MDCTrainingRuntime,
    set_mdc_data_loader_epoch,
    prepare_mdc_training_runtime,
)
from libs.core.pretrain.training import (
    compute_mdc_causal_lm_loss,
)
from libs.core.pretrain.training_config import (
    apply_protein_training_optimizer_settings,
    build_protein_training_data_config,
    create_protein_training_optimizer,
    load_protein_training_config,
)
from libs.notebook_runtime import apply_training_config_notebook_overrides
from libs.core.pretrain.protein_lm.core import (
    build_or_load_protein_tokenizer_from_text_paths,
    discover_protein_train_text_paths,
    load_protein_pretrain_checkpoint,
    _natural_path_sort_key,
)
from libs.core.pretrain.protein_lm.support.backbone import (
    build_mdc_config_from_progen_config,
    build_progen_config,
    is_supported_protein_checkpoint_family,
)
from libs.core.pretrain.protein_lm.resume_state import (
    create_resume_state,
    load_resume_state,
    save_resume_state,
    update_resume_state_metrics,
    update_resume_state_progress,
)
from libs.core.pretrain.protein_lm.memory import run_preflight_vram_check
from libs.core.pretrain.protein_lm.services import (
    CheckpointService,
    DataLoaderFactory,
    Evaluator,
    GradientAccumulator,
    LoaderBundle,
    MetricsWriter,
    OptimizerBundle,
    PrecisionContext,
    TrainerComponents,
    TrainerState,
    TrainingLoopSettings,
    optimizer_list,
    resolve_precision_context,
    step_optimizers,
    zero_grad,
)
from libs.data.training.tokenizer import SequenceTokenizer


@dataclass(slots=True)
class ProteinPretrainResult:
    checkpoint_path: Path
    best_checkpoint_path: Path | None
    final_checkpoint_path: Path | None
    resume_state_path: Path | None
    global_step: int
    tokens_seen: int
    epochs_completed: int
    best_loss: float | None
    final_train_loss: float | None
    final_val_loss: float | None


def _is_finite_loss(value: float | None) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _format_eval_loss(value: float, *, available: bool = True) -> str:
    if not available:
        return "n/a"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.4f}"


def _restore_best_val_loss(checkpoint_meta: Mapping[str, Any], best_loss: float | None) -> float:
    if best_loss is None or not _is_finite_loss(best_loss):
        return math.inf
    metric_name = checkpoint_meta.get("best_metric_name")
    if metric_name is not None and metric_name != "val_loss":
        return math.inf
    val_losses = checkpoint_meta.get("val_losses", [])
    if val_losses and not any(_is_finite_loss(loss) for loss in val_losses):
        return math.inf
    return float(best_loss)


class ProteinPretrainTrainer:
    def __init__(
        self,
        project_root: Path | str,
        config_path: Path | str | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.config = apply_training_config_notebook_overrides(
            load_protein_training_config(self.project_root, config_path=config_path)
        )
        self.minio_data_config = build_protein_training_data_config(self.project_root, self.config)

        self._paths = self.config["paths"]
        self._data_cfg = self.config["data"]
        self._model_cfg = self.config["model"]
        self._training_cfg = self.config["training"]
        self._optimizer_cfg = self.config["optimizer"]
        self._runtime_cfg = self.config["runtime"]
        self._resume_cfg = self.config["resume"]
        self._minio_cfg = self.config["minio"]
        self._mode_cfg = self.config["mode"]

        self._components = TrainerComponents()
        self._state = TrainerState()
        self._resume_state: dict[str, Any] | None = None

        self._checkpoint_service = CheckpointService(
            paths=self._paths,
            resume_cfg=self._resume_cfg,
            data_cfg=self._data_cfg,
            model_cfg=self._model_cfg,
            optimizer_cfg=self._optimizer_cfg,
            runtime_cfg=self._runtime_cfg,
            minio_cfg=self._minio_cfg,
        )
        self._metrics_writer = MetricsWriter(self._paths.get("metrics_history_path"))
        self._evaluator = Evaluator()

    def _log(self, message: str) -> None:
        if self.runtime is None or self.runtime.is_main_process:
            print(message, flush=True)

    @property
    def is_main_process(self) -> bool:
        return self.runtime is not None and self.runtime.is_main_process

    def _ensure_components(self) -> TrainerComponents:
        components = getattr(self, "_components", None)
        if components is None:
            components = TrainerComponents()
            self._components = components
        return components

    def _ensure_state(self) -> TrainerState:
        state = getattr(self, "_state", None)
        if state is None:
            state = TrainerState()
            self._state = state
        return state

    @property
    def runtime(self) -> MDCTrainingRuntime | None:
        return self._ensure_components().runtime

    @runtime.setter
    def runtime(self, value: MDCTrainingRuntime | None) -> None:
        self._ensure_components().runtime = value

    @property
    def model(self) -> torch.nn.Module | None:
        return self._ensure_components().model

    @model.setter
    def model(self, value: torch.nn.Module | None) -> None:
        self._ensure_components().model = value

    @property
    def optimizer(self) -> OptimizerBundle | None:
        return self._ensure_components().optimizer

    @optimizer.setter
    def optimizer(self, value: OptimizerBundle | None) -> None:
        self._ensure_components().optimizer = value

    @property
    def tokenizer(self) -> SequenceTokenizer | None:
        return self._ensure_components().tokenizer

    @tokenizer.setter
    def tokenizer(self, value: SequenceTokenizer | None) -> None:
        self._ensure_components().tokenizer = value

    @property
    def model_config(self) -> MDCModelConfig | None:
        return self._ensure_components().model_config

    @model_config.setter
    def model_config(self, value: MDCModelConfig | None) -> None:
        self._ensure_components().model_config = value

    @property
    def _global_step(self) -> int:
        return self._ensure_state().global_step

    @_global_step.setter
    def _global_step(self, value: int) -> None:
        self._ensure_state().global_step = int(value)

    @property
    def _tokens_seen(self) -> int:
        return self._ensure_state().tokens_seen

    @_tokens_seen.setter
    def _tokens_seen(self, value: int) -> None:
        self._ensure_state().tokens_seen = int(value)

    @property
    def _epoch(self) -> int:
        return self._ensure_state().epoch

    @_epoch.setter
    def _epoch(self, value: int) -> None:
        self._ensure_state().epoch = int(value)

    @property
    def _train_losses(self) -> list[float]:
        return self._ensure_state().train_losses

    @_train_losses.setter
    def _train_losses(self, value: list[float]) -> None:
        self._ensure_state().train_losses = list(value)

    @property
    def _val_losses(self) -> list[float]:
        return self._ensure_state().val_losses

    @_val_losses.setter
    def _val_losses(self, value: list[float]) -> None:
        self._ensure_state().val_losses = list(value)

    @property
    def _best_val_loss(self) -> float:
        return self._ensure_state().best_val_loss

    @_best_val_loss.setter
    def _best_val_loss(self, value: float) -> None:
        self._ensure_state().best_val_loss = float(value)

    @property
    def _best_metric_name(self) -> str:
        return self._ensure_state().best_metric_name

    @_best_metric_name.setter
    def _best_metric_name(self, value: str) -> None:
        self._ensure_state().best_metric_name = str(value)

    def train(self) -> ProteinPretrainResult:
        mode = self._resolve_mode()
        self._log(f"🚀 Training mode: {mode}")
        if mode == "resume":
            return self._run_resume()
        return self._run_from_scratch()

    def _resolve_mode(self) -> str:
        mode_name = self._mode_cfg["name"]
        if mode_name == "auto":
            checkpoint_path = self._resume_cfg["checkpoint_path"]
            resume_state_path = self._paths.get("resume_state_path") or self._resume_cfg.get("resume_state_path")
            if checkpoint_path.exists():
                return "resume"
            if resume_state_path and Path(resume_state_path).exists():
                return "resume"
            return "train_from_scratch"
        if mode_name == "resume" and self._mode_cfg["resume_if_available"]:
            if not self._resume_cfg["checkpoint_path"].exists():
                return "train_from_scratch"
        return mode_name

    def _run_from_scratch(self) -> ProteinPretrainResult:
        self._log("📦 [1/6] Loading tokenizer...")
        self._setup_tokenizer()
        self._log(f"✅ Tokenizer loaded — vocab_size={self.tokenizer.vocab_size}")

        self._log("🧠 [2/6] Building model...")
        self._setup_model_from_config()
        param_count = sum(p.numel() for p in self.model.parameters())
        self._log(f"✅ Model built — {param_count:,} parameters")

        self._log("⚙️  [3/6] Preparing runtime...")
        self._setup_runtime()
        self._log(f"✅ Device: {self.runtime.device} | Distributed: {self.runtime.distributed}")

        self._log("🔧 [4/6] Creating optimizer...")
        self._setup_optimizer()
        self._log("✅ Optimizer ready")

        self._save_config_snapshot()

        # Preflight VRAM check
        self._run_preflight_if_enabled()

        self._log("📂 [5/6] Building data loaders...")
        loaders = self._build_data_loaders()
        self._log("✅ Data loaders ready")

        self._init_resume_state("train_from_scratch")

        self._log("🏋️ [6/6] Starting training loop...")
        self._training_loop(loaders)
        self._log("🎉 Training complete!")
        return self._build_result()

    def _run_resume(self) -> ProteinPretrainResult:
        checkpoint_path = self._resume_cfg["checkpoint_path"]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing resume checkpoint: {checkpoint_path}")

        self._log(f"📥 [1/6] Loading checkpoint: {checkpoint_path.name}")
        checkpoint_meta = torch.load(checkpoint_path, map_location="cpu")
        if not is_supported_protein_checkpoint_family(checkpoint_meta.get("model_family")):
            raise ValueError(
                f"Unsupported checkpoint family: {checkpoint_meta.get('model_family')!r}"
            )

        self._log("📦 [2/6] Restoring tokenizer from checkpoint...")
        self._restore_tokenizer_from_checkpoint(checkpoint_meta)
        self._log(f"✅ Tokenizer restored — vocab_size={self.tokenizer.vocab_size}")

        self._log("🧠 [3/6] Restoring model from checkpoint...")
        self._restore_model_from_checkpoint(checkpoint_meta)
        self._log("✅ Model restored")

        self._log("⚙️  [4/6] Preparing runtime...")
        self._setup_runtime()
        self._log(f"✅ Device: {self.runtime.device}")

        self._log("🔧 [5/6] Creating optimizer & restoring state...")
        self._setup_optimizer()

        load_protein_pretrain_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer if self._resume_cfg["restore_optimizer_state"] else None,
            map_location=self.runtime.device,
        )

        if self._resume_cfg["override_optimizer_hyperparameters"]:
            apply_protein_training_optimizer_settings(self.optimizer, self._optimizer_cfg)

        self._epoch = int(checkpoint_meta.get("epoch", 0))
        self._global_step = int(checkpoint_meta.get("global_step", 0))
        self._tokens_seen = int(checkpoint_meta.get("tokens_seen", 0))
        self._train_losses = list(checkpoint_meta.get("train_losses", []))
        self._val_losses = list(checkpoint_meta.get("val_losses", []))
        best_val = checkpoint_meta.get("best_val_loss")
        self._best_val_loss = _restore_best_val_loss(checkpoint_meta, best_val)
        self._best_metric_name = "val_loss"
        self._log(f"✅ Resumed from epoch={self._epoch} step={self._global_step} tokens={self._tokens_seen:,}")

        self._save_config_snapshot()

        # Preflight VRAM check
        self._run_preflight_if_enabled()

        self._log("📂 [6/6] Building data loaders...")
        loaders = self._build_data_loaders()
        self._log("✅ Data loaders ready")
        self._init_resume_state("resume")

        self._log("🏋️ Resuming training loop...")
        self._training_loop(loaders)
        self._log("🎉 Training complete!")
        return self._build_result()

    def _setup_tokenizer(self) -> None:
        train_text_path = self._paths["train_text_path"]
        tokenizer_map_path = self._paths["tokenizer_map_path"]
        vocab_size = self._model_cfg["tokenizer_vocab_size"]
        rebuild = self._model_cfg["rebuild_tokenizer"]

        local_train_paths = self._discover_local_paths()
        if local_train_paths:
            artifact = build_or_load_protein_tokenizer_from_text_paths(
                local_train_paths,
                tokenizer_map_path=tokenizer_map_path,
                vocab_size=vocab_size,
                rebuild=rebuild,
            )
        elif tokenizer_map_path.exists():
            self.tokenizer = SequenceTokenizer.load_map(tokenizer_map_path)
            return
        else:
            raise FileNotFoundError(
                f"Neither corpus at {train_text_path} nor tokenizer at {tokenizer_map_path} found."
            )
        self.tokenizer = artifact.tokenizer

    def _restore_tokenizer_from_checkpoint(self, checkpoint_meta: dict[str, Any]) -> None:
        tokenizer_map_path = Path(checkpoint_meta["tokenizer_map_path"])
        if not tokenizer_map_path.exists():
            tokenizer_map_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer_map_path.write_text(
                json.dumps(checkpoint_meta["tokenizer_map"], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        self.tokenizer = SequenceTokenizer.load_map(tokenizer_map_path)

    def _setup_model_from_config(self) -> None:
        mixed_precision = self._runtime_cfg.get("mixed_precision", "auto")
        model_dtype = self._resolve_model_dtype(mixed_precision)
        progen_config = build_progen_config(
            self._model_cfg["progen_model_size"],
            vocab_size=self.tokenizer.vocab_size,
            context_length=self._model_cfg["context_length"],
            dtype=model_dtype,
        )
        overrides = self._model_cfg["progen_config_overrides"]
        if overrides:
            progen_config = {**progen_config, **overrides}
        self.model_config = build_mdc_config_from_progen_config(progen_config, dtype=model_dtype)
        self.model = MDCDecoderModel(self.model_config)

    def _resolve_model_dtype(self, mixed_precision: str) -> torch.dtype:
        if mixed_precision == "bf16":
            return torch.bfloat16
        if mixed_precision == "fp16":
            # FP16 AMP requires FP32 master weights; autocast handles FP16 compute.
            # GradScaler cannot unscale FP16 gradients.
            return torch.float32
        if mixed_precision == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            if torch.cuda.is_available():
                # Same as explicit fp16: keep model in fp32 for GradScaler compatibility.
                return torch.float32
        return torch.float32

    def _restore_model_from_checkpoint(self, checkpoint_meta: dict[str, Any]) -> None:
        model_config_payload = dict(checkpoint_meta["model_config"])
        if model_config_payload.get("layer_types") is not None:
            model_config_payload["layer_types"] = tuple(model_config_payload["layer_types"])
        self.model_config = MDCModelConfig(**model_config_payload)
        self.model = MDCDecoderModel(self.model_config)

    def _setup_runtime(self) -> None:
        requested_device = self._runtime_cfg["device"]
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.runtime = prepare_mdc_training_runtime(
            self.model,
            device=requested_device,
            multi_gpu=self._runtime_cfg["multi_gpu_mode"],
            find_unused_parameters=self._runtime_cfg["ddp_find_unused_parameters"],
            data_parallel_device_ids=self._runtime_cfg["data_parallel_device_ids"],
        )
        self.model = self.runtime.model

    def _setup_optimizer(self) -> None:
        self.optimizer = create_protein_training_optimizer(
            self.model,
            self._optimizer_cfg,
            device=self.runtime.device,
        )

    def _run_preflight_if_enabled(self) -> None:
        """Run VRAM preflight check if runtime.preflight_vram_check is enabled."""
        if not self._runtime_cfg.get("preflight_vram_check", False):
            return
        if not self.is_main_process:
            return

        target_vram_gb = float(self._runtime_cfg.get("target_vram_gb", 14.0))
        batch_size = self._data_cfg["batch_size"]
        context_length = int(self.model_config.context_length)
        mixed_precision = self._runtime_cfg.get("mixed_precision", "auto")
        gradient_accumulation_steps = self._training_cfg.get("gradient_accumulation_steps", 1)

        report_path = self._paths["checkpoint_dir"] / "vram_report.json"

        self._log(f"🔍 Running VRAM preflight check (target={target_vram_gb:.1f}GB)...")

        # This will raise RuntimeError if config doesn't fit
        result = run_preflight_vram_check(
            model=self.model,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            batch_size=batch_size,
            context_length=context_length,
            device=self.runtime.device,
            target_vram_gb=target_vram_gb,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            report_output_path=report_path,
        )

        peak = result.get("peak_allocated_gb")
        if peak is not None:
            self._log(f"✅ Preflight passed — peak={peak:.2f}GB / target={target_vram_gb:.1f}GB")
        else:
            self._log("✅ Preflight skipped (no CUDA measurement available)")

    def _discover_local_paths(self) -> tuple[Path, ...]:
        train_text_path = self._paths["train_text_path"]
        train_part_cache_dir = self._paths["train_part_cache_dir"]
        part_glob = self._data_cfg["train_part_glob"]
        prefer_parts = self._data_cfg["prefer_local_train_parts"]
        cached_part_paths = tuple(sorted(train_part_cache_dir.glob(part_glob), key=_natural_path_sort_key))

        if prefer_parts and cached_part_paths:
            return cached_part_paths
        if train_text_path.exists():
            return (train_text_path,)
        if cached_part_paths:
            return cached_part_paths

        try:
            return discover_protein_train_text_paths(
                train_text_path,
                part_glob=part_glob,
                prefer_parts=prefer_parts,
            )
        except FileNotFoundError:
            return ()

    def _build_data_loaders(self) -> LoaderBundle:
        factory = DataLoaderFactory(
            tokenizer=self.tokenizer,
            model_config=self.model_config,
            runtime=self.runtime,
            paths=self._paths,
            data_cfg=self._data_cfg,
            model_cfg=self._model_cfg,
            minio_cfg=self._minio_cfg,
            minio_data_config=self.minio_data_config,
            local_paths_provider=self._discover_local_paths,
            is_main_process=self.is_main_process,
        )
        return factory.build()

    def _training_loop(self, loaders: LoaderBundle) -> None:
        train_loader = loaders.train_loader
        train_eval_loader = loaders.train_eval_loader
        val_loader = loaders.val_loader
        device = self.runtime.device
        settings = TrainingLoopSettings.from_config(self._training_cfg)
        optimizers = optimizer_list(self.optimizer)
        step_eval_enabled = self._step_eval_enabled(settings, loaders)
        self._validate_best_checkpoint_settings(settings, loaders)
        start_epoch = self._epoch

        precision = resolve_precision_context(
            str(self._runtime_cfg.get("mixed_precision", "auto")),
            device,
        )
        self._configure_linear_attention_precision()

        accumulator = GradientAccumulator(settings.gradient_accumulation_steps)
        _last_loss = float("nan")

        for epoch_offset in range(1, settings.num_epochs + 1):
            epoch = start_epoch + epoch_offset
            self._epoch = epoch
            set_mdc_data_loader_epoch(train_loader, epoch - 1)
            self.model.train()

            max_steps_str = f"/{settings.max_steps}" if settings.max_steps else ""
            self._log(f"📈 Epoch {epoch}/{start_epoch + settings.num_epochs} | step={self._global_step}{max_steps_str} | tokens={self._tokens_seen:,}")

            zero_grad(optimizers)

            for batch in train_loader:
                try:
                    batch = self._move_batch(batch, device)
                    accumulator.next_micro_step()
                    loss = self._forward_loss(batch, precision)
                    precision.backward(accumulator.scale_loss(loss))

                    self._tokens_seen += self._count_step_tokens(batch)

                    # Optimizer step at accumulation boundary
                    if accumulator.at_boundary:
                        self._optimizer_step(optimizers, precision, settings)
                        self._global_step += 1
                        _last_loss = float(loss.item())

                        if self._global_step % settings.log_every_steps == 0:
                            self._log(
                                f"  🔄 step={self._global_step} | loss={_last_loss:.4f} | tokens={self._tokens_seen:,}"
                            )

                        if self._should_run_step_eval(settings, step_eval_enabled=step_eval_enabled):
                            self._run_eval(
                                train_eval_loader,
                                val_loader,
                                settings.eval_batches,
                                settings.save_best,
                                autocast_dtype=precision.autocast_dtype,
                            )

                        if self._should_save_step_checkpoint(settings):
                            if self.is_main_process and settings.save_last:
                                self._log(f"  💾 Saving checkpoint at step {self._global_step}...")
                                self._save_last_checkpoint()
                                self._save_resume_state()

                        if settings.max_steps is not None and self._global_step >= settings.max_steps:
                            break

                except torch.cuda.OutOfMemoryError:
                    self._handle_oom(batch_size=self._data_cfg["batch_size"], context_length=int(self.model_config.context_length))

            # Preserve the existing absolute micro-step cadence across epochs.
            if accumulator.has_leftover:
                self._optimizer_step(optimizers, precision, settings)
                self._global_step += 1

            self._distributed_barrier()

            if self.is_main_process:
                self._log(f"  📊 Epoch {epoch} end — evaluating...")
                self._run_epoch_end_eval(
                    train_eval_loader,
                    val_loader,
                    settings.eval_batches,
                    settings.save_best,
                    autocast_dtype=precision.autocast_dtype,
                )
                if settings.save_last:
                    self._log("  💾 Saving epoch checkpoint...")
                    self._save_last_checkpoint()
                self._save_resume_state()

            self._distributed_barrier()

            if settings.max_steps is not None and self._global_step >= settings.max_steps:
                self._log(f"⏹️  Reached max_steps={settings.max_steps}")
                break

        if self.is_main_process and settings.save_final:
            self._log("💾 Saving final checkpoint...")
            self._save_final_checkpoint()
            self._save_resume_state()

    def _forward_loss(self, batch: CausalLMBatch, precision: PrecisionContext) -> torch.Tensor:
        with precision.autocast():
            logits = self.model(batch.input_ids, attn_mask=batch.attention_mask)
            return compute_mdc_causal_lm_loss(logits, batch.labels)

    def _optimizer_step(
        self,
        optimizers: list[torch.optim.Optimizer],
        precision: PrecisionContext,
        settings: TrainingLoopSettings,
    ) -> None:
        step_optimizers(
            model=self.model,
            optimizers=optimizers,
            precision=precision,
            grad_clip_norm=settings.grad_clip_norm,
        )

    def _step_eval_enabled(self, settings: TrainingLoopSettings, loaders: LoaderBundle) -> bool:
        return settings.eval_freq > 0 and not self.runtime.distributed and loaders.train_eval_loader is not None

    def _should_run_step_eval(
        self,
        settings: TrainingLoopSettings,
        *,
        step_eval_enabled: bool,
    ) -> bool:
        return step_eval_enabled and self._global_step % settings.eval_freq == 0

    def _should_save_step_checkpoint(self, settings: TrainingLoopSettings) -> bool:
        return (
            self.is_main_process
            and settings.save_last
            and bool(settings.save_every_steps)
            and self._global_step % settings.save_every_steps == 0
        )

    def _validate_best_checkpoint_settings(
        self,
        settings: TrainingLoopSettings,
        loaders: LoaderBundle,
    ) -> None:
        if self.is_main_process and settings.save_best and loaders.val_loader is None:
            raise ValueError(
                "save_best=true but val_loader is None. "
                "No fallback to train loss is allowed. "
                "Configure a validation split with data.train_ratio and data.split_seed."
            )

    def _configure_linear_attention_precision(self) -> None:
        # Configure linear attention fallback precision.
        import libs.core.mdc.linear_attention as _la_module

        _la_module.use_fp32_fallback_linear_attention = self._runtime_cfg.get(
            "use_fp32_fallback_linear_attention", True
        )

    def _resolve_autocast_dtype(self, mixed_precision: str, device: torch.device) -> torch.dtype | None:
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

    def _run_eval(self, train_eval_loader, val_loader, eval_batches: int, save_best: bool, autocast_dtype: torch.dtype | None = None) -> None:
        device = self.runtime.device
        train_eval_loss, val_loss = self._evaluator.evaluate(
            self.model, train_eval_loader, val_loader, device=device, max_batches=eval_batches, autocast_dtype=autocast_dtype,
        )
        self._train_losses.append(train_eval_loss)
        self._val_losses.append(val_loss)
        self._append_metrics(train_eval_loss, val_loss)

        improved = self._maybe_save_best(val_loss, has_validation_loader=val_loader is not None, save_best=save_best)

        if self.is_main_process:
            best_marker = " 🏆 new best val_loss!" if improved else ""
            print(
                "  📊 eval "
                f"step={self._global_step} | "
                f"train={_format_eval_loss(train_eval_loss)} | "
                f"val={_format_eval_loss(val_loss, available=val_loader is not None)}"
                f"{best_marker}",
                flush=True,
            )

    def _run_epoch_end_eval(self, train_eval_loader, val_loader, eval_batches: int, save_best: bool, autocast_dtype: torch.dtype | None = None) -> None:
        device = self.runtime.device
        train_eval_loss, val_loss = self._evaluator.evaluate(
            self.model, train_eval_loader, val_loader, device=device, max_batches=eval_batches, autocast_dtype=autocast_dtype,
        )
        self._train_losses.append(train_eval_loss)
        self._val_losses.append(val_loss)
        self._append_metrics(train_eval_loss, val_loss)

        self._maybe_save_best(val_loss, has_validation_loader=val_loader is not None, save_best=save_best)

        self._log(
            f"  ✅ Epoch {self._epoch} done | "
            f"train={_format_eval_loss(train_eval_loss)} | "
            f"val={_format_eval_loss(val_loss, available=val_loader is not None)}"
        )

    def _maybe_save_best(
        self,
        val_loss: float,
        *,
        has_validation_loader: bool,
        save_best: bool,
    ) -> bool:
        improved = save_best and has_validation_loader and _is_finite_loss(val_loss) and val_loss < self._best_val_loss
        if improved:
            self._best_val_loss = val_loss
            self._best_metric_name = "val_loss"
            self._save_best_checkpoint()
        return improved

    def _handle_oom(self, batch_size: int, context_length: int) -> None:
        """Handle CUDA OOM during training: cleanup, save state, raise with guidance."""
        import gc as _gc
        torch.cuda.empty_cache()
        _gc.collect()

        param_count = sum(p.numel() for p in self.model.parameters())
        peak_gb = None
        if torch.cuda.is_available():
            try:
                peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            except Exception:
                pass

        # Try to save emergency resume state
        if self.is_main_process:
            try:
                self._save_resume_state()
                self._log("💾 Emergency resume state saved.")
            except Exception:
                pass

        suggested_fixes = [
            "1. Set batch_size=1",
            "2. Reduce context_length to 512 or 384",
            "3. Set eval_batches=1",
            "4. Increase eval_freq to reduce evaluation frequency",
            "5. Use the 16GB-optimized config: config/train.16gb.yaml",
        ]
        peak_str = f"{peak_gb:.2f}" if peak_gb is not None else "unknown"
        msg = (
            f"\n{'='*60}\n"
            f"CUDA OUT OF MEMORY during training\n"
            f"{'='*60}\n"
            f"  batch_size={batch_size}\n"
            f"  context_length={context_length}\n"
            f"  model_params={param_count:,}\n"
            f"  peak_memory_gb={peak_str}\n"
            f"\nSuggested fixes:\n"
        )
        for fix in suggested_fixes:
            msg += f"  {fix}\n"
        msg += f"{'='*60}\n"
        raise RuntimeError(msg)

    def _save_last_checkpoint(self) -> Path:
        return self._do_save_checkpoint(self._resume_cfg["output_checkpoint_path"])

    def _save_best_checkpoint(self) -> Path:
        return self._do_save_checkpoint(self._resume_cfg["best_checkpoint_path"])

    def _save_final_checkpoint(self) -> Path:
        return self._do_save_checkpoint(self._resume_cfg["final_checkpoint_path"])

    def _do_save_checkpoint(self, path: Path) -> Path:
        return self._checkpoint_service.save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            model_config=self.model_config,
            tokenizer=self.tokenizer,
            epoch=self._epoch,
            global_step=self._global_step,
            tokens_seen=self._tokens_seen,
            train_losses=self._train_losses,
            val_losses=self._val_losses,
            best_val_loss=None if math.isinf(self._best_val_loss) else self._best_val_loss,
            best_metric_name=self._best_metric_name,
            local_paths=self._discover_local_paths(),
        )

    def _init_resume_state(self, mode: str) -> None:
        if not self.is_main_process:
            return
        resume_state_path = self._paths.get("resume_state_path") or self._resume_cfg.get("resume_state_path")
        if not resume_state_path:
            return

        existing = load_resume_state(resume_state_path)
        if existing and mode == "resume":
            self._resume_state = existing
            update_resume_state_progress(
                self._resume_state,
                epoch=self._epoch,
                global_step=self._global_step,
                tokens_seen=self._tokens_seen,
            )
        else:
            self._resume_state = create_resume_state(
                mode=mode,
                config_path=str(self.config["config_path"]),
                training_config_snapshot_path=str(self._paths.get("training_config_snapshot_path", "")),
                checkpoint_path=str(self._resume_cfg["best_checkpoint_path"]),
                best_checkpoint_path=str(self._resume_cfg["best_checkpoint_path"]),
                final_checkpoint_path=str(self._resume_cfg["final_checkpoint_path"]),
                model_info={
                    "progen_model_size": self._model_cfg["progen_model_size"],
                    "context_length": self._model_cfg["context_length"],
                    "stride": self._model_cfg["stride"],
                    "progen_config_overrides": self._model_cfg["progen_config_overrides"],
                },
                optimizer_info={
                    "type": self._optimizer_cfg["type"],
                    "learning_rate": self._optimizer_cfg["learning_rate"],
                    "weight_decay": self._optimizer_cfg["weight_decay"],
                },
                runtime_info={
                    "device": str(self.runtime.device),
                    "distributed": self.runtime.distributed,
                    "data_parallel": self.runtime.data_parallel,
                    "rank": self.runtime.rank,
                    "world_size": self.runtime.world_size,
                    "mixed_precision": self._runtime_cfg.get("mixed_precision", "auto"),
                },
            )
        save_resume_state(self._resume_state, resume_state_path)

    def _save_resume_state(self) -> None:
        if not self.is_main_process or self._resume_state is None:
            return
        resume_state_path = self._paths.get("resume_state_path") or self._resume_cfg.get("resume_state_path")
        if not resume_state_path:
            return

        update_resume_state_progress(
            self._resume_state,
            epoch=self._epoch,
            global_step=self._global_step,
            tokens_seen=self._tokens_seen,
        )
        best_loss = None if math.isinf(self._best_val_loss) else self._best_val_loss
        update_resume_state_metrics(
            self._resume_state,
            best_loss=best_loss,
            best_metric_name=self._best_metric_name if best_loss is not None else None,
        )
        save_resume_state(self._resume_state, resume_state_path)

    def _save_config_snapshot(self) -> None:
        if not self.is_main_process:
            return
        snapshot_path = self._paths.get("training_config_snapshot_path")
        if not snapshot_path:
            return
        snapshot_path = Path(snapshot_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        snapshot = {
            "mode": self._mode_cfg,
            "model": self._model_cfg,
            "optimizer": self._optimizer_cfg,
            "training": self._training_cfg,
            "runtime": self._runtime_cfg,
            "data": {k: v for k, v in self._data_cfg.items()},
            "paths": {k: str(v) for k, v in self._paths.items()},
        }
        snapshot_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def _append_metrics(self, train_loss: float, val_loss: float) -> None:
        if not self.is_main_process:
            return
        self._metrics_writer.append(
            epoch=self._epoch,
            global_step=self._global_step,
            tokens_seen=self._tokens_seen,
            train_loss=train_loss,
            val_loss=val_loss,
        )

        if self._resume_state is not None:
            update_resume_state_metrics(
                self._resume_state,
                train_loss=train_loss,
                val_loss=val_loss if _is_finite_loss(val_loss) else None,
            )

    def _move_batch(self, batch: CausalLMBatch, device) -> CausalLMBatch:
        return CausalLMBatch(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            labels=batch.labels.to(device),
        )

    def _count_step_tokens(self, batch: CausalLMBatch) -> int:
        token_count = torch.tensor(
            int(batch.attention_mask.sum().item()),
            device=self.runtime.device,
            dtype=torch.long,
        )
        if self.runtime.distributed:
            torch.distributed.all_reduce(token_count, op=torch.distributed.ReduceOp.SUM)
        return int(token_count.item())

    def _distributed_barrier(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _build_result(self) -> ProteinPretrainResult:
        final_train = self._train_losses[-1] if self._train_losses else None
        final_val = self._val_losses[-1] if self._val_losses else None
        best = None if math.isinf(self._best_val_loss) else self._best_val_loss
        best_checkpoint_path = self._resume_cfg["best_checkpoint_path"]
        resume_state_path = self._paths.get("resume_state_path") or self._resume_cfg.get("resume_state_path")
        return ProteinPretrainResult(
            checkpoint_path=best_checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            final_checkpoint_path=self._resume_cfg["final_checkpoint_path"],
            resume_state_path=Path(resume_state_path) if resume_state_path else None,
            global_step=self._global_step,
            tokens_seen=self._tokens_seen,
            epochs_completed=self._epoch,
            best_loss=best,
            final_train_loss=final_train,
            final_val_loss=final_val,
        )

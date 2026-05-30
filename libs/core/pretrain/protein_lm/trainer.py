from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from libs.core.interfaces import CausalLMBatch
from libs.core.mdc.config import MDCModelConfig
from libs.core.mdc import MDCDecoderModel
from libs.core.pretrain.distributed import (
    MDCTrainingRuntime,
    prepare_mdc_training_runtime,
    set_mdc_data_loader_epoch,
    unwrap_mdc_training_model,
    cleanup_mdc_distributed_training,
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
from libs.core.pretrain.protein_lm.core import (
    build_or_load_protein_tokenizer,
    build_or_load_protein_tokenizer_from_text_paths,
    count_trainable_parameters,
    create_protein_lm_dataloader,
    create_streaming_protein_lm_dataloader,
    discover_protein_train_text_paths,
    generate_protein_text,
    load_protein_corpus_text_parts,
    load_protein_pretrain_checkpoint,
    split_protein_corpus_text,
)
from libs.core.pretrain.protein_lm.support.backbone import (
    build_mdc_config_from_progen_config,
    build_progen_config,
    extract_protein_backbone_config,
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
from libs.core.pretrain.protein_lm.services import CheckpointService, Evaluator, MetricsWriter
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


class ProteinPretrainTrainer:
    def __init__(
        self,
        project_root: Path | str,
        config_path: Path | str | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.config = load_protein_training_config(self.project_root, config_path=config_path)
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

        self.runtime: MDCTrainingRuntime | None = None
        self.model: torch.nn.Module | None = None
        self.optimizer: Any = None
        self.tokenizer: SequenceTokenizer | None = None
        self.model_config: MDCModelConfig | None = None

        self._global_step = 0
        self._tokens_seen = 0
        self._epoch = 0
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._best_val_loss = math.inf
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
        train_loader, train_eval_loader, val_loader = self._build_data_loaders()
        self._log("✅ Data loaders ready")

        self._init_resume_state("train_from_scratch")

        self._log("🏋️ [6/6] Starting training loop...")
        self._training_loop(train_loader, train_eval_loader, val_loader)
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
        self._best_val_loss = math.inf if best_val is None else float(best_val)
        self._log(f"✅ Resumed from epoch={self._epoch} step={self._global_step} tokens={self._tokens_seen:,}")

        self._save_config_snapshot()

        # Preflight VRAM check
        self._run_preflight_if_enabled()

        self._log("📂 [6/6] Building data loaders...")
        train_loader, train_eval_loader, val_loader = self._build_data_loaders()
        self._log("✅ Data loaders ready")
        self._init_resume_state("resume")

        self._log("🏋️ Resuming training loop...")
        self._training_loop(train_loader, train_eval_loader, val_loader)
        self._log("🎉 Training complete!")
        return self._build_result()

    def _setup_tokenizer(self) -> None:
        train_text_path = self._paths["train_text_path"]
        tokenizer_map_path = self._paths["tokenizer_map_path"]
        vocab_size = self._model_cfg["tokenizer_vocab_size"]
        rebuild = self._model_cfg["rebuild_tokenizer"]

        local_train_paths = self._discover_local_paths()
        if local_train_paths and len(local_train_paths) > 1:
            artifact = build_or_load_protein_tokenizer_from_text_paths(
                local_train_paths,
                tokenizer_map_path=tokenizer_map_path,
                vocab_size=vocab_size,
                rebuild=rebuild,
            )
        elif train_text_path.exists():
            artifact = build_or_load_protein_tokenizer(
                train_text_path,
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
            self._log(f"✅ Preflight skipped (no CUDA measurement available)")

    def _discover_local_paths(self) -> tuple[Path, ...]:
        train_text_path = self._paths["train_text_path"]
        part_glob = self._data_cfg["train_part_glob"]
        prefer_parts = self._data_cfg["prefer_local_train_parts"]
        try:
            return discover_protein_train_text_paths(
                train_text_path,
                part_glob=part_glob,
                prefer_parts=prefer_parts,
            )
        except FileNotFoundError:
            return ()

    def _build_data_loaders(self):
        context_length = int(self.model_config.context_length)
        stride = self._model_cfg["stride"] or max(1, context_length // 2)
        batch_size = self._data_cfg["batch_size"]
        num_workers = self._data_cfg["num_workers"]
        pin_memory = self._data_cfg["pin_memory"]
        rank = self.runtime.rank
        world_size = self.runtime.world_size
        distributed = self.runtime.distributed

        loader_kwargs = {
            "context_length": context_length,
            "stride": stride,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        minio_prefix = self._minio_cfg["train_parts_prefix_uri"]
        minio_uris = self._minio_cfg["train_part_uris"]
        use_minio = bool(minio_prefix or minio_uris)
        local_paths = self._discover_local_paths()
        stream_local = self._data_cfg["stream_local_train_parts"]
        use_streaming = use_minio or (stream_local and len(local_paths) > 1)

        shuffle_parts = self._data_cfg["shuffle_parts"]
        shuffle_examples = self._data_cfg["shuffle_examples"]
        shuffle_buffer_size = self._data_cfg["shuffle_buffer_size"]
        keep_parts = self._data_cfg["keep_downloaded_train_parts"]
        cache_dir = self._paths["train_part_cache_dir"]

        if use_minio:
            train_loader = create_streaming_protein_lm_dataloader(
                self.tokenizer,
                prefix_uri=minio_prefix or None,
                part_uris=minio_uris or None,
                config=self.minio_data_config,
                cache_dir=cache_dir,
                keep_downloaded_parts=keep_parts,
                shuffle_parts=shuffle_parts,
                shuffle_examples=shuffle_examples,
                shuffle_buffer_size=shuffle_buffer_size,
                seed=rank,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                **loader_kwargs,
            )
            train_eval_loader = self._build_eval_streaming_loader(loader_kwargs) if self.is_main_process else None
        elif use_streaming:
            train_loader = create_streaming_protein_lm_dataloader(
                self.tokenizer,
                part_paths=local_paths,
                shuffle_parts=shuffle_parts,
                shuffle_examples=shuffle_examples,
                shuffle_buffer_size=shuffle_buffer_size,
                seed=rank,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                **loader_kwargs,
            )
            train_eval_loader = (
                create_streaming_protein_lm_dataloader(
                    self.tokenizer,
                    part_paths=local_paths,
                    shuffle_parts=False,
                    shuffle_examples=False,
                    seed=0,
                    distributed=False,
                    **loader_kwargs,
                )
                if self.is_main_process
                else None
            )
        else:
            corpus_text = load_protein_corpus_text_parts(local_paths) if local_paths else ""
            if not corpus_text:
                raise ValueError("No local corpus or MinIO parts configured.")
            train_text, val_text = split_protein_corpus_text(corpus_text, train_ratio=self._data_cfg["train_ratio"])
            train_loader = create_protein_lm_dataloader(
                train_text,
                self.tokenizer,
                shuffle=True,
                sampler_seed=0,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                **loader_kwargs,
            )
            train_eval_loader = (
                create_protein_lm_dataloader(train_text, self.tokenizer, shuffle=False, distributed=False, **loader_kwargs)
                if self.is_main_process
                else None
            )
            val_loader = (
                create_protein_lm_dataloader(val_text, self.tokenizer, shuffle=False, distributed=False, **loader_kwargs)
                if val_text and self.is_main_process
                else None
            )
            return train_loader, train_eval_loader, val_loader

        val_loader = None
        return train_loader, train_eval_loader, val_loader

    def _build_eval_streaming_loader(self, loader_kwargs: dict) -> Any:
        minio_prefix = self._minio_cfg["train_parts_prefix_uri"]
        minio_uris = self._minio_cfg["train_part_uris"]
        cache_dir = self._paths["train_part_cache_dir"]
        keep_parts = self._data_cfg["keep_downloaded_train_parts"]
        return create_streaming_protein_lm_dataloader(
            self.tokenizer,
            prefix_uri=minio_prefix or None,
            part_uris=minio_uris or None,
            config=self.minio_data_config,
            cache_dir=cache_dir,
            keep_downloaded_parts=keep_parts,
            shuffle_parts=False,
            shuffle_examples=False,
            seed=0,
            distributed=False,
            **loader_kwargs,
        )

    def _training_loop(self, train_loader, train_eval_loader, val_loader) -> None:
        device = self.runtime.device
        num_epochs = self._training_cfg["num_epochs"]
        max_steps = self._training_cfg.get("max_steps")
        eval_freq = self._training_cfg["eval_freq"]
        eval_batches = self._training_cfg["eval_batches"]
        grad_clip_norm = self._training_cfg["grad_clip_norm"]
        save_every_steps = self._training_cfg.get("save_every_steps")
        save_best = self._training_cfg.get("save_best", True)
        save_last = self._training_cfg.get("save_last", True)
        gradient_accumulation_steps = self._training_cfg.get("gradient_accumulation_steps", 1)
        log_every_steps = max(1, eval_freq // 2) if eval_freq > 0 else 50

        optimizer_list = list(self.optimizer) if isinstance(self.optimizer, (list, tuple)) else [self.optimizer]
        step_eval_enabled = eval_freq > 0 and not self.runtime.distributed and train_eval_loader is not None
        start_epoch = self._epoch

        # Mixed precision setup
        mixed_precision = self._runtime_cfg.get("mixed_precision", "auto")
        autocast_dtype = self._resolve_autocast_dtype(mixed_precision, device)
        use_autocast = autocast_dtype is not None and device.type == "cuda"
        use_grad_scaler = use_autocast and autocast_dtype == torch.float16
        grad_scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None

        # Configure linear attention fallback precision
        import libs.core.mdc.linear_attention as _la_module
        _la_module.use_fp32_fallback_linear_attention = self._runtime_cfg.get(
            "use_fp32_fallback_linear_attention", True
        )

        micro_step = 0
        _last_loss = float("nan")

        for epoch_offset in range(1, num_epochs + 1):
            epoch = start_epoch + epoch_offset
            self._epoch = epoch
            set_mdc_data_loader_epoch(train_loader, epoch - 1)
            self.model.train()

            max_steps_str = f"/{max_steps}" if max_steps else ""
            self._log(f"📈 Epoch {epoch}/{start_epoch + num_epochs} | step={self._global_step}{max_steps_str} | tokens={self._tokens_seen:,}")

            for opt in optimizer_list:
                opt.zero_grad(set_to_none=True)

            for batch in train_loader:
                try:
                    batch = self._move_batch(batch, device)
                    micro_step += 1

                    # Forward with optional autocast
                    if use_autocast:
                        with torch.amp.autocast("cuda", dtype=autocast_dtype):
                            logits = self.model(batch.input_ids, attn_mask=batch.attention_mask)
                            loss = compute_mdc_causal_lm_loss(logits, batch.labels)
                    else:
                        logits = self.model(batch.input_ids, attn_mask=batch.attention_mask)
                        loss = compute_mdc_causal_lm_loss(logits, batch.labels)

                    # Scale loss for gradient accumulation
                    scaled_loss = loss / gradient_accumulation_steps

                    if grad_scaler is not None:
                        grad_scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    self._tokens_seen += self._count_step_tokens(batch)

                    # Optimizer step at accumulation boundary
                    if micro_step % gradient_accumulation_steps == 0:
                        if grad_clip_norm is not None:
                            if grad_scaler is not None:
                                for opt in optimizer_list:
                                    grad_scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)

                        if grad_scaler is not None:
                            for opt in optimizer_list:
                                grad_scaler.step(opt)
                            grad_scaler.update()
                        else:
                            for opt in optimizer_list:
                                opt.step()

                        for opt in optimizer_list:
                            opt.zero_grad(set_to_none=True)

                        self._global_step += 1
                        _last_loss = float(loss.item())

                        if self._global_step % log_every_steps == 0:
                            self._log(
                                f"  🔄 step={self._global_step} | loss={_last_loss:.4f} | tokens={self._tokens_seen:,}"
                            )

                        if step_eval_enabled and self._global_step % eval_freq == 0:
                            self._run_eval(train_eval_loader, val_loader, eval_batches, save_best, autocast_dtype=autocast_dtype)

                        if save_every_steps and self._global_step % save_every_steps == 0:
                            if self.is_main_process and save_last:
                                self._log(f"  💾 Saving checkpoint at step {self._global_step}...")
                                self._save_last_checkpoint()
                                self._save_resume_state()

                        if max_steps is not None and self._global_step >= max_steps:
                            break

                except torch.cuda.OutOfMemoryError:
                    self._handle_oom(batch_size=self._data_cfg["batch_size"], context_length=int(self.model_config.context_length))

            # Handle leftover microbatches at end of epoch
            if micro_step % gradient_accumulation_steps != 0:
                if grad_clip_norm is not None:
                    if grad_scaler is not None:
                        for opt in optimizer_list:
                            grad_scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                if grad_scaler is not None:
                    for opt in optimizer_list:
                        grad_scaler.step(opt)
                    grad_scaler.update()
                else:
                    for opt in optimizer_list:
                        opt.step()
                for opt in optimizer_list:
                    opt.zero_grad(set_to_none=True)
                self._global_step += 1

            self._distributed_barrier()

            if self.is_main_process:
                self._log(f"  📊 Epoch {epoch} end — evaluating...")
                self._run_epoch_end_eval(train_eval_loader, val_loader, eval_batches, save_best, autocast_dtype=autocast_dtype)
                if save_last:
                    self._log(f"  💾 Saving epoch checkpoint...")
                    self._save_last_checkpoint()
                self._save_resume_state()

            self._distributed_barrier()

            if max_steps is not None and self._global_step >= max_steps:
                self._log(f"⏹️  Reached max_steps={max_steps}")
                break

        if self.is_main_process and self._training_cfg.get("save_final", True):
            self._log("💾 Saving final checkpoint...")
            self._save_final_checkpoint()
            self._save_resume_state()

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

        metric = val_loss if val_loader is not None else train_eval_loss
        if save_best and not math.isnan(metric) and metric < self._best_val_loss:
            self._best_val_loss = metric
            self._save_best_checkpoint()

        if self.is_main_process:
            best_marker = " 🏆 new best!" if save_best and not math.isnan(metric) and metric <= self._best_val_loss else ""
            print(f"  📊 eval step={self._global_step} | train={train_eval_loss:.4f} | val={val_loss:.4f}{best_marker}", flush=True)

    def _run_epoch_end_eval(self, train_eval_loader, val_loader, eval_batches: int, save_best: bool, autocast_dtype: torch.dtype | None = None) -> None:
        device = self.runtime.device
        train_eval_loss, val_loss = self._evaluator.evaluate(
            self.model, train_eval_loader, val_loader, device=device, max_batches=eval_batches, autocast_dtype=autocast_dtype,
        )
        self._train_losses.append(train_eval_loss)
        self._val_losses.append(val_loss)
        self._append_metrics(train_eval_loss, val_loss)

        metric = val_loss if val_loader is not None else train_eval_loss
        if save_best and not math.isnan(metric) and metric < self._best_val_loss:
            self._best_val_loss = metric
            self._save_best_checkpoint()

        self._log(f"  ✅ Epoch {self._epoch} done | train={train_eval_loss:.4f} | val={val_loss:.4f}")

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
            "5. Use the 16GB-optimized config: train.16gb.yaml",
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
                checkpoint_path=str(self._resume_cfg["output_checkpoint_path"]),
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
                    "muon_learning_rate": self._optimizer_cfg.get("muon_learning_rate"),
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
                val_loss=val_loss if not math.isnan(val_loss) else None,
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
        resume_state_path = self._paths.get("resume_state_path") or self._resume_cfg.get("resume_state_path")
        return ProteinPretrainResult(
            checkpoint_path=self._resume_cfg["output_checkpoint_path"],
            best_checkpoint_path=self._resume_cfg["best_checkpoint_path"] if best is not None else None,
            final_checkpoint_path=self._resume_cfg["final_checkpoint_path"],
            resume_state_path=Path(resume_state_path) if resume_state_path else None,
            global_step=self._global_step,
            tokens_seen=self._tokens_seen,
            epochs_completed=self._epoch,
            best_loss=best,
            final_train_loss=final_train,
            final_val_loss=final_val,
        )

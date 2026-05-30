"""VRAM memory estimation and profiling for protein pretraining on 16GB GPUs."""
from __future__ import annotations

import gc
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from libs.core.interfaces import CausalLMBatch
from libs.core.mdc import MDCDecoderModel
from libs.core.mdc.config import MDCModelConfig
from libs.core.mdc.linear_attention import (
    is_fast_path_available as _is_fast_path_available,
    _missing_fast_path_libs as _missing_libs,
)
from libs.core.pretrain.training import compute_mdc_causal_lm_loss
from libs.core.pretrain.training_config import (
    create_protein_training_optimizer,
    load_protein_training_config,
)
from libs.core.pretrain.protein_lm.support.backbone import (
    build_mdc_config_from_progen_config,
    build_progen_config,
)
from libs.data.training.tokenizer import SequenceTokenizer


def _bytes_per_element(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _resolve_dtype_from_mixed_precision(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
    return torch.float32


def estimate_protein_pretrain_memory(
    model_config: MDCModelConfig | Mapping[str, Any],
    tokenizer: SequenceTokenizer,
    batch_size: int,
    context_length: int,
    optimizer_type: str = "muon",
    dtype: torch.dtype = torch.float32,
    mixed_precision: str = "no",
    include_optimizer_state: bool = True,
) -> dict[str, Any]:
    """Estimate VRAM usage from model config without running a forward pass.

    Returns a dict with memory breakdown in bytes and GB.
    """
    model = MDCDecoderModel(model_config)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    bytes_per_param = _bytes_per_element(dtype)
    param_memory_bytes = param_count * bytes_per_param

    # Gradients stored in the same dtype for most cases
    # With mixed precision autocast, gradients are still in float32 for optimizer
    grad_dtype = torch.float32 if mixed_precision in ("fp16", "bf16") else dtype
    bytes_per_grad = _bytes_per_element(grad_dtype)
    gradient_memory_bytes = trainable_param_count * bytes_per_grad

    # Optimizer state estimation
    optimizer_state_bytes = 0
    if include_optimizer_state:
        # AdamW: 2 states per param (momentum + variance) in float32
        # Muon: momentum only for muon params, AdamW for the rest
        if optimizer_type == "muon":
            # Muon params: 2D non-embedding params get 1 momentum state
            # AdamW params: embedding/bias/norm get 2 states
            muon_param_count = 0
            adamw_param_count = 0
            embedding_param_names: set[str] = set()
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    for pname, _ in module.named_parameters(recurse=False):
                        full_name = f"{module_name}.{pname}" if module_name else pname
                        embedding_param_names.add(full_name)

            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim == 2 and name not in embedding_param_names:
                    muon_param_count += p.numel()
                else:
                    adamw_param_count += p.numel()

            # Muon stores momentum in same dtype as param
            muon_state_bytes = muon_param_count * _bytes_per_element(torch.float32)
            # AdamW stores exp_avg + exp_avg_sq in float32
            adamw_state_bytes = adamw_param_count * 2 * _bytes_per_element(torch.float32)
            optimizer_state_bytes = muon_state_bytes + adamw_state_bytes
        else:
            # Pure AdamW: 2 states per trainable param
            optimizer_state_bytes = trainable_param_count * 2 * _bytes_per_element(torch.float32)

    # Activation memory estimate (rough):
    # Forward: input embeddings + attention masks + intermediate activations per layer
    # Key contributors: attention scores, FFN intermediates, logits
    seq_len = context_length
    n_layers = int(model_config.n_layers) if hasattr(model_config, "n_layers") else int(model_config["n_layers"])
    emb_dim_val = int(model_config.emb_dim) if hasattr(model_config, "emb_dim") else int(model_config["emb_dim"])
    hidden_dim_val = int(model_config.hidden_dim) if hasattr(model_config, "hidden_dim") else int(model_config["hidden_dim"])
    n_heads_val = int(model_config.n_heads) if hasattr(model_config, "n_heads") else int(model_config["n_heads"])
    vocab_size_val = int(model_config.vocab_size) if hasattr(model_config, "vocab_size") else int(model_config["vocab_size"])

    act_dtype = dtype if mixed_precision == "no" else (torch.bfloat16 if mixed_precision == "bf16" else torch.float16)
    act_bytes = _bytes_per_element(act_dtype)

    # Per-layer activations kept for backward:
    # - input to layer: batch * seq * emb_dim
    # - attention scores: batch * n_heads * seq * seq (for full attention)
    # - FFN intermediate: batch * seq * hidden_dim
    # - residual connections: batch * seq * emb_dim
    per_layer_input = batch_size * seq_len * emb_dim_val * act_bytes
    per_layer_attn_scores = batch_size * n_heads_val * seq_len * seq_len * act_bytes
    per_layer_ffn = batch_size * seq_len * hidden_dim_val * act_bytes
    per_layer_residual = batch_size * seq_len * emb_dim_val * act_bytes
    per_layer_total = per_layer_input + per_layer_attn_scores + per_layer_ffn + per_layer_residual

    # Total activation memory (rough upper bound)
    activation_memory_bytes = n_layers * per_layer_total

    # Logits: batch * seq * vocab
    logits_memory_bytes = batch_size * seq_len * vocab_size_val * act_bytes

    # Total estimate
    total_estimate_bytes = (
        param_memory_bytes
        + gradient_memory_bytes
        + optimizer_state_bytes
        + activation_memory_bytes
        + logits_memory_bytes
    )

    del model
    gc.collect()

    return {
        "resolved_vocab_size": tokenizer.vocab_size,
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "bytes_per_param": bytes_per_param,
        "param_memory_bytes": param_memory_bytes,
        "param_memory_gb": param_memory_bytes / (1024**3),
        "gradient_memory_bytes": gradient_memory_bytes,
        "gradient_memory_gb": gradient_memory_bytes / (1024**3),
        "optimizer_state_bytes": optimizer_state_bytes,
        "optimizer_state_gb": optimizer_state_bytes / (1024**3),
        "activation_memory_bytes": activation_memory_bytes,
        "activation_memory_gb": activation_memory_bytes / (1024**3),
        "logits_memory_bytes": logits_memory_bytes,
        "logits_memory_gb": logits_memory_bytes / (1024**3),
        "total_estimate_bytes": total_estimate_bytes,
        "total_estimate_gb": total_estimate_bytes / (1024**3),
        "batch_size": batch_size,
        "context_length": context_length,
        "dtype": str(dtype),
        "mixed_precision": mixed_precision,
        "optimizer_type": optimizer_type,
        "fast_path_available": _is_fast_path_available,
        "missing_fast_path_libs": list(_missing_libs) if _missing_libs else [],
        "is_estimate": True,
        "measured_on_cuda": False,
    }


def profile_protein_pretrain_memory(
    model: torch.nn.Module,
    tokenizer: SequenceTokenizer,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
    optimizer: Any,
    run_forward_backward: bool = True,
) -> dict[str, Any]:
    """Profile actual VRAM usage by running a dummy batch on CUDA.

    Requires CUDA to be available. Returns measured peak memory.
    """
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise RuntimeError("profile_protein_pretrain_memory requires a CUDA device.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for memory profiling.")

    model = model.to(resolved_device)

    # Clear memory stats
    torch.cuda.reset_peak_memory_stats(resolved_device)
    torch.cuda.empty_cache()
    gc.collect()

    mem_before = torch.cuda.memory_allocated(resolved_device)

    # Create dummy batch
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, context_length), device=resolved_device)
    attention_mask = torch.ones(batch_size, context_length, dtype=torch.bool, device=resolved_device)
    labels = input_ids.clone()

    if run_forward_backward:
        model.train()
        optimizer_list = list(optimizer) if isinstance(optimizer, (list, tuple)) else [optimizer]

        for opt in optimizer_list:
            opt.zero_grad(set_to_none=True)

        logits = model(input_ids, attn_mask=attention_mask)
        loss = compute_mdc_causal_lm_loss(logits, labels)
        loss.backward()

        for opt in optimizer_list:
            opt.step()

    peak_allocated = torch.cuda.max_memory_allocated(resolved_device)
    peak_reserved = torch.cuda.max_memory_reserved(resolved_device)
    mem_after = torch.cuda.memory_allocated(resolved_device)

    # Cleanup
    del input_ids, attention_mask, labels
    if run_forward_backward:
        del logits, loss
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "measured_on_cuda": True,
        "device": str(resolved_device),
        "mem_before_bytes": mem_before,
        "mem_after_bytes": mem_after,
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_gb": peak_allocated / (1024**3),
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_gb": peak_reserved / (1024**3),
        "batch_size": batch_size,
        "context_length": context_length,
    }


def build_vram_report(
    project_root: Path | str,
    config: Mapping[str, Any],
    tokenizer: SequenceTokenizer,
    model_config: MDCModelConfig,
    max_vram_gb: float = 16.0,
    target_vram_fraction: float = 0.85,
) -> dict[str, Any]:
    """Build a full VRAM report for the given config.

    Uses estimation. If CUDA is available, also runs profiling.
    """
    project_root = Path(project_root).resolve()

    batch_size = config["data"]["batch_size"]
    context_length = config["model"]["context_length"]
    optimizer_type = config["optimizer"]["type"]
    mixed_precision = config["runtime"]["mixed_precision"]

    resolved_dtype = _resolve_dtype_from_mixed_precision(mixed_precision)

    estimate = estimate_protein_pretrain_memory(
        model_config=model_config,
        tokenizer=tokenizer,
        batch_size=batch_size,
        context_length=context_length,
        optimizer_type=optimizer_type,
        dtype=resolved_dtype,
        mixed_precision=mixed_precision,
    )

    target_budget_gb = min(max_vram_gb * target_vram_fraction, max_vram_gb - 2.0)
    estimated_peak_gb = estimate["total_estimate_gb"]

    report: dict[str, Any] = {
        "tokenizer_map_path": str(config["paths"]["tokenizer_map_path"]),
        "resolved_vocab_size": tokenizer.vocab_size,
        "parameter_count": estimate["param_count"],
        "model_config": model_config.to_dict() if hasattr(model_config, "to_dict") else dict(model_config),
        "batch_size": batch_size,
        "context_length": context_length,
        "gradient_accumulation_steps": config["training"].get("gradient_accumulation_steps", 1),
        "optimizer_type": optimizer_type,
        "mixed_precision": mixed_precision,
        "resolved_dtype": str(resolved_dtype),
        "total_vram_gb": max_vram_gb,
        "target_budget_gb": target_budget_gb,
        "estimated_peak_gb": estimated_peak_gb,
        "peak_allocated_gb": None,
        "peak_reserved_gb": None,
        "margin_gb": target_budget_gb - estimated_peak_gb,
        "fit": estimated_peak_gb <= target_budget_gb,
        "fast_path_available": _is_fast_path_available,
        "missing_fast_path_libs": list(_missing_libs) if _missing_libs else [],
        "estimate": estimate,
        "measured_peak_gb": None,
        "profile": None,
        "recommended_config": None,
    }

    if torch.cuda.is_available():
        try:
            model = MDCDecoderModel(model_config).to(resolved_dtype)
            optimizer = create_protein_training_optimizer(
                model, config["optimizer"], device="cuda"
            )
            profile = profile_protein_pretrain_memory(
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                context_length=context_length,
                device="cuda",
                optimizer=optimizer,
                run_forward_backward=True,
            )
            report["measured_peak_gb"] = profile["peak_allocated_gb"]
            report["peak_allocated_gb"] = profile["peak_allocated_gb"]
            report["peak_reserved_gb"] = profile["peak_reserved_gb"]
            report["profile"] = profile
            report["fit"] = profile["peak_allocated_gb"] <= target_budget_gb

            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            report["fit"] = False
            report["profile_error"] = "OOM during preflight profiling — config does not fit"
            gc.collect()
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            report["profile_error"] = str(exc)

    return report


def recommend_16gb_train_config(
    project_root: Path | str,
    config_path: Path | str | None = None,
    max_vram_gb: float = 16.0,
    target_vram_fraction: float = 0.85,
) -> dict[str, Any]:
    """Find a training config that fits within the VRAM budget.

    Tries candidates from small to large, keeping model config unchanged
    unless batch/context reduction is insufficient.
    """
    project_root = Path(project_root).resolve()
    config = load_protein_training_config(project_root, config_path=config_path)

    tokenizer_map_path = config["paths"]["tokenizer_map_path"]
    if not tokenizer_map_path.exists():
        raise FileNotFoundError(
            f"tokenizer_map.json not found at {tokenizer_map_path}. "
            "Cannot estimate memory without real tokenizer."
        )
    tokenizer = SequenceTokenizer.load_map(tokenizer_map_path)

    mixed_precision = config["runtime"]["mixed_precision"]
    resolved_dtype = _resolve_dtype_from_mixed_precision(mixed_precision)

    model_cfg = config["model"]
    progen_config = build_progen_config(
        model_cfg["progen_model_size"],
        vocab_size=tokenizer.vocab_size,
        context_length=model_cfg["context_length"],
        dtype=resolved_dtype,
    )
    overrides = model_cfg["progen_config_overrides"]
    if overrides:
        progen_config = {**progen_config, **overrides}
    model_config = build_mdc_config_from_progen_config(progen_config, dtype=resolved_dtype)

    target_budget_gb = min(max_vram_gb * target_vram_fraction, max_vram_gb - 2.0)

    # Check if current config already fits
    current_estimate = estimate_protein_pretrain_memory(
        model_config=model_config,
        tokenizer=tokenizer,
        batch_size=config["data"]["batch_size"],
        context_length=model_cfg["context_length"],
        optimizer_type=config["optimizer"]["type"],
        dtype=resolved_dtype,
        mixed_precision=mixed_precision,
    )

    if current_estimate["total_estimate_gb"] <= target_budget_gb:
        return {
            "status": "current_config_fits",
            "config": config,
            "model_config": model_config,
            "tokenizer": tokenizer,
            "estimate": current_estimate,
            "target_budget_gb": target_budget_gb,
            "reason": "Current config estimated to fit within budget.",
            "recommended_changes": {},
        }

    # Try candidates: reduce batch_size and context_length first
    context_candidates = [256, 384, 512, 768, 1024]
    batch_candidates = [1, 2, 4]
    grad_accum_candidates = [1, 2, 4, 8]

    # Original effective batch = batch_size * 1 (no grad accum currently)
    original_effective_batch = config["data"]["batch_size"]

    best_candidate = None
    candidate_table = []

    for ctx_len in context_candidates:
        for bs in batch_candidates:
            test_progen_config = build_progen_config(
                model_cfg["progen_model_size"],
                vocab_size=tokenizer.vocab_size,
                context_length=ctx_len,
                dtype=resolved_dtype,
            )
            if overrides:
                test_progen_config = {**test_progen_config, **overrides}
            test_model_config = build_mdc_config_from_progen_config(test_progen_config, dtype=resolved_dtype)

            est = estimate_protein_pretrain_memory(
                model_config=test_model_config,
                tokenizer=tokenizer,
                batch_size=bs,
                context_length=ctx_len,
                optimizer_type=config["optimizer"]["type"],
                dtype=resolved_dtype,
                mixed_precision=mixed_precision,
            )

            fits = est["total_estimate_gb"] <= target_budget_gb

            # Pick gradient_accumulation to match original effective batch
            grad_accum = max(1, original_effective_batch // bs)
            # Find closest from candidates
            best_accum = min(grad_accum_candidates, key=lambda x: abs(x - grad_accum))

            entry = {
                "batch_size": bs,
                "context_length": ctx_len,
                "gradient_accumulation_steps": best_accum,
                "effective_batch_size": bs * best_accum,
                "estimated_peak_gb": est["total_estimate_gb"],
                "fits": fits,
            }
            candidate_table.append(entry)

            if fits and best_candidate is None:
                # Prefer largest context that fits with reasonable batch
                pass
            if fits:
                # Keep updating — prefer larger context_length and batch_size
                if best_candidate is None:
                    best_candidate = entry
                elif ctx_len > best_candidate["context_length"]:
                    best_candidate = entry
                elif ctx_len == best_candidate["context_length"] and bs > best_candidate["batch_size"]:
                    best_candidate = entry

    if best_candidate is None:
        # Nothing fits — need to reduce model
        # Try smallest config
        reduced_overrides = {
            **overrides,
            "n_layers": max(4, overrides.get("n_layers", 16) // 2),
            "emb_dim": max(256, overrides.get("emb_dim", 1024) // 2),
            "n_heads": max(4, overrides.get("n_heads", 16) // 2),
            "hidden_dim": max(1024, overrides.get("hidden_dim", 4096) // 2),
        }
        reduced_progen = build_progen_config(
            model_cfg["progen_model_size"],
            vocab_size=tokenizer.vocab_size,
            context_length=512,
            dtype=resolved_dtype,
        )
        reduced_progen = {**reduced_progen, **reduced_overrides}
        reduced_model_config = build_mdc_config_from_progen_config(reduced_progen, dtype=resolved_dtype)
        est = estimate_protein_pretrain_memory(
            model_config=reduced_model_config,
            tokenizer=tokenizer,
            batch_size=2,
            context_length=512,
            optimizer_type=config["optimizer"]["type"],
            dtype=resolved_dtype,
            mixed_precision=mixed_precision,
        )
        best_candidate = {
            "batch_size": 2,
            "context_length": 512,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
            "estimated_peak_gb": est["total_estimate_gb"],
            "fits": est["total_estimate_gb"] <= target_budget_gb,
            "model_reduced": True,
            "reduced_overrides": reduced_overrides,
        }
        candidate_table.append(best_candidate)

    # Build recommended changes
    recommended_changes: dict[str, Any] = {
        "data": {"batch_size": best_candidate["batch_size"]},
        "model": {"context_length": best_candidate["context_length"]},
        "training": {"gradient_accumulation_steps": best_candidate["gradient_accumulation_steps"]},
    }
    if best_candidate.get("model_reduced"):
        recommended_changes["model"]["progen_config_overrides"] = best_candidate["reduced_overrides"]

    return {
        "status": "recommended",
        "config": config,
        "model_config": model_config,
        "tokenizer": tokenizer,
        "estimate": current_estimate,
        "target_budget_gb": target_budget_gb,
        "candidate_table": candidate_table,
        "chosen": best_candidate,
        "reason": (
            f"Current config estimated {current_estimate['total_estimate_gb']:.2f}GB "
            f"exceeds budget {target_budget_gb:.2f}GB. "
            f"Recommended: batch_size={best_candidate['batch_size']}, "
            f"context_length={best_candidate['context_length']}, "
            f"gradient_accumulation_steps={best_candidate['gradient_accumulation_steps']}."
        ),
        "recommended_changes": recommended_changes,
    }


def write_vram_report(report: dict[str, Any], output_path: Path | str) -> Path:
    """Write VRAM report to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Make report JSON-serializable
    serializable = _make_json_serializable(report)
    output_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return output_path


def run_preflight_vram_check(
    model: torch.nn.Module,
    tokenizer: SequenceTokenizer,
    optimizer: Any,
    *,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
    target_vram_gb: float = 14.0,
    mixed_precision: str = "auto",
    gradient_accumulation_steps: int = 1,
    report_output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run a preflight VRAM check before training starts.

    If CUDA is available, runs a dummy forward/backward/optimizer step and
    measures peak memory. Catches OOM and reports config doesn't fit.

    Returns a dict with fit=True/False, peak memory, and recommendations.
    Raises RuntimeError if config doesn't fit and training should not proceed.
    """
    resolved_device = torch.device(device)
    param_count = sum(p.numel() for p in model.parameters())

    result: dict[str, Any] = {
        "fit": True,
        "device": str(resolved_device),
        "target_vram_gb": target_vram_gb,
        "parameter_count": param_count,
        "batch_size": batch_size,
        "context_length": context_length,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "mixed_precision": mixed_precision,
        "fast_path_available": _is_fast_path_available,
        "missing_fast_path_libs": list(_missing_libs) if _missing_libs else [],
        "peak_allocated_gb": None,
        "peak_reserved_gb": None,
        "total_vram_gb": None,
    }

    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        # Cannot measure on CPU — estimate only
        result["note"] = "Preflight skipped: no CUDA device. Using estimates only."
        return result

    total_vram = torch.cuda.get_device_properties(resolved_device).total_mem
    result["total_vram_gb"] = total_vram / (1024**3)

    # Resolve autocast dtype
    autocast_dtype = None
    if mixed_precision == "bf16":
        autocast_dtype = torch.bfloat16
    elif mixed_precision == "fp16":
        autocast_dtype = torch.float16
    elif mixed_precision == "auto":
        if torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16
    use_autocast = autocast_dtype is not None

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(resolved_device)
        gc.collect()

        # Create dummy batch
        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=resolved_device)
        attention_mask = torch.ones(batch_size, context_length, dtype=torch.bool, device=resolved_device)
        labels = input_ids.clone()

        model.train()
        optimizer_list = list(optimizer) if isinstance(optimizer, (list, tuple)) else [optimizer]

        for opt in optimizer_list:
            opt.zero_grad(set_to_none=True)

        # Forward with autocast
        if use_autocast:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                logits = model(input_ids, attn_mask=attention_mask)
                loss = compute_mdc_causal_lm_loss(logits, labels)
        else:
            logits = model(input_ids, attn_mask=attention_mask)
            loss = compute_mdc_causal_lm_loss(logits, labels)

        # Backward
        (loss / gradient_accumulation_steps).backward()

        # Optimizer step
        for opt in optimizer_list:
            opt.step()

        peak_allocated = torch.cuda.max_memory_allocated(resolved_device)
        peak_reserved = torch.cuda.max_memory_reserved(resolved_device)

        result["peak_allocated_gb"] = peak_allocated / (1024**3)
        result["peak_reserved_gb"] = peak_reserved / (1024**3)
        result["fit"] = (peak_allocated / (1024**3)) <= target_vram_gb

        # Cleanup
        del input_ids, attention_mask, labels, logits, loss
        for opt in optimizer_list:
            opt.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        result["fit"] = False
        result["oom_during_preflight"] = True

    # Write report if path provided
    if report_output_path is not None:
        write_vram_report(result, report_output_path)

    if not result["fit"]:
        suggested_fixes = [
            "1. Use batch_size=1",
            "2. Reduce context_length to 512 or 384",
            "3. Set eval_batches=1",
            "4. Increase eval_freq to reduce eval frequency",
            "5. Use the 16GB-optimized config: train.16gb.yaml",
        ]
        msg = (
            f"\n{'='*60}\n"
            f"VRAM PREFLIGHT CHECK FAILED\n"
            f"{'='*60}\n"
            f"  batch_size={batch_size}\n"
            f"  context_length={context_length}\n"
            f"  model_params={param_count:,}\n"
            f"  target_vram_gb={target_vram_gb:.1f}\n"
            f"  peak_allocated_gb={result.get('peak_allocated_gb', 'OOM')}\n"
            f"  fast_path_available={_is_fast_path_available}\n"
        )
        if _missing_libs:
            msg += f"  missing_fast_path_libs={_missing_libs}\n"
        msg += f"\nSuggested fixes:\n"
        for fix in suggested_fixes:
            msg += f"  {fix}\n"
        msg += f"{'='*60}\n"
        raise RuntimeError(msg)

    return result


def _make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, (SequenceTokenizer, MDCModelConfig)):
        return str(type(obj).__name__)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj

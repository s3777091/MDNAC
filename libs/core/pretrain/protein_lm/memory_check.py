"""CLI: Check VRAM memory requirements for protein pretraining.

Usage:
    python -m libs.core.pretrain.protein_lm.memory_check --config train.yaml --max-vram-gb 16
    python -m libs.core.pretrain.protein_lm.memory_check --config train.yaml --max-vram-gb 16 --write train.16gb.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
import torch

from libs.core.pretrain.protein_lm.memory import (
    build_vram_report,
    estimate_protein_pretrain_memory,
    recommend_16gb_train_config,
    write_vram_report,
    _resolve_dtype_from_mixed_precision,
)
from libs.core.pretrain.protein_lm.support.backbone import (
    build_mdc_config_from_progen_config,
    build_progen_config,
)
from libs.core.pretrain.training_config import load_protein_training_config
from libs.data.training.tokenizer import SequenceTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check VRAM memory requirements for protein pretraining."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="train.yaml",
        help="Path to train.yaml config file.",
    )
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=16.0,
        help="Maximum VRAM budget in GB (default: 16).",
    )
    parser.add_argument(
        "--target-fraction",
        type=float,
        default=0.85,
        help="Fraction of VRAM to target (default: 0.85).",
    )
    parser.add_argument(
        "--write",
        type=str,
        default=None,
        help="Write recommended config to this YAML path.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Write VRAM report JSON to this path.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = args.config
    max_vram_gb = args.max_vram_gb
    target_fraction = args.target_fraction

    print(f"Project root: {project_root}")
    print(f"Config: {config_path}")
    print(f"Max VRAM: {max_vram_gb:.1f} GB")
    print(f"Target fraction: {target_fraction:.0%}")
    print()

    # Load config
    config = load_protein_training_config(project_root, config_path=config_path)

    # Load tokenizer
    tokenizer_map_path = config["paths"]["tokenizer_map_path"]
    if not tokenizer_map_path.exists():
        print(f"ERROR: tokenizer_map.json not found at {tokenizer_map_path}")
        print("Cannot estimate memory without real tokenizer.")
        sys.exit(1)

    tokenizer = SequenceTokenizer.load_map(tokenizer_map_path)
    print(f"Tokenizer loaded from: {tokenizer_map_path}")
    print(f"Resolved vocab_size: {tokenizer.vocab_size}")
    print()

    # Build model config
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

    # Estimate memory
    estimate = estimate_protein_pretrain_memory(
        model_config=model_config,
        tokenizer=tokenizer,
        batch_size=config["data"]["batch_size"],
        context_length=model_cfg["context_length"],
        optimizer_type=config["optimizer"]["type"],
        dtype=resolved_dtype,
        mixed_precision=mixed_precision,
    )

    target_budget_gb = min(max_vram_gb * target_fraction, max_vram_gb - 2.0)

    print("=" * 60)
    print("CURRENT CONFIG MEMORY ESTIMATE")
    print("=" * 60)
    print(f"  Parameter count:     {estimate['param_count']:>15,}")
    print(f"  Trainable params:    {estimate['trainable_param_count']:>15,}")
    print(f"  Model dtype:         {resolved_dtype}")
    print(f"  Mixed precision:     {mixed_precision}")
    print(f"  Batch size:          {config['data']['batch_size']}")
    print(f"  Context length:      {model_cfg['context_length']}")
    print(f"  Grad accum steps:    {config['training'].get('gradient_accumulation_steps', 1)}")
    print()
    print(f"  Parameters:          {estimate['param_memory_gb']:>8.3f} GB")
    print(f"  Gradients:           {estimate['gradient_memory_gb']:>8.3f} GB")
    print(f"  Optimizer state:     {estimate['optimizer_state_gb']:>8.3f} GB")
    print(f"  Activations (est):   {estimate['activation_memory_gb']:>8.3f} GB")
    print(f"  Logits:              {estimate['logits_memory_gb']:>8.3f} GB")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL ESTIMATED:     {estimate['total_estimate_gb']:>8.3f} GB")
    print(f"  Target budget:       {target_budget_gb:>8.3f} GB")
    print(f"  Margin:              {target_budget_gb - estimate['total_estimate_gb']:>+8.3f} GB")
    print()

    fits = estimate["total_estimate_gb"] <= target_budget_gb
    if fits:
        print(f"  ✓ Current config ESTIMATED to fit within {max_vram_gb:.0f}GB budget.")
    else:
        print(f"  ✗ Current config ESTIMATED to EXCEED {max_vram_gb:.0f}GB budget!")

    if not torch.cuda.is_available():
        print()
        print("  ⚠ No CUDA available — this is only an ESTIMATE.")
        print("    Run on a machine with GPU to measure actual peak memory.")
    print()

    # Run recommendation
    result = recommend_16gb_train_config(
        project_root,
        config_path=config_path,
        max_vram_gb=max_vram_gb,
        target_vram_fraction=target_fraction,
    )

    if result["status"] == "current_config_fits":
        print("Current config already fits within budget. No changes needed.")
    else:
        print("=" * 60)
        print("RECOMMENDED CONFIG FOR 16GB VRAM")
        print("=" * 60)
        chosen = result["chosen"]
        print(f"  Batch size:              {chosen['batch_size']}")
        print(f"  Context length:          {chosen['context_length']}")
        print(f"  Grad accumulation steps: {chosen['gradient_accumulation_steps']}")
        print(f"  Effective batch size:    {chosen['effective_batch_size']}")
        print(f"  Estimated peak:          {chosen['estimated_peak_gb']:.3f} GB")
        print(f"  Fits budget:             {chosen['fits']}")
        if chosen.get("model_reduced"):
            print(f"  Model reduced:           Yes")
            print(f"  Reduced overrides:       {chosen['reduced_overrides']}")
        print()
        print(f"  Reason: {result['reason']}")
        print()

    # Write report
    report_path = args.report_path or str(
        config["paths"]["checkpoint_dir"] / "vram_16gb_report.json"
    )
    report_data = {
        "tokenizer_map_path": str(tokenizer_map_path),
        "resolved_vocab_size": tokenizer.vocab_size,
        "model_config": model_config.to_dict() if hasattr(model_config, "to_dict") else {},
        "train_yaml_input": str(config["config_path"]),
        "parameter_count": estimate["param_count"],
        "trainable_parameter_count": estimate["trainable_param_count"],
        "optimizer_type": config["optimizer"]["type"],
        "dtype": str(resolved_dtype),
        "mixed_precision": mixed_precision,
        "current_estimate": estimate,
        "target_budget_gb": target_budget_gb,
        "max_vram_gb": max_vram_gb,
        "recommendation": {
            "status": result["status"],
            "reason": result.get("reason", ""),
            "chosen": result.get("chosen"),
            "recommended_changes": result.get("recommended_changes", {}),
        },
        "candidate_table": result.get("candidate_table", []),
        "cuda_available": torch.cuda.is_available(),
        "measured_peak_memory": None,
    }
    write_vram_report(report_data, report_path)
    print(f"Report written to: {report_path}")

    # Optionally write recommended YAML
    if args.write:
        _write_recommended_yaml(
            project_root,
            config,
            result,
            tokenizer,
            resolved_dtype,
            output_path=args.write,
        )
        print(f"Recommended config written to: {args.write}")


def _to_relative_path(value: Any, project_root: Path) -> str:
    """Convert a Path to a relative POSIX string for portability."""
    try:
        path = Path(value) if not isinstance(value, Path) else value
        rel = path.relative_to(project_root)
        return rel.as_posix()
    except (ValueError, TypeError):
        return str(value)


def _write_recommended_yaml(
    project_root: Path,
    config: dict,
    result: dict,
    tokenizer: SequenceTokenizer,
    resolved_dtype: torch.dtype,
    output_path: str,
) -> None:
    changes = result.get("recommended_changes", {})
    chosen = result.get("chosen", {})

    # Start from current config values
    batch_size = changes.get("data", {}).get("batch_size", config["data"]["batch_size"])
    context_length = changes.get("model", {}).get("context_length", config["model"]["context_length"])
    grad_accum = changes.get("training", {}).get(
        "gradient_accumulation_steps",
        config["training"].get("gradient_accumulation_steps", 1),
    )
    stride = context_length // 2

    # Model overrides
    model_overrides = config["model"]["progen_config_overrides"].copy()
    if changes.get("model", {}).get("progen_config_overrides"):
        model_overrides.update(changes["model"]["progen_config_overrides"])

    # Determine mixed precision string
    mp = config["runtime"]["mixed_precision"]
    if mp == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            mp = "bf16"
        elif torch.cuda.is_available():
            mp = "fp16"

    yaml_content = {
        "mode": {"name": config["mode"]["name"], "resume_if_available": True},
        "paths": {k: _to_relative_path(v, project_root) for k, v in config["paths"].items()},
        "data": {
            "train_part_glob": config["data"]["train_part_glob"],
            "prefer_local_train_parts": config["data"]["prefer_local_train_parts"],
            "stream_local_train_parts": config["data"]["stream_local_train_parts"],
            "keep_downloaded_train_parts": config["data"]["keep_downloaded_train_parts"],
            "cleanup_completed_parts": config["data"]["cleanup_completed_parts"],
            "validate_cached_parts": config["data"]["validate_cached_parts"],
            "train_ratio": config["data"]["train_ratio"],
            "batch_size": batch_size,
            "num_workers": min(config["data"]["num_workers"], 2),
            "pin_memory": True if torch.cuda.is_available() else False,
            "shuffle_parts": config["data"]["shuffle_parts"],
            "shuffle_examples": config["data"]["shuffle_examples"],
            "shuffle_buffer_size": min(config["data"]["shuffle_buffer_size"], 10000),
        },
        "model": {
            "progen_model_size": config["model"]["progen_model_size"],
            "context_length": context_length,
            "stride": stride,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "rebuild_tokenizer": False,
            "progen_config_overrides": model_overrides,
        },
        "training": {
            "num_epochs": config["training"]["num_epochs"],
            "max_steps": config["training"]["max_steps"],
            "gradient_accumulation_steps": grad_accum,
            "save_every_steps": config["training"]["save_every_steps"],
            "eval_freq": config["training"]["eval_freq"],
            "eval_batches": min(config["training"]["eval_batches"], 10),
            "grad_clip_norm": config["training"]["grad_clip_norm"],
            "save_last": True,
            "save_best": True,
            "save_final": True,
        },
        "optimizer": {
            "type": config["optimizer"]["type"],
            "learning_rate": config["optimizer"]["learning_rate"],
            "muon_learning_rate": config["optimizer"].get("muon_learning_rate"),
            "weight_decay": config["optimizer"]["weight_decay"],
            "fused": "auto",
        },
        "runtime": {
            "device": "auto",
            "multi_gpu_mode": "auto",
            "ddp_find_unused_parameters": False,
            "data_parallel_device_ids": None,
            "mixed_precision": mp,
            "preflight_vram_check": True,
            "target_vram_gb": 16,
        },
        "resume": {k: _to_relative_path(v, project_root) if isinstance(v, Path) else v for k, v in config["resume"].items()},
        "minio": {
            "train_parts_prefix_uri": config["minio"]["train_parts_prefix_uri"],
            "train_part_uris": list(config["minio"]["train_part_uris"]),
            "manifest_uri": config["minio"]["manifest_uri"],
            "endpoint_url": config["minio"]["endpoint_url"],
            "access_key": None,
            "secret_key": None,
            "bucket_name": config["minio"]["bucket_name"],
            "region_name": config["minio"]["region_name"],
            "secure": config["minio"]["secure"],
        },
    }

    output = Path(output_path)
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.dump(yaml_content, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

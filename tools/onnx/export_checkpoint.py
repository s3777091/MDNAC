from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OPSET = 17
DEFAULT_ATOL = 1e-4


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device | torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class CausalLMExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)


def _default_output_path(checkpoint_path: Path) -> Path:
    export_dir = Path(__file__).resolve().parent / "exports"
    stem = checkpoint_path.stem
    if stem == "checkpoint_last" and checkpoint_path.parent.name:
        stem = checkpoint_path.parent.name
    return export_dir / f"{stem}.onnx"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Export a train_ava training checkpoint (.pt) to ONNX.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a training checkpoint such as checkpoint_best.pt or checkpoint_last.pt.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output ONNX path. Defaults to tools/onnx/exports/<run_name>.onnx.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size used during export and optional verification.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Dummy input length used during export. Defaults to min(context_length, 16).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help="ONNX opset version passed to torch.onnx.export.",
    )
    parser.add_argument(
        "--static-shapes",
        action="store_true",
        help="Export fixed input shapes instead of dynamic batch/sequence axes.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNX Runtime output comparison after export.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help="Absolute tolerance used when comparing PyTorch and ONNX Runtime outputs.",
    )
    return parser.parse_args()


def _require_onnx_export_dependencies() -> None:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The `onnx` package is required for export. "
            "Install it with `uv sync --extra onnx`."
        ) from exc


def _resolve_seq_len(model_config: dict[str, Any], seq_len: int | None) -> int:
    context_length = int(model_config["context_length"])
    effective_seq_len = min(context_length, 16) if seq_len is None else seq_len
    if effective_seq_len <= 0 or effective_seq_len > context_length:
        raise ValueError(
            f"seq_len must be in the range [1, {context_length}], got {effective_seq_len}."
        )
    return effective_seq_len


def _build_example_inputs(
    model_config: dict[str, Any],
    *,
    batch_size: int,
    seq_len: int,
) -> tuple[tuple[torch.Tensor, ...], list[str], dict[str, dict[int, str]]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    vocab_size = int(model_config["vocab_size"])
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len)
    input_ids = input_ids.remainder(vocab_size)

    input_names = ["input_ids"]
    dynamic_axes: dict[str, dict[int, str]] = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }
    return (input_ids,), input_names, dynamic_axes


def _load_checkpoint_model(
    checkpoint_path: Path,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any], str]:
    from interfere.artifacts import detect_model_family
    from train.models.qwen3_5.ch05 import build_model as build_qwen3_5_model
    from train.pipeline.runtime.artifacts import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Expected a training checkpoint containing `model_config` and `model_state_dict`. "
            f"Got keys: {sorted(checkpoint.keys())}"
        )

    model_config = checkpoint["model_config"]
    model_family = detect_model_family(checkpoint)
    if model_family != "qwen3_5":
        raise ValueError(
            "Only Qwen3.5 checkpoints can be exported to ONNX in this repo. "
            f"Received: {model_family}"
        )
    model = build_qwen3_5_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_config, checkpoint, model_family


def _verify_with_onnxruntime(
    output_path: Path,
    reference_output: np.ndarray,
    example_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    *,
    atol: float,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError:
        return {
            "verified": False,
            "verification_skipped": True,
            "reason": "onnxruntime_not_installed",
        }

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(input_names, example_inputs, strict=True)
    }
    ort_output = session.run(["logits"], ort_inputs)[0]
    max_abs_diff = float(np.max(np.abs(reference_output - ort_output)))
    if not np.allclose(reference_output, ort_output, atol=atol, rtol=0.0):
        raise RuntimeError(
            "ONNX Runtime verification failed. "
            f"max_abs_diff={max_abs_diff:.6g}, atol={atol:.6g}"
        )

    return {
        "verified": True,
        "verification_skipped": False,
        "max_abs_diff": max_abs_diff,
        "providers": session.get_providers(),
    }


def _write_metadata(
    metadata_path: Path,
    *,
    checkpoint_path: Path,
    output_path: Path,
    checkpoint: dict[str, Any],
    model_family: str,
    model_config: dict[str, Any],
    input_names: list[str],
    seq_len: int,
    batch_size: int,
    opset: int,
    dynamic_shapes: bool,
    verification: dict[str, Any],
) -> None:
    metadata = {
        "artifact_format": "onnx",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "onnx_path": str(output_path.resolve()),
        "model_family": model_family,
        "preset_name": checkpoint.get("preset_name"),
        "model_config": model_config,
        "tokenizer_file_path": checkpoint.get("tokenizer_file_path"),
        "tokenizer_repo_id": checkpoint.get("tokenizer_repo_id"),
        "tokenizer_settings": checkpoint.get("tokenizer_settings"),
        "inference_tokenizer_settings": checkpoint.get("inference_tokenizer_settings"),
        "instruction_settings": checkpoint.get("instruction_settings"),
        "instruction_mode": checkpoint.get("instruction_mode"),
        "reasoning_settings": checkpoint.get("reasoning_settings"),
        "reasoning_mode": checkpoint.get("reasoning_mode"),
        "chatbot_settings": checkpoint.get("chatbot_settings"),
        "export": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "opset": opset,
            "dynamic_shapes": dynamic_shapes,
            "input_names": input_names,
            "output_names": ["logits"],
        },
        "verification": verification,
    }
    metadata_path.write_text(
        json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    _require_onnx_export_dependencies()

    model, model_config, checkpoint, model_family = _load_checkpoint_model(checkpoint_path)
    seq_len = _resolve_seq_len(model_config, args.seq_len)

    wrapper = CausalLMExportWrapper(model).cpu()
    example_inputs, input_names, dynamic_axes = _build_example_inputs(
        model_config,
        batch_size=args.batch_size,
        seq_len=seq_len,
    )
    with torch.no_grad():
        reference_output = wrapper(*example_inputs).detach().cpu().numpy()

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(checkpoint_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = {
        "input_names": input_names,
        "output_names": ["logits"],
        "opset_version": args.opset,
        "do_constant_folding": True,
        "dynamo": False,
    }
    if not args.static_shapes:
        export_kwargs["dynamic_axes"] = dynamic_axes

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_inputs,
            str(output_path),
            **export_kwargs,
        )

    verification: dict[str, Any]
    if args.skip_verify:
        verification = {
            "verified": False,
            "verification_skipped": True,
            "reason": "skip_verify_flag",
        }
    else:
        verification = _verify_with_onnxruntime(
            output_path,
            reference_output,
            example_inputs,
            input_names,
            atol=args.atol,
        )

    metadata_path = output_path.with_suffix(".json")
    _write_metadata(
        metadata_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        checkpoint=checkpoint,
        model_family=model_family,
        model_config=model_config,
        input_names=input_names,
        seq_len=seq_len,
        batch_size=args.batch_size,
        opset=args.opset,
        dynamic_shapes=not args.static_shapes,
        verification=verification,
    )

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "onnx_path": str(output_path),
                "metadata_path": str(metadata_path),
                "verification": verification,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

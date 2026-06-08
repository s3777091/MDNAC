from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from transformers import Qwen3_5ForCausalLM

from train.models.qwen3_5 import (  # noqa: E402
    DEFAULT_OLLAMA_STOP_TOKENS,
    DEFAULT_OLLAMA_TEMPLATE,
    build_default_ollama_system_prompt,
    build_hf_text_config,
    build_model,
    convert_repo_state_dict_to_hf,
    export_checkpoint_to_hf_directory,
    load_export_tokenizer,
    resolve_checkpoint_tokenizer,
)
from train.pipeline.runtime.artifacts import load_checkpoint  # noqa: E402


DEFAULT_ATOL = 1e-5


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype | torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sanitize_model_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-_.")
    return normalized or "ava-qwen3_5"


def _default_run_name(checkpoint_path: Path) -> str:
    if checkpoint_path.stem in {"checkpoint_best", "checkpoint_last"} and checkpoint_path.parent.name:
        return checkpoint_path.parent.name
    return checkpoint_path.stem


def _default_ollama_model_name(checkpoint_path: Path) -> str:
    return _sanitize_model_name(f"{_default_run_name(checkpoint_path)}-ollama")


def _build_export_dir_name(ollama_model_name: str) -> str:
    normalized = _sanitize_model_name(ollama_model_name)
    base_name = re.sub(r"(?i)(?:[-_.]?ollama)$", "", normalized).strip("-_.")
    return f"{base_name or normalized}_ollama"


def _resolve_unique_output_dir(output_root: Path, ollama_model_name: str) -> Path:
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(f"--output-dir must point to a directory root: {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    folder_name = _build_export_dir_name(ollama_model_name)
    candidate = output_root / folder_name
    suffix = 2
    while candidate.exists():
        candidate = output_root / f"{folder_name}_{suffix}"
        suffix += 1
    return candidate


def _resolve_unique_file_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        return path

    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    candidate = path
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        counter += 1
    return candidate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Export a repo Qwen3.5 checkpoint to Hugging Face format and scaffold the GGUF/Ollama hosting path."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a training checkpoint such as checkpoint_best.pt or checkpoint_last.pt.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Parent export directory. The exporter creates a unique child folder "
            "named <model_name>_ollama[_N] under this root."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional explicit tokenizer.json path to copy into the export.",
    )
    parser.add_argument(
        "--weights-format",
        choices=("safetensors", "pytorch_bin"),
        default="safetensors",
        help="HF weight file format written into the export directory.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Load the converted HF model and compare logits with the repo model on a dummy input.",
    )
    parser.add_argument(
        "--verify-seq-len",
        type=int,
        default=8,
        help="Dummy sequence length used by --verify.",
    )
    parser.add_argument(
        "--verify-atol",
        type=float,
        default=DEFAULT_ATOL,
        help="Absolute tolerance used by --verify.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default=None,
        help="Optional llama.cpp directory (or direct convert_hf_to_gguf.py path) to run GGUF conversion automatically.",
    )
    parser.add_argument(
        "--gguf-output",
        default=None,
        help=(
            "Optional GGUF output path. Defaults to <resolved_output_dir>/model.gguf. "
            "If the path already exists, a unique sibling filename is used."
        ),
    )
    parser.add_argument(
        "--gguf-outtype",
        default="auto",
        help="llama.cpp --outtype passed to convert_hf_to_gguf.py when --llama-cpp-dir is provided.",
    )
    parser.add_argument(
        "--ollama-model-name",
        default=None,
        help=(
            "Model name used in generated helper scripts and optional `ollama create`. "
            "This name also determines the export folder prefix before the `_ollama` suffix."
        ),
    )
    parser.add_argument(
        "--create-ollama-model",
        action="store_true",
        help="Run `ollama create` automatically after GGUF is available.",
    )
    return parser.parse_args()


def _resolve_converter_path(llama_cpp_dir: str | Path) -> Path:
    candidate = Path(llama_cpp_dir).expanduser().resolve()
    if candidate.is_file():
        if candidate.name != "convert_hf_to_gguf.py":
            raise FileNotFoundError(
                "Expected a llama.cpp converter script named convert_hf_to_gguf.py."
            )
        return candidate
    converter = candidate / "convert_hf_to_gguf.py"
    if converter.is_file():
        return converter
    raise FileNotFoundError(f"Could not find convert_hf_to_gguf.py under: {candidate}")


def _checkpoint_uses_tagged_thinking_prompt(checkpoint: dict[str, Any]) -> bool:
    tokenizer_settings = checkpoint.get("inference_tokenizer_settings")
    if isinstance(tokenizer_settings, dict) and bool(tokenizer_settings.get("add_thinking")):
        return str(tokenizer_settings.get("thinking_template") or "") == "tagged"

    reasoning_settings = checkpoint.get("reasoning_settings")
    return bool(
        isinstance(reasoning_settings, dict)
        and reasoning_settings.get("use_think_tokens")
    )


def _build_ollama_template(checkpoint: dict[str, Any]) -> str:
    template = DEFAULT_OLLAMA_TEMPLATE
    if _checkpoint_uses_tagged_thinking_prompt(checkpoint):
        template = template.replace(
            "<|im_start|>assistant\n",
            "<|im_start|>assistant\n<think>\n",
            1,
        )
    return template


def _format_modelfile_contents(
    *,
    gguf_path: Path,
    checkpoint: dict[str, Any],
    ollama_model_name: str,
) -> str:
    source_ref = Path(gguf_path)
    if source_ref.is_absolute():
        from_value = str(source_ref)
    else:
        from_value = str(source_ref)

    context_length = int(checkpoint["model_config"]["context_length"])
    system_prompt = build_default_ollama_system_prompt(checkpoint)

    lines = [
        f"# Generated for {ollama_model_name}",
        "# This file packages the exported GGUF without changing the repo's Qwen3.5 algorithm.",
        f"FROM {from_value}",
        "",
        f'TEMPLATE """{_build_ollama_template(checkpoint)}"""',
    ]

    if system_prompt:
        lines.extend(["", f'SYSTEM """{system_prompt}"""'])

    if context_length <= 16384:
        lines.extend(["", f"PARAMETER num_ctx {context_length}"])
    else:
        lines.extend(
            [
                "",
                f"# The checkpoint context_length is {context_length}.",
                "# Uncomment and lower this if your Ollama host cannot afford the full context window.",
                f"# PARAMETER num_ctx {context_length}",
            ]
        )

    for stop_token in DEFAULT_OLLAMA_STOP_TOKENS:
        lines.append(f'PARAMETER stop "{stop_token}"')

    return "\n".join(lines) + "\n"


def _write_helper_scripts(
    *,
    output_dir: Path,
    hf_dir: Path,
    gguf_path: Path,
    modelfile_path: Path,
    ollama_model_name: str,
) -> tuple[Path, Path]:
    convert_script_path = output_dir / "convert_to_gguf.ps1"
    convert_script_path.write_text(
        "\n".join(
            [
                "param(",
                '  [string]$LlamaCppDir = "C:\\\\path\\\\to\\\\llama.cpp",',
                '  [string]$OutType = "auto"',
                ")",
                "",
                '$Converter = if ((Test-Path $LlamaCppDir) -and (Get-Item $LlamaCppDir).PSIsContainer) {',
                '  Join-Path $LlamaCppDir "convert_hf_to_gguf.py"',
                "} else {",
                "  $LlamaCppDir",
                "}",
                'if (-not (Test-Path $Converter)) { throw "convert_hf_to_gguf.py not found." }',
                f'python $Converter "{hf_dir}" --outfile "{gguf_path}" --outtype $OutType',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ollama_script_path = output_dir / "create_ollama_model.ps1"
    ollama_script_path.write_text(
        "\n".join(
            [
                "param(",
                f'  [string]$ModelName = "{ollama_model_name}"',
                ")",
                "",
                "ollama show $ModelName *> $null",
                'if ($LASTEXITCODE -eq 0) { throw "Ollama model already exists. Use a different model name to avoid overwrite." }',
                "",
                f'ollama create $ModelName -f "{modelfile_path}"',
                "ollama run $ModelName",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return convert_script_path, ollama_script_path


def _run_command(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, check=True, cwd=str(cwd) if cwd is not None else None)


def _ollama_model_exists(model_name: str) -> bool:
    result = subprocess.run(
        ["ollama", "show", model_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _resolve_verify_model_class() -> tuple[type["Qwen3_5ForCausalLM"] | None, str | None]:
    try:
        from transformers import Qwen3_5ForCausalLM
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}"
    return Qwen3_5ForCausalLM, None


def _build_verification_dependency_hint(detail: str | None) -> str | None:
    if not detail:
        return None

    normalized = detail.lower()
    if "huggingface-hub" in normalized:
        return (
            "Install a compatible verification stack in the active kernel, for example "
            '`uv sync` or `pip install "huggingface_hub<1.0" "transformers>=4.45"`.'
        )
    if "no module named 'transformers'" in normalized or 'no module named "transformers"' in normalized:
        return (
            "Install the optional verification dependency in the active kernel with "
            "`uv sync` or `pip install transformers`."
        )
    return (
        "Install a compatible `transformers` stack in the active Python environment, "
        "or rerun the export without `--verify`."
    )


def _verify_export(
    checkpoint: dict[str, Any],
    *,
    checkpoint_path: Path,
    tokenizer_path: str | Path | None,
    seq_len: int,
    atol: float,
) -> dict[str, Any]:
    qwen3_5_for_causal_lm_cls, dependency_error = _resolve_verify_model_class()
    if qwen3_5_for_causal_lm_cls is None:
        result = {
            "verified": False,
            "verification_skipped": True,
            "reason": "transformers_unavailable",
            "detail": dependency_error,
        }
        hint = _build_verification_dependency_hint(dependency_error)
        if hint is not None:
            result["hint"] = hint
        return result

    if seq_len <= 0:
        raise ValueError("--verify-seq-len must be positive.")

    model_config = checkpoint["model_config"]
    effective_seq_len = min(seq_len, int(model_config["context_length"]))
    if effective_seq_len <= 0:
        raise ValueError("The checkpoint context_length must be positive.")

    repo_model = build_model(model_config)
    repo_model.load_state_dict(checkpoint["model_state_dict"])
    repo_model.eval()

    resolved_tokenizer_path, tokenizer_repo_id = resolve_checkpoint_tokenizer(
        checkpoint,
        checkpoint_path,
        tokenizer_path=tokenizer_path,
        project_root=PROJECT_ROOT,
    )
    export_tokenizer = load_export_tokenizer(
        resolved_tokenizer_path,
        repo_id=tokenizer_repo_id,
    )

    hf_config = build_hf_text_config(
        model_config,
        pad_token_id=export_tokenizer.pad_token_id,
        eos_token_id=export_tokenizer.eos_token_id,
    )
    hf_model = qwen3_5_for_causal_lm_cls(hf_config)
    hf_model.load_state_dict(
        convert_repo_state_dict_to_hf(
            checkpoint["model_state_dict"],
            model_config,
        )
    )
    hf_model.eval()

    vocab_size = int(model_config["vocab_size"])
    example_input = (
        torch.arange(effective_seq_len, dtype=torch.long).unsqueeze(0).remainder(vocab_size)
    )

    with torch.no_grad():
        repo_logits = repo_model(example_input)
        hf_logits = hf_model(example_input).logits

    max_abs_diff = float((repo_logits - hf_logits).abs().max().item())
    if not torch.allclose(repo_logits, hf_logits, atol=atol, rtol=0.0):
        raise RuntimeError(
            "HF export verification failed. "
            f"max_abs_diff={max_abs_diff:.6g}, atol={atol:.6g}"
        )

    return {
        "verified": True,
        "verification_skipped": False,
        "max_abs_diff": max_abs_diff,
        "seq_len": effective_seq_len,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    ollama_model_name = args.ollama_model_name or _default_ollama_model_name(checkpoint_path)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_dir = _resolve_unique_output_dir(output_root, ollama_model_name)
    output_dir.mkdir(parents=True, exist_ok=False)

    hf_dir = output_dir / "hf"

    artifact = export_checkpoint_to_hf_directory(
        checkpoint_path,
        hf_dir,
        tokenizer_path=args.tokenizer,
        project_root=PROJECT_ROOT,
        weights_format=args.weights_format,
    )

    verification: dict[str, Any] = {
        "verified": False,
        "verification_skipped": not args.verify,
        "reason": "verify_flag_not_set",
    }
    if args.verify:
        verification = _verify_export(
            checkpoint,
            checkpoint_path=checkpoint_path,
            tokenizer_path=args.tokenizer,
            seq_len=args.verify_seq_len,
            atol=args.verify_atol,
        )

    gguf_output = (
        _resolve_unique_file_path(Path(args.gguf_output))
        if args.gguf_output is not None
        else (output_dir / "model.gguf").resolve()
    )

    converter_command: list[str] | None = None
    if args.llama_cpp_dir:
        converter_path = _resolve_converter_path(args.llama_cpp_dir)
        converter_command = [
            sys.executable,
            str(converter_path),
            str(artifact.model_dir),
            "--outfile",
            str(gguf_output),
            "--outtype",
            str(args.gguf_outtype),
        ]
        _run_command(converter_command)

    modelfile_path = output_dir / "Modelfile"
    modelfile_contents = _format_modelfile_contents(
        gguf_path=Path("./model.gguf") if gguf_output.parent == output_dir else gguf_output,
        checkpoint=checkpoint,
        ollama_model_name=ollama_model_name,
    )
    modelfile_path.write_text(modelfile_contents, encoding="utf-8")

    convert_script_path, ollama_script_path = _write_helper_scripts(
        output_dir=output_dir,
        hf_dir=artifact.model_dir,
        gguf_path=gguf_output,
        modelfile_path=modelfile_path,
        ollama_model_name=ollama_model_name,
    )

    ollama_create_command: list[str] | None = None
    if args.create_ollama_model:
        if not gguf_output.is_file():
            raise FileNotFoundError(
                "GGUF file is required before `--create-ollama-model`. "
                "Provide --llama-cpp-dir or generate the GGUF first."
            )
        if shutil.which("ollama") is None:
            raise RuntimeError(
                "Could not find `ollama` on PATH. Install Ollama before using --create-ollama-model."
            )
        if _ollama_model_exists(ollama_model_name):
            raise FileExistsError(
                "Ollama model already exists. "
                f"Use a different --ollama-model-name to avoid overwrite: {ollama_model_name}"
            )
        ollama_create_command = [
            "ollama",
            "create",
            ollama_model_name,
            "-f",
            str(modelfile_path),
        ]
        _run_command(ollama_create_command)

    summary_path = output_dir / "ollama_export_summary.json"
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "output_root": str(output_root),
        "output_dir": str(output_dir),
        "hf_dir": str(artifact.model_dir),
        "weights_path": str(artifact.weights_path),
        "config_path": str(artifact.config_path),
        "generation_config_path": str(artifact.generation_config_path),
        "tokenizer_path": str(artifact.tokenizer_path),
        "metadata_path": str(artifact.metadata_path),
        "modelfile_path": str(modelfile_path),
        "gguf_output": str(gguf_output),
        "ollama_model_name": ollama_model_name,
        "verification": verification,
        "converter_command": converter_command,
        "ollama_create_command": ollama_create_command,
        "helper_scripts": {
            "convert_to_gguf": str(convert_script_path),
            "create_ollama_model": str(ollama_script_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_safe),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_safe))


if __name__ == "__main__":
    main()

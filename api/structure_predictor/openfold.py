from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import OpenFoldSettings, load_config


VALID_PROTEIN_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYX")
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class StructurePredictionRequest:
    sequence: str
    name: str = "candidate"
    output_format: str | None = None
    config_preset: str | None = None
    include_structure_text: bool | None = None
    job_id: str | None = None


@dataclass(frozen=True)
class OpenFoldPredictionResult:
    job_id: str
    name: str
    sequence_length: int
    structure_format: str
    structure_path: str
    structure_text: str | None
    command: tuple[str, ...]
    returncode: int
    stdout_tail: str
    stderr_tail: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "sequence_length": self.sequence_length,
            "structure_format": self.structure_format,
            "structure_path": self.structure_path,
            "structure_text": self.structure_text,
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "metadata": self.metadata,
        }


class OpenFoldRunner:
    def __init__(self, settings: OpenFoldSettings) -> None:
        self.settings = settings

    def readiness(self) -> dict[str, Any]:
        script_path = self.settings.repo_path / "run_pretrained_openfold.py"
        return {
            "repo_path": str(self.settings.repo_path),
            "script_path": str(script_path),
            "script_exists": script_path.is_file(),
            "template_mmcif_dir": str(self.settings.template_mmcif_dir),
            "template_mmcif_dir_exists": self.settings.template_mmcif_dir.is_dir(),
            "output_root": str(self.settings.output_root),
            "config_preset": self.settings.config_preset,
            "model_device": self.settings.model_device,
            "output_format": self.settings.output_format,
            "use_precomputed_alignments": self.settings.use_precomputed_alignments,
            "precomputed_alignments_dir": (
                str(self.settings.precomputed_alignments_dir)
                if self.settings.precomputed_alignments_dir
                else None
            ),
            "openfold_checkpoint_path": (
                str(self.settings.openfold_checkpoint_path)
                if self.settings.openfold_checkpoint_path
                else None
            ),
        }

    def predict(self, request: StructurePredictionRequest) -> OpenFoldPredictionResult:
        sequence = normalize_protein_sequence(
            request.sequence,
            min_length=self.settings.min_sequence_length,
            max_length=self.settings.max_sequence_length,
        )
        output_format = _normalize_output_format(
            request.output_format or self.settings.output_format
        )
        name = sanitize_name(request.name or "candidate")
        job_id = sanitize_name(request.job_id or uuid.uuid4().hex)
        job_root = self.settings.output_root / job_id
        fasta_dir = job_root / "fasta"
        output_dir = job_root / "openfold"

        fasta_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_fasta(fasta_dir / f"{name}.fasta", name=name, sequence=sequence)

        command = build_openfold_command(
            self.settings,
            fasta_dir=fasta_dir,
            output_dir=output_dir,
            output_format=output_format,
            config_preset=request.config_preset or self.settings.config_preset,
        )
        completed = subprocess.run(
            command,
            cwd=self.settings.repo_path,
            capture_output=True,
            text=True,
            timeout=self.settings.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "OpenFold inference failed with return code "
                f"{completed.returncode}.\n{_tail_text(completed.stderr)}"
            )

        openfold_structure_path = find_structure_output(output_dir, output_format=output_format)
        structure_path = _stage_structure_output(
            openfold_structure_path,
            job_root=job_root,
            output_format=output_format,
        )
        include_text = (
            self.settings.include_structure_text
            if request.include_structure_text is None
            else bool(request.include_structure_text)
        )
        structure_text = (
            _read_structure_text(
                structure_path,
                max_bytes=self.settings.max_response_structure_bytes,
            )
            if include_text
            else None
        )
        metadata = {
            "fasta_dir": str(fasta_dir),
            "output_dir": str(output_dir),
            "openfold_structure_path": str(openfold_structure_path),
            "config_preset": request.config_preset or self.settings.config_preset,
            "model_device": self.settings.model_device,
        }
        return OpenFoldPredictionResult(
            job_id=job_id,
            name=name,
            sequence_length=len(sequence),
            structure_format=output_format,
            structure_path=str(structure_path),
            structure_text=structure_text,
            command=tuple(command),
            returncode=completed.returncode,
            stdout_tail=_tail_text(completed.stdout),
            stderr_tail=_tail_text(completed.stderr),
            metadata=metadata,
        )


def build_openfold_command(
    settings: OpenFoldSettings,
    *,
    fasta_dir: Path,
    output_dir: Path,
    output_format: str,
    config_preset: str,
) -> list[str]:
    command = [
        settings.python_executable,
        str(settings.repo_path / "run_pretrained_openfold.py"),
        str(fasta_dir),
        str(settings.template_mmcif_dir),
        "--output_dir",
        str(output_dir),
        "--config_preset",
        str(config_preset),
        "--model_device",
        settings.model_device,
        "--cpus",
        str(settings.cpus),
    ]
    if output_format == "cif":
        command.append("--cif_output")
    if settings.use_precomputed_alignments:
        if settings.precomputed_alignments_dir is None:
            raise ValueError(
                "use_precomputed_alignments is enabled but precomputed_alignments_dir is missing."
            )
        command.extend(["--use_precomputed_alignments", str(settings.precomputed_alignments_dir)])
    else:
        command.extend(_path_args(settings.database_paths))
    command.extend(_path_args(settings.binary_paths))
    if settings.openfold_checkpoint_path is not None:
        command.extend(["--openfold_checkpoint_path", str(settings.openfold_checkpoint_path)])
    if settings.jax_param_path is not None:
        command.extend(["--jax_param_path", str(settings.jax_param_path)])
    if settings.data_random_seed is not None:
        command.extend(["--data_random_seed", str(settings.data_random_seed)])
    if settings.skip_relaxation:
        command.append("--skip_relaxation")
    if settings.long_sequence_inference:
        command.append("--long_sequence_inference")
    if settings.use_single_seq_mode:
        command.append("--use_single_seq_mode")
    command.extend(settings.extra_args)
    return command


def normalize_protein_sequence(
    sequence: str,
    *,
    min_length: int,
    max_length: int,
) -> str:
    normalized = "".join(str(sequence or "").upper().split())
    if len(normalized) < min_length:
        raise ValueError(f"Protein sequence must be at least {min_length} residues.")
    if len(normalized) > max_length:
        raise ValueError(f"Protein sequence exceeds the configured limit of {max_length} residues.")

    invalid = sorted({residue for residue in normalized if residue not in VALID_PROTEIN_AMINO_ACIDS})
    if invalid:
        raise ValueError(f"Protein sequence contains invalid residues: {', '.join(invalid)}")
    return normalized


def sanitize_name(value: str) -> str:
    normalized = _SAFE_NAME_PATTERN.sub("_", str(value).strip())
    normalized = normalized.strip("._-")
    return normalized[:80] or "candidate"


def find_structure_output(output_dir: Path, *, output_format: str) -> Path:
    suffix = ".cif" if output_format == "cif" else ".pdb"
    candidates = [path for path in output_dir.rglob(f"*{suffix}") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"OpenFold completed but no `{suffix}` structure file was found under {output_dir}."
        )

    def sort_key(path: Path) -> tuple[int, int, float, str]:
        name = path.name.lower()
        is_predictions = 1 if path.parent.name == "predictions" else 0
        is_relaxed = 1 if "relaxed" in name and "unrelaxed" not in name else 0
        return (is_predictions, is_relaxed, path.stat().st_mtime, path.name)

    return max(candidates, key=sort_key).resolve()


def _path_args(paths: dict[str, Path]) -> list[str]:
    args: list[str] = []
    for key in sorted(paths):
        option = key if key.startswith("--") else f"--{key}"
        args.extend([option, str(paths[key])])
    return args


def _write_fasta(path: Path, *, name: str, sequence: str) -> None:
    wrapped = "\n".join(sequence[index : index + 80] for index in range(0, len(sequence), 80))
    path.write_text(f">{name}\n{wrapped}\n", encoding="utf-8")


def _normalize_output_format(value: str) -> str:
    normalized = str(value).lower().strip()
    if normalized not in {"pdb", "cif"}:
        raise ValueError("output_format must be `pdb` or `cif`.")
    return normalized


def _read_structure_text(path: Path, *, max_bytes: int) -> str | None:
    if path.stat().st_size > max_bytes:
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _stage_structure_output(path: Path, *, job_root: Path, output_format: str) -> Path:
    stable_path = job_root / f"structure.{output_format}"
    if path.resolve() != stable_path.resolve():
        shutil.copyfile(path, stable_path)
    return stable_path.resolve()


def _tail_text(value: str, *, max_chars: int = 4000) -> str:
    if len(value) <= max_chars:
        return value
    return value[-max_chars:]


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run OpenFold structure prediction through the MDNAC structure API wrapper.",
    )
    parser.add_argument("--config", default=None, help="Path to api/config.structure.yaml.")
    parser.add_argument("--env", default=None, help="Environment name from config.structure.yaml.")
    parser.add_argument("--sequence", required=True, help="Protein sequence to fold.")
    parser.add_argument("--name", default="candidate", help="FASTA record name.")
    parser.add_argument("--job-id", default=None, help="Stable output job ID.")
    parser.add_argument("--output-format", choices=("pdb", "cif"), default=None)
    parser.add_argument("--config-preset", default=None, help="OpenFold config preset override.")
    parser.add_argument("--no-structure-text", action="store_true")
    args = parser.parse_args()

    settings = load_config(config_path=args.config, environment=args.env)
    result = OpenFoldRunner(settings.openfold).predict(
        StructurePredictionRequest(
            sequence=args.sequence,
            name=args.name,
            job_id=args.job_id,
            output_format=args.output_format,
            config_preset=args.config_preset,
            include_structure_text=not args.no_structure_text,
        )
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

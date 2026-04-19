from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONCAT_BUFFER_SIZE = 1024 * 1024


@dataclass(slots=True)
class ConcatenateTextFilesSummary:
    output_path: str
    source_paths: tuple[str, ...]
    source_count: int
    source_bytes: int
    output_bytes: int
    inserted_separator_newlines: int
    ensure_line_boundary: bool
    overwrite: bool


def concatenate_text_files(
    input_paths: Sequence[str | os.PathLike[str]],
    *,
    output_path: str | os.PathLike[str],
    overwrite: bool = False,
    ensure_line_boundary: bool = True,
    buffer_size: int = DEFAULT_CONCAT_BUFFER_SIZE,
) -> ConcatenateTextFilesSummary:
    if isinstance(input_paths, (str, os.PathLike)):
        raise TypeError("input_paths must be a sequence of paths, not a single path.")
    if len(input_paths) < 2:
        raise ValueError("At least two input files are required.")
    if buffer_size <= 0:
        raise ValueError("buffer_size must be greater than 0.")

    source_paths = tuple(Path(path) for path in input_paths)
    resolved_output_path = Path(output_path)

    for source_path in source_paths:
        if not source_path.is_file():
            raise FileNotFoundError(f"Input file was not found: {source_path}")
        if source_path.resolve() == resolved_output_path.resolve(strict=False):
            raise ValueError(f"output_path must be different from input file: {source_path}")

    if resolved_output_path.exists():
        if resolved_output_path.is_dir():
            raise IsADirectoryError(f"Output path is a directory: {resolved_output_path}")
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output file: {resolved_output_path}")

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    source_bytes = 0
    output_bytes = 0
    inserted_separator_newlines = 0
    output_has_content = False
    output_ends_with_lf = True

    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            delete=False,
            dir=resolved_output_path.parent,
            prefix=f".{resolved_output_path.name}.",
            suffix=".tmp",
        ) as target:
            temp_path = Path(target.name)
            for source_path in source_paths:
                with source_path.open("rb") as source:
                    first_chunk = source.read(buffer_size)
                    if not first_chunk:
                        continue

                    if ensure_line_boundary and output_has_content and not output_ends_with_lf:
                        target.write(b"\n")
                        output_bytes += 1
                        inserted_separator_newlines += 1
                        output_ends_with_lf = True

                    target.write(first_chunk)
                    source_bytes += len(first_chunk)
                    output_bytes += len(first_chunk)
                    output_has_content = True
                    output_ends_with_lf = first_chunk.endswith(b"\n")

                    while True:
                        chunk = source.read(buffer_size)
                        if not chunk:
                            break
                        target.write(chunk)
                        source_bytes += len(chunk)
                        output_bytes += len(chunk)
                        output_ends_with_lf = chunk.endswith(b"\n")

        temp_path.replace(resolved_output_path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    return ConcatenateTextFilesSummary(
        output_path=str(resolved_output_path),
        source_paths=tuple(str(path) for path in source_paths),
        source_count=len(source_paths),
        source_bytes=source_bytes,
        output_bytes=output_bytes,
        inserted_separator_newlines=inserted_separator_newlines,
        ensure_line_boundary=ensure_line_boundary,
        overwrite=overwrite,
    )

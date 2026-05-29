"""Slim instruction.jsonl by removing unnecessary fields.

This temp script reads the original instruction.jsonl (streaming, line-by-line)
and writes a slimmed version that keeps only the fields needed for:
  - Training (instruction, input, output)
  - KB search (accession, organism, product, taxonomy, keywords)
  - Profile search (derived_labels)

Fields removed:
  - description           (already encoded in `instruction`)
  - metadata.description  (duplicate)
  - metadata.fasta_header (duplicate of accession + description)
  - metadata.dataset_group (pipeline internal)
  - metadata.dataset_bundle (pipeline internal)
  - metadata.source_name  (pipeline internal)
  - label_source          (pipeline internal)
  - origin                (always "paired")
  - output_format         (always "single protein sequence")
  - derived_keywords      (not in keep list)

The metadata sub-fields product, taxonomy, and keywords are promoted to
top-level fields for a flatter, more compact structure.

Usage:
    python temp/slim_instruction_jsonl.py [input_path] [-o output_path]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# Top-level fields to keep as-is
KEEP_TOP_LEVEL = frozenset({
    "instruction",
    "input",
    "output",
    "accession",
    "organism",
    "derived_labels",
})

# Metadata sub-fields to promote to top-level
METADATA_PROMOTE = frozenset({
    "product",
    "taxonomy",
    "keywords",
})

# Everything else (top-level or metadata) is dropped.


def slim_record(payload: dict) -> dict:
    """Return a slimmed copy of a single JSONL record."""
    slimmed: dict = {}

    # 1) Keep selected top-level fields (preserve original order)
    for key in (
        "instruction",
        "input",
        "output",
        "accession",
        "organism",
    ):
        if key in payload:
            slimmed[key] = payload[key]

    # 2) Promote metadata sub-fields to top-level
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for meta_key in ("product", "taxonomy", "keywords"):
            if meta_key in metadata:
                slimmed[meta_key] = metadata[meta_key]

    # 3) Also check for top-level product/taxonomy/keywords (fallback)
    for promoted_key in ("product", "taxonomy", "keywords"):
        if promoted_key not in slimmed and promoted_key in payload:
            slimmed[promoted_key] = payload[promoted_key]

    # 4) Keep derived_labels
    if "derived_labels" in payload:
        slimmed["derived_labels"] = payload["derived_labels"]

    return slimmed


def slim_instruction_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Stream-process the instruction.jsonl file and write a slimmed version."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_size = input_path.stat().st_size
    total_lines = 0
    written_lines = 0
    error_lines = 0
    start_time = time.time()
    last_report_time = start_time
    bytes_read = 0

    with input_path.open("r", encoding="utf-8") as source, \
         output_path.open("w", encoding="utf-8") as target:

        for line_number, raw_line in enumerate(source, start=1):
            bytes_read += len(raw_line.encode("utf-8"))

            stripped = raw_line.strip()
            if not stripped:
                continue

            total_lines += 1

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                error_lines += 1
                continue

            if not isinstance(payload, dict):
                error_lines += 1
                continue

            slimmed = slim_record(payload)
            target.write(json.dumps(slimmed, ensure_ascii=False))
            target.write("\n")
            written_lines += 1

            # Progress report every 30 seconds
            now = time.time()
            if now - last_report_time >= 30.0:
                elapsed = now - start_time
                pct = (bytes_read / input_size * 100) if input_size > 0 else 0
                rate = written_lines / elapsed if elapsed > 0 else 0
                print(
                    f"  [{elapsed:.0f}s] {written_lines:,} lines written "
                    f"({pct:.1f}% of input, {rate:.0f} lines/s)",
                    flush=True,
                )
                last_report_time = now

    elapsed = time.time() - start_time
    output_size = output_path.stat().st_size
    reduction_pct = (1 - output_size / input_size) * 100 if input_size > 0 else 0

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_size_bytes": input_size,
        "output_size_bytes": output_size,
        "reduction_percent": round(reduction_pct, 2),
        "total_lines": total_lines,
        "written_lines": written_lines,
        "error_lines": error_lines,
        "elapsed_seconds": round(elapsed, 2),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Slim instruction.jsonl by removing unnecessary fields for S3 upload.",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=Path("instruction.jsonl"),
        help="Source instruction.jsonl file. Default: instruction.jsonl",
    )
    parser.add_argument(
        "-o", "--output-path",
        type=Path,
        default=None,
        help="Output slimmed JSONL path. Default: <input>.slim.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path: Path = args.input_path
    output_path: Path = args.output_path
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}.slim{input_path.suffix}")

    print(f"[slim] Input:  {input_path}")
    print(f"[slim] Output: {output_path}")
    print(f"[slim] Input size: {input_path.stat().st_size / (1024**3):.2f} GB")
    print(f"[slim] Processing...", flush=True)

    summary = slim_instruction_jsonl(
        input_path,
        output_path,
        overwrite=args.overwrite,
    )

    print()
    print(f"[slim] Done!")
    print(f"[slim] Input size:      {summary['input_size_bytes'] / (1024**3):.2f} GB")
    print(f"[slim] Output size:     {summary['output_size_bytes'] / (1024**3):.2f} GB")
    print(f"[slim] Reduction:       {summary['reduction_percent']:.1f}%")
    print(f"[slim] Total lines:     {summary['total_lines']:,}")
    print(f"[slim] Written lines:   {summary['written_lines']:,}")
    print(f"[slim] Error lines:     {summary['error_lines']:,}")
    print(f"[slim] Elapsed:         {summary['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

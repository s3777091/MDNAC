from __future__ import annotations

import argparse
from pathlib import Path

from libs.core.pretrain.file_concat import concatenate_text_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate multiple line-oriented text files into one output file. "
            "The implementation streams bytes, does not parse records, and does not remove duplicates."
        ),
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        type=Path,
        help="Input files to concatenate in the exact order provided.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        required=True,
        help="Destination file path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Do exact byte concatenation without inserting separator newlines between files.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if len(args.input_paths) < 2:
        parser.error("At least two input files are required.")

    summary = concatenate_text_files(
        args.input_paths,
        output_path=args.output_path,
        overwrite=args.overwrite,
        ensure_line_boundary=not args.raw,
    )
    print(f"[output] {summary.output_path}")
    for index, source_path in enumerate(summary.source_paths, 1):
        print(f"[source_{index}] {source_path}")
    print(f"[source_count] {summary.source_count}")
    print(f"[source_bytes] {summary.source_bytes}")
    print(f"[output_bytes] {summary.output_bytes}")
    print(f"[separator_newlines] {summary.inserted_separator_newlines}")
    print(f"[line_boundary] {summary.ensure_line_boundary}")
    print(f"[overwrite] {summary.overwrite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

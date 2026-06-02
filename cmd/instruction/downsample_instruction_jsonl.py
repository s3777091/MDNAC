from __future__ import annotations

import argparse
from pathlib import Path

from libs.core.pretrain.instruction_downsample import (
    DEFAULT_KEEP_RATIO,
    DEFAULT_SUBLINEAR_ALPHA,
    downsample_instruction_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Downsample instruction.jsonl with a two-pass streaming sampler. "
            "Coverage is preserved per dataset_group/product bucket while heavily repeated protein buckets "
            "are compressed with sublinear quotas."
        ),
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=Path("data/instruction.jsonl"),
        help="Source instruction.jsonl file.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=None,
        help="Destination JSONL path. Default: <input>.downsampled.jsonl",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=DEFAULT_KEEP_RATIO,
        help="Fraction of lines to keep. Default: 0.5",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_SUBLINEAR_ALPHA,
        help=(
            "Sublinear exponent for per-protein quotas. "
            "1.0 is proportional sampling; lower values protect rarer protein buckets more. Default: 0.8"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute targets and quotas without writing the output file.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_path = args.output_path
    if output_path is None:
        output_path = args.input_path.with_name(f"{args.input_path.stem}.downsampled{args.input_path.suffix}")

    summary = downsample_instruction_jsonl(
        args.input_path,
        output_path=output_path,
        keep_ratio=args.keep_ratio,
        alpha=args.alpha,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"[input] {summary.input_path}")
    print(f"[output] {summary.output_path}")
    print(f"[dry_run] {args.dry_run}")
    print(f"[keep_ratio] {summary.keep_ratio}")
    print(f"[alpha] {summary.alpha}")
    print(f"[dataset_groups] {summary.dataset_group_count}")
    print(f"[protein_buckets] {summary.unique_stratum_count}")
    print(f"[input_lines] {summary.total_line_count}")
    print(f"[target_lines] {summary.target_line_count}")
    print(f"[written_lines] {summary.written_line_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

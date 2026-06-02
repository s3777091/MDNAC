from __future__ import annotations

import argparse
from pathlib import Path

from libs.core.pretrain.refseq_local import dedupe_local_refseq_sequence_only_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove duplicate entries from sequence-only RefSeq artifacts in-place. "
            "Both train.txt and instruction.jsonl are deduplicated by normalized non-empty line "
            "while preserving the first occurrence order."
        ),
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("data/compiled/refseq_bacteria_protein"),
        help="Directory containing train.txt and instruction.jsonl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without rewriting any files.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    summary = dedupe_local_refseq_sequence_only_artifacts(
        args.output_dir,
        dry_run=args.dry_run,
    )
    print(f"[output] {summary.output_dir}")
    print(f"[train.txt] {summary.train_text_path}")
    print(f"[instruction.jsonl] {summary.instruction_path}")
    print(f"[dry_run] {summary.dry_run}")
    print(f"[train_original] {summary.original_train_line_count}")
    print(f"[train_deduped] {summary.deduped_train_line_count}")
    print(f"[train_removed] {summary.removed_train_duplicates}")
    print(f"[instruction_original] {summary.original_instruction_line_count}")
    print(f"[instruction_deduped] {summary.deduped_instruction_line_count}")
    print(f"[instruction_removed] {summary.removed_instruction_duplicates}")
    print(f"[train_changed] {summary.train_text_changed}")
    print(f"[instruction_changed] {summary.instruction_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

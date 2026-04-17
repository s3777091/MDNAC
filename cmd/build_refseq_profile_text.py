from __future__ import annotations

import argparse
from pathlib import Path

from libs.core.pretrain.refseq_local import build_local_refseq_profile_text_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile local RefSeq .gpff.gz/.faa.gz archives into sequence-only protein pretrain "
            "artifacts (train.txt + tokenizer_map.json) and instruction.jsonl for metadata-to-protein tuning. "
            "Existing train.txt and instruction.jsonl files are appended instead of being diffed or rewritten."
        ),
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        type=Path,
        default=Path("data/raw/refseq_bacteria_protein"),
        help="Directory containing local RefSeq .gpff.gz and .faa.gz files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data/compiled/refseq_bacteria_protein"),
        help=(
            "Directory where train.txt, tokenizer_map.json, and instruction.jsonl will be written. "
            "If the output folder name matches a direct child folder under input_root, only that child folder "
            "will be compiled."
        ),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=256,
        help="Target BPE vocabulary size for the sequence-only protein tokenizer.",
    )
    parser.add_argument(
        "--profile-vocab-size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--profile-sample-char-limit",
        type=int,
        default=2_000_000,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--kmer-size",
        type=int,
        default=3,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--instruction-min-proteins",
        type=int,
        default=10,
        help="Minimum proteins required for a derived condition to count toward instruction coverage stats.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for the number of kept records. Useful for smoke tests.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU worker processes for record compilation and instruction rendering. Use 0 to auto-detect.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help=(
            "Skip writing specific artifacts. Accepts comma-separated values and can be repeated. "
            "Supported values: train, train.txt, tokenizer_map.json, instruction.jsonl."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    summary = build_local_refseq_profile_text_artifacts(
        args.input_root,
        args.output_dir,
        vocab_size=args.vocab_size if args.profile_vocab_size is None else args.profile_vocab_size,
        instruction_min_proteins=args.instruction_min_proteins,
        kmer_size=args.kmer_size,
        profile_vocab_size=args.profile_vocab_size,
        profile_sample_char_limit=args.profile_sample_char_limit,
        max_records=args.max_records,
        workers=args.workers,
        skip_artifacts=args.skip,
    )
    print(f"[input] {summary.input_root}")
    print(f"[output] {summary.output_dir}")
    print(f"[train.txt] {summary.train_text_path}")
    print(f"[tokenizer_map.json] {summary.tokenizer_map_path}")
    print(f"[instruction.jsonl] {summary.instruction_path}")
    print(f"[source_records] {summary.source_record_count}")
    print(f"[records] {summary.record_count}")
    print(f"[instruction_records] {summary.instruction_record_count}")
    print(f"[instruction_conditions] {summary.instruction_condition_count}")
    print(f"[skipped_instruction_conditions] {summary.skipped_instruction_condition_count}")
    print(f"[duplicate_accessions] {summary.duplicate_accession_count}")
    print(f"[duplicate_sequences] {summary.duplicate_sequence_count}")
    print(f"[paired_records] {summary.paired_record_count}")
    print(f"[gpff_only_records] {summary.gpff_only_record_count}")
    print(f"[faa_only_records] {summary.faa_only_record_count}")
    print(f"[truncated_inputs] {summary.truncated_input_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

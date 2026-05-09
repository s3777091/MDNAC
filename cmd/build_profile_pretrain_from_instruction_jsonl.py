from __future__ import annotations

import argparse
from pathlib import Path

from libs.core.pretrain import save_mdc_profile_sequence_pretrain_from_instruction_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build profile-aware MDC pretrain artifacts from instruction.jsonl. "
            "Each JSONL record is treated as instruction/input conditioning text with output as the protein target."
        ),
    )
    parser.add_argument(
        "instruction_jsonl",
        type=Path,
        help="Source instruction.jsonl file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where profile-aware train.txt and tokenizer_map.json will be written.",
    )
    parser.add_argument(
        "--profile-vocab-size",
        type=int,
        default=256,
        help="Target BPE vocabulary size for profile/instruction text.",
    )
    parser.add_argument(
        "--kmer-size",
        type=int,
        default=3,
        help="K-mer size for protein sequence targets.",
    )
    parser.add_argument(
        "--sequence-type",
        default="protein",
        help="Default sequence type when a JSONL record does not declare one.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    artifact = save_mdc_profile_sequence_pretrain_from_instruction_jsonl(
        args.instruction_jsonl,
        args.output_dir,
        default_sequence_type=args.sequence_type,
        kmer_size=args.kmer_size,
        profile_vocab_size=args.profile_vocab_size,
    )

    print(f"[instruction.jsonl] {args.instruction_jsonl}")
    print(f"[output] {artifact.output_dir}")
    print(f"[train.txt] {artifact.train_text_path}")
    print(f"[tokenizer_map.json] {artifact.tokenizer_map_path}")
    print(f"[records] {artifact.record_count}")
    print(f"[sequence_type] {artifact.sequence_type}")
    print(f"[profile_vocab_size] {artifact.profile_vocab_size}")
    print(f"[sequence_vocab_size] {artifact.sequence_vocab_size}")
    print(f"[kmer_size] {artifact.kmer_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

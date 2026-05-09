from __future__ import annotations

import argparse
import sys
from pathlib import Path

from libs.core.pretrain.refseq_local import (
    build_local_refseq_profile_text_artifacts,
    rebuild_local_refseq_tokenizer_map_from_train_text,
)


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
        "--tokenizer-train-line-limit",
        type=int,
        default=None,
        help=(
            "Optional non-empty train.txt line limit for fitting tokenizer_map.json. "
            "By default the tokenizer is fit from the full train.txt using disk-backed streaming."
        ),
    )
    parser.add_argument(
        "--no-tokenizer-progress",
        action="store_true",
        help="Disable tokenizer build progress logs.",
    )
    parser.add_argument(
        "--no-tokenizer-resume",
        action="store_true",
        help=(
            "Disable resumable tokenizer cache checkpoints. By default tokenizer builds can resume from "
            "the last completed BPE merge after rerunning the same command."
        ),
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
            "Supported values: train, train.txt, tokenizer_map.json, instruction.jsonl. "
            "Use --rebuild-tokenizer-map-from-train when you only need tokenizer_map.json from an "
            "existing train.txt."
        ),
    )
    parser.add_argument(
        "--rebuild-tokenizer-map-from-train",
        action="store_true",
        help=(
            "Ignore RefSeq input archives and rebuild tokenizer_map.json only from the existing "
            "train.txt under --output-dir."
        ),
    )
    return parser


def _skip_mentions_tokenizer_map(skip_values: list[str]) -> bool:
    for raw_value in skip_values:
        for token in str(raw_value).split(","):
            normalized = token.strip().lower()
            if normalized in {"tokenizer_map", "tokenizer", "tokenizer_map.json"}:
                return True
    return False


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    effective_vocab_size = args.vocab_size if args.profile_vocab_size is None else args.profile_vocab_size
    tokenizer_progress_callback = None if args.no_tokenizer_progress else _build_tokenizer_progress_reporter()

    if args.rebuild_tokenizer_map_from_train:
        if _skip_mentions_tokenizer_map(args.skip):
            parser.error("--rebuild-tokenizer-map-from-train cannot be combined with --skip tokenizer_map.json.")

        summary = rebuild_local_refseq_tokenizer_map_from_train_text(
            args.output_dir,
            vocab_size=effective_vocab_size,
            tokenizer_train_line_limit=args.tokenizer_train_line_limit,
            tokenizer_resume=not args.no_tokenizer_resume,
            tokenizer_progress_callback=tokenizer_progress_callback,
        )
        print("[mode] rebuild-tokenizer-map-from-train")
        print(f"[output] {summary.output_dir}")
        print(f"[train.txt] {summary.train_text_path}")
        print(f"[tokenizer_map.json] {summary.tokenizer_map_path}")
        print(f"[records] {summary.record_count}")
        print(f"[tokenizer_train_records] {summary.tokenizer_train_record_count}")
        print(f"[vocab_size_requested] {summary.vocab_size_requested}")
        return 0

    summary = build_local_refseq_profile_text_artifacts(
        args.input_root,
        args.output_dir,
        vocab_size=effective_vocab_size,
        instruction_min_proteins=args.instruction_min_proteins,
        kmer_size=args.kmer_size,
        profile_vocab_size=args.profile_vocab_size,
        profile_sample_char_limit=args.profile_sample_char_limit,
        max_records=args.max_records,
        workers=args.workers,
        skip_artifacts=args.skip,
        tokenizer_train_line_limit=args.tokenizer_train_line_limit,
        tokenizer_resume=not args.no_tokenizer_resume,
        tokenizer_progress_callback=tokenizer_progress_callback,
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


def _build_tokenizer_progress_reporter():
    def report(event: dict[str, object]) -> None:
        event_name = str(event.get("event", ""))
        if event_name == "tokenizer_resume_loaded":
            print(
                f"[tokenizer] resume loaded merges={event.get('completed_merges')} "
                f"vocab={event.get('vocab_size')}/{event.get('target_vocab_size')} "
                f"cache={_format_bytes(event.get('cache_bytes'))}",
                file=sys.stderr,
                flush=True,
            )
        elif event_name == "tokenizer_resume_ignored":
            print(
                f"[tokenizer] resume ignored reason={event.get('reason')}",
                file=sys.stderr,
                flush=True,
            )
        elif event_name == "tokenizer_checkpoint_saved":
            print(
                f"[tokenizer] checkpoint merges={event.get('completed_merges')} "
                f"vocab={event.get('vocab_size')}/{event.get('target_vocab_size')} "
                f"cache={_format_bytes(event.get('cache_bytes'))}",
                file=sys.stderr,
                flush=True,
            )
        elif event_name == "token_cache_start":
            print(
                f"[tokenizer] cache start total={_format_bytes(event.get('total_bytes'))} "
                f"line_limit={event.get('line_limit')}",
                file=sys.stderr,
                flush=True,
            )
        elif event_name == "token_cache_progress":
            print(
                f"[tokenizer] cache {_format_percent(event)} "
                f"records={event.get('records_seen')} used={event.get('records_used')} "
                f"tokens={event.get('token_count')}",
                file=sys.stderr,
                flush=True,
            )
        elif event_name == "token_cache_complete":
            print(
                f"[tokenizer] cache complete {_format_percent(event)} "
                f"records={event.get('records_seen')} used={event.get('records_used')} "
                f"token_cache={_format_bytes(event.get('cache_bytes'))}",
                file=sys.stderr,
                flush=True,
            )
        elif event_name == "bpe_count_start":
            print(
                f"[tokenizer] merge {event.get('merge_index')}/{event.get('merge_total')} "
                f"count start vocab={event.get('vocab_size')}/{event.get('target_vocab_size')} "
                f"cache={_format_bytes(event.get('total_bytes'))}",
                file=sys.stderr,
                flush=True,
            )
        elif event_name == "bpe_count_progress":
            print(
                f"[tokenizer] merge {event.get('merge_index')}/{event.get('merge_total')} "
                f"count {_format_percent(event)} pairs={event.get('pair_kinds')} "
                f"seq={event.get('sequences')}",
                file=sys.stderr,
                flush=True,
            )
        elif event_name == "bpe_count_complete":
            print(
                f"[tokenizer] merge {event.get('merge_index')}/{event.get('merge_total')} "
                f"count complete {_format_percent(event)} pairs={event.get('pair_kinds')}",
                file=sys.stderr,
                flush=True,
            )
        elif event_name == "bpe_merge_selected":
            print(
                f"[tokenizer] merge {event.get('merge_index')}/{event.get('merge_total')} "
                f"selected pair={event.get('pair')} freq={event.get('frequency')} "
                f"new_id={event.get('new_id')} vocab={event.get('vocab_size')}/{event.get('target_vocab_size')}",
                file=sys.stderr,
                flush=True,
            )
        elif event_name == "bpe_rewrite_start":
            print(
                f"[tokenizer] merge {event.get('merge_index')}/{event.get('merge_total')} "
                f"rewrite start cache={_format_bytes(event.get('total_bytes'))}",
                file=sys.stderr,
                flush=True,
            )
        elif event_name == "bpe_rewrite_progress":
            print(
                f"[tokenizer] merge {event.get('merge_index')}/{event.get('merge_total')} "
                f"rewrite {_format_percent(event)} seq={event.get('sequences')} "
                f"kept={event.get('rewritten_sequences')}",
                file=sys.stderr,
                flush=True,
            )
        elif event_name == "bpe_rewrite_complete":
            print(
                f"[tokenizer] merge {event.get('merge_index')}/{event.get('merge_total')} "
                f"rewrite complete {_format_percent(event)} "
                f"next_cache={_format_bytes(event.get('cache_bytes'))}",
                file=sys.stderr,
                flush=True,
            )
        elif event_name == "bpe_complete":
            print(
                f"[tokenizer] complete reason={event.get('reason')} "
                f"vocab={event.get('vocab_size')}/{event.get('target_vocab_size')} "
                f"merges={event.get('merge_count')}",
                file=sys.stderr,
                flush=True,
            )

    return report


def _format_percent(event: dict[str, object]) -> str:
    bytes_read = int(event.get("bytes_read", 0) or 0)
    total_bytes = int(event.get("total_bytes", 0) or 0)
    if total_bytes <= 0:
        return f"{_format_bytes(bytes_read)}"
    return f"{(bytes_read / total_bytes) * 100:.1f}% {_format_bytes(bytes_read)}/{_format_bytes(total_bytes)}"


def _format_bytes(value: object) -> str:
    size = int(value or 0)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    scaled = float(size)
    for unit in units:
        if scaled < 1024.0 or unit == units[-1]:
            return f"{scaled:.1f}{unit}" if unit != "B" else f"{size}B"
        scaled /= 1024.0


if __name__ == "__main__":
    raise SystemExit(main())

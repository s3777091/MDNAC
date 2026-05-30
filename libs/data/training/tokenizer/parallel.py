"""Parallel workers for BPE pair counting and cache rewriting."""

from __future__ import annotations

import os
import shutil
import struct
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .cache_io import (
    _array_item_size,
    _encoded_sequence_byte_size,
    _read_token_id_sequence,
    _replace_pair_once_in_sequence,
    _write_token_id_sequence,
)
from .resume import _emit_progress

if TYPE_CHECKING:
    from .sequence_tokenizer import TokenizerProgressCallback


@dataclass(slots=True, frozen=True)
class _TokenCacheChunk:
    index: int
    start_offset: int
    end_offset: int

    @property
    def byte_size(self) -> int:
        return self.end_offset - self.start_offset


def _build_token_cache_chunks(
    cache_path: Path,
    *,
    id_typecode: str,
    worker_count: int,
) -> list[_TokenCacheChunk]:
    total_bytes = cache_path.stat().st_size if cache_path.exists() else 0
    if total_bytes <= 0 or worker_count <= 1:
        return [_TokenCacheChunk(index=0, start_offset=0, end_offset=total_bytes)] if total_bytes > 0 else []

    target_chunk_bytes = max(total_bytes // worker_count, 1)
    chunks: list[_TokenCacheChunk] = []
    item_size = _array_item_size(id_typecode)
    chunk_start = 0
    offset = 0

    with cache_path.open("rb") as handle:
        while True:
            raw_length = handle.read(4)
            if not raw_length:
                break
            if len(raw_length) != 4:
                raise ValueError("Truncated tokenizer cache record length.")

            length = struct.unpack("<I", raw_length)[0]
            payload_bytes = length * item_size
            handle.seek(payload_bytes, os.SEEK_CUR)
            offset += 4 + payload_bytes
            if offset > total_bytes:
                raise ValueError("Truncated tokenizer cache record payload.")

            if (
                len(chunks) < worker_count - 1
                and offset - chunk_start >= target_chunk_bytes
                and offset < total_bytes
            ):
                chunks.append(
                    _TokenCacheChunk(
                        index=len(chunks),
                        start_offset=chunk_start,
                        end_offset=offset,
                    )
                )
                chunk_start = offset

    if chunk_start < offset:
        chunks.append(
            _TokenCacheChunk(
                index=len(chunks),
                start_offset=chunk_start,
                end_offset=offset,
            )
        )
    return chunks


def _count_cached_pairs_parallel(
    cache_path: Path,
    *,
    id_typecode: str,
    chunks: list[_TokenCacheChunk],
    merge_index: int,
    merge_total: int,
    vocab_size: int,
    target_vocab_size: int,
    progress_callback: "TokenizerProgressCallback | None",
    executor: ProcessPoolExecutor,
) -> Counter[tuple[int, int]]:
    total_bytes = cache_path.stat().st_size if cache_path.exists() else 0
    pair_counts: Counter[tuple[int, int]] = Counter()
    progress_pair_counts: Counter[tuple[int, int]] = Counter()
    ordered_results: dict[int, tuple[Counter[tuple[int, int]], int, int]] = {}
    bytes_read = 0
    sequence_count = 0

    _emit_progress(
        progress_callback,
        {
            "event": "bpe_count_start",
            "merge_index": merge_index,
            "merge_total": merge_total,
            "vocab_size": vocab_size,
            "target_vocab_size": target_vocab_size,
            "total_bytes": total_bytes,
            "workers": len(chunks),
        },
    )
    futures = [
        executor.submit(
            _count_cached_pairs_range_worker,
            str(cache_path),
            id_typecode,
            chunk.index,
            chunk.start_offset,
            chunk.end_offset,
        )
        for chunk in chunks
    ]
    for future in as_completed(futures):
        chunk_index, chunk_pair_counts, chunk_bytes_read, chunk_sequence_count = future.result()
        ordered_results[chunk_index] = (chunk_pair_counts, chunk_bytes_read, chunk_sequence_count)
        progress_pair_counts.update(chunk_pair_counts)
        bytes_read += chunk_bytes_read
        sequence_count += chunk_sequence_count
        _emit_progress(
            progress_callback,
            {
                "event": "bpe_count_progress",
                "merge_index": merge_index,
                "merge_total": merge_total,
                "bytes_read": min(bytes_read, total_bytes),
                "total_bytes": total_bytes,
                "sequences": sequence_count,
                "pair_kinds": len(progress_pair_counts),
                "workers": len(chunks),
            },
        )

    for chunk_index in sorted(ordered_results):
        chunk_pair_counts, _, _ = ordered_results[chunk_index]
        pair_counts.update(chunk_pair_counts)

    _emit_progress(
        progress_callback,
        {
            "event": "bpe_count_complete",
            "merge_index": merge_index,
            "merge_total": merge_total,
            "bytes_read": min(bytes_read, total_bytes),
            "total_bytes": total_bytes,
            "sequences": sequence_count,
            "pair_kinds": len(pair_counts),
            "workers": len(chunks),
        },
    )
    return pair_counts


def _count_cached_pairs_range_worker(
    cache_path: str,
    id_typecode: str,
    chunk_index: int,
    start_offset: int,
    end_offset: int,
) -> tuple[int, Counter[tuple[int, int]], int, int]:
    pair_counts: Counter[tuple[int, int]] = Counter()
    bytes_read = 0
    sequence_count = 0

    with Path(cache_path).open("rb") as cache_handle:
        cache_handle.seek(start_offset)
        while cache_handle.tell() < end_offset:
            token_ids = _read_token_id_sequence(cache_handle, id_typecode)
            if token_ids is None:
                break
            sequence_count += 1
            bytes_read += _encoded_sequence_byte_size(token_ids, id_typecode)
            for index in range(len(token_ids) - 1):
                pair_counts[(int(token_ids[index]), int(token_ids[index + 1]))] += 1

    return chunk_index, pair_counts, bytes_read, sequence_count


def _rewrite_token_cache_with_merge_parallel(
    source_path: Path,
    target_path: Path,
    *,
    id_typecode: str,
    pair: tuple[int, int],
    new_id: int,
    chunks: list[_TokenCacheChunk],
    merge_index: int,
    merge_total: int,
    vocab_size: int,
    target_vocab_size: int,
    progress_callback: "TokenizerProgressCallback | None",
    executor: ProcessPoolExecutor,
) -> None:
    total_bytes = source_path.stat().st_size if source_path.exists() else 0
    part_paths = [
        target_path.with_name(f"{target_path.name}.part-{merge_index}-{chunk.index:04d}") for chunk in chunks
    ]
    for part_path in part_paths:
        part_path.unlink(missing_ok=True)

    bytes_read = 0
    sequence_count = 0
    rewritten_sequence_count = 0
    part_cache_bytes = 0

    _emit_progress(
        progress_callback,
        {
            "event": "bpe_rewrite_start",
            "merge_index": merge_index,
            "merge_total": merge_total,
            "vocab_size": vocab_size,
            "target_vocab_size": target_vocab_size,
            "pair": pair,
            "new_id": new_id,
            "total_bytes": total_bytes,
            "workers": len(chunks),
        },
    )
    futures = [
        executor.submit(
            _rewrite_token_cache_range_worker,
            str(source_path),
            str(part_paths[chunk.index]),
            id_typecode,
            chunk.index,
            pair,
            new_id,
            chunk.start_offset,
            chunk.end_offset,
        )
        for chunk in chunks
    ]
    try:
        for future in as_completed(futures):
            chunk_index, chunk_bytes_read, chunk_sequence_count, chunk_rewritten_sequence_count, chunk_cache_bytes = (
                future.result()
            )
            bytes_read += chunk_bytes_read
            sequence_count += chunk_sequence_count
            rewritten_sequence_count += chunk_rewritten_sequence_count
            part_cache_bytes += chunk_cache_bytes
            _emit_progress(
                progress_callback,
                {
                    "event": "bpe_rewrite_progress",
                    "merge_index": merge_index,
                    "merge_total": merge_total,
                    "bytes_read": min(bytes_read, total_bytes),
                    "total_bytes": total_bytes,
                    "sequences": sequence_count,
                    "rewritten_sequences": rewritten_sequence_count,
                    "cache_bytes": part_cache_bytes,
                    "workers": len(chunks),
                    "chunk_index": chunk_index,
                },
            )

        target_path.unlink(missing_ok=True)
        with target_path.open("wb") as target_handle:
            for part_path in part_paths:
                with part_path.open("rb") as part_handle:
                    shutil.copyfileobj(part_handle, target_handle, length=16 * 1024 * 1024)

        _emit_progress(
            progress_callback,
            {
                "event": "bpe_rewrite_complete",
                "merge_index": merge_index,
                "merge_total": merge_total,
                "bytes_read": min(bytes_read, total_bytes),
                "total_bytes": total_bytes,
                "sequences": sequence_count,
                "rewritten_sequences": rewritten_sequence_count,
                "cache_bytes": target_path.stat().st_size if target_path.exists() else 0,
                "workers": len(chunks),
            },
        )
    except BaseException:
        for future in futures:
            future.cancel()
        for future in futures:
            try:
                future.result()
            except BaseException:
                pass
        raise
    finally:
        for part_path in part_paths:
            part_path.unlink(missing_ok=True)


def _rewrite_token_cache_range_worker(
    source_path: str,
    target_path: str,
    id_typecode: str,
    chunk_index: int,
    pair: tuple[int, int],
    new_id: int,
    start_offset: int,
    end_offset: int,
) -> tuple[int, int, int, int, int]:
    bytes_read = 0
    sequence_count = 0
    rewritten_sequence_count = 0
    target = Path(target_path)

    with Path(source_path).open("rb") as source_handle, target.open("wb") as target_handle:
        source_handle.seek(start_offset)
        while source_handle.tell() < end_offset:
            token_ids = _read_token_id_sequence(source_handle, id_typecode)
            if token_ids is None:
                break
            sequence_count += 1
            bytes_read += _encoded_sequence_byte_size(token_ids, id_typecode)
            merged_token_ids = _replace_pair_once_in_sequence(token_ids, pair, new_id)
            if len(merged_token_ids) >= 2:
                _write_token_id_sequence(target_handle, merged_token_ids, id_typecode)
                rewritten_sequence_count += 1

    return (
        chunk_index,
        bytes_read,
        sequence_count,
        rewritten_sequence_count,
        target.stat().st_size if target.exists() else 0,
    )

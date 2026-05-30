"""Resume/checkpoint state persistence for tokenizer training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sequence_tokenizer import SequenceTokenizer, SequenceTokenizerTextTrainingStats, TokenizerProgressCallback


def _emit_progress(
    progress_callback: "TokenizerProgressCallback | None",
    event: dict[str, object],
) -> None:
    if progress_callback is not None:
        progress_callback(event)


def _tokenizer_resume_paths(
    *,
    source_path: Path,
    cache_dir: Path,
    sequence_type: str,
    vocab_size: int,
    line_limit: int | None,
    allowed_special: set[str],
) -> tuple[Path, Path, Path]:
    fingerprint_payload = {
        "source_path": str(source_path.resolve()),
        "sequence_type": sequence_type,
        "vocab_size": vocab_size,
        "line_limit": line_limit,
        "allowed_special": sorted(allowed_special),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    base_path = cache_dir / f"sequence-tokenizer-resume-{fingerprint}"
    return (
        base_path.with_suffix(".state.json"),
        base_path.with_suffix(".cache.bin"),
        base_path.with_suffix(".rewrite.bin"),
    )


def _source_signature(source_path: Path) -> dict[str, object]:
    stat = source_path.stat()
    return {
        "path": str(source_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _stats_to_resume_payload(stats: "SequenceTokenizerTextTrainingStats") -> dict[str, int]:
    return {
        "record_count": stats.record_count,
        "tokenizer_train_record_count": stats.tokenizer_train_record_count,
        "token_sequence_count": stats.token_sequence_count,
        "token_count": stats.token_count,
    }


def _stats_from_resume_payload(payload: object) -> "SequenceTokenizerTextTrainingStats":
    from .sequence_tokenizer import SequenceTokenizerTextTrainingStats

    if not isinstance(payload, dict):
        raise ValueError("Tokenizer resume state is missing training stats.")
    return SequenceTokenizerTextTrainingStats(
        record_count=int(payload["record_count"]),
        tokenizer_train_record_count=int(payload["tokenizer_train_record_count"]),
        token_sequence_count=int(payload["token_sequence_count"]),
        token_count=int(payload["token_count"]),
    )


def _load_tokenizer_resume_state(
    state_path: Path | None,
    *,
    source_path: Path,
    sequence_type: str,
    vocab_size: int,
    target_vocab_size: int,
    line_limit: int | None,
    allowed_special: set[str],
    id_typecode: str,
    progress_callback: "TokenizerProgressCallback | None",
) -> dict[str, object] | None:
    if state_path is None or not state_path.exists():
        return None

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state_path.unlink(missing_ok=True)
        return None

    expected = {
        "version": 1,
        "source": _source_signature(source_path),
        "sequence_type": sequence_type,
        "vocab_size": vocab_size,
        "target_vocab_size": target_vocab_size,
        "line_limit": line_limit,
        "allowed_special": sorted(allowed_special),
        "id_typecode": id_typecode,
    }
    ignored_reason = ""
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            ignored_reason = f"{key}_mismatch"
            break

    raw_cache_path = payload.get("cache_path")
    cache_path = Path(raw_cache_path) if isinstance(raw_cache_path, str) and raw_cache_path else None
    if not ignored_reason and cache_path is None:
        ignored_reason = "cache_path_missing"
    if not ignored_reason and cache_path is not None and not cache_path.exists():
        ignored_reason = "cache_missing"

    if ignored_reason:
        _emit_progress(
            progress_callback,
            {
                "event": "tokenizer_resume_ignored",
                "path": str(state_path),
                "reason": ignored_reason,
            },
        )
        state_path.unlink(missing_ok=True)
        if cache_path is not None:
            cache_path.unlink(missing_ok=True)
        return None

    payload["cache_path"] = str(cache_path)
    return payload


def _write_tokenizer_resume_state(
    state_path: Path,
    *,
    source_path: Path,
    cache_path: Path,
    sequence_type: str,
    vocab_size: int,
    target_vocab_size: int,
    line_limit: int | None,
    allowed_special: set[str],
    id_typecode: str,
    stats: "SequenceTokenizerTextTrainingStats",
    tokenizer: "SequenceTokenizer",
    progress_callback: "TokenizerProgressCallback | None",
) -> None:
    payload = {
        "version": 1,
        "source": _source_signature(source_path),
        "sequence_type": sequence_type,
        "vocab_size": vocab_size,
        "target_vocab_size": target_vocab_size,
        "line_limit": line_limit,
        "allowed_special": sorted(allowed_special),
        "id_typecode": id_typecode,
        "cache_path": str(cache_path),
        "cache_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
        "stats": _stats_to_resume_payload(stats),
        "tokenizer": json.loads(tokenizer.to_json()),
        "completed_merges": len(tokenizer.merge_ranks),
    }
    temp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        temp_path.replace(state_path)
    except PermissionError:
        state_path.unlink(missing_ok=True)
        temp_path.replace(state_path)
    _emit_progress(
        progress_callback,
        {
            "event": "tokenizer_checkpoint_saved",
            "path": str(state_path),
            "cache_path": str(cache_path),
            "cache_bytes": payload["cache_bytes"],
            "completed_merges": len(tokenizer.merge_ranks),
            "vocab_size": tokenizer.vocab_size,
            "target_vocab_size": target_vocab_size,
        },
    )

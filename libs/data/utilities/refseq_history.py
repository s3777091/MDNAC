from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REFSEQ_HISTORY_FILENAME = "history.json"
REFSEQ_HISTORY_FORMAT = "microbial_dna_compiler/refseq_history/v1"
REFSEQ_RAW_ROOT_NAME = "refseq_bacteria_protein"
REFSEQ_ARCHIVE_SUFFIXES = (".gpff.gz", ".faa.gz")


def resolve_refseq_history_root(path: Path | str) -> Path:
    resolved = Path(path)
    parts = resolved.parts
    for index, part in enumerate(parts):
        if part == REFSEQ_RAW_ROOT_NAME:
            return Path(*parts[: index + 1])
    return resolved


def resolve_refseq_history_path(path: Path | str) -> Path:
    return resolve_refseq_history_root(path) / REFSEQ_HISTORY_FILENAME


def load_refseq_history(
    history_path: Path | str,
    *,
    input_root: Path | str | None = None,
) -> dict[str, Any]:
    resolved_history_path = Path(history_path)
    resolved_input_root = (
        Path(input_root) if input_root is not None else resolve_refseq_history_root(resolved_history_path.parent)
    )
    if not resolved_history_path.exists():
        return _empty_history_payload(resolved_input_root)

    payload = json.loads(resolved_history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return _empty_history_payload(resolved_input_root)

    history = _empty_history_payload(resolved_input_root)
    history.update(payload)
    if not isinstance(history.get("archives"), dict):
        history["archives"] = {}
    if not isinstance(history.get("builds"), dict):
        history["builds"] = {}
    history["format"] = REFSEQ_HISTORY_FORMAT
    history["input_root"] = str(resolved_input_root)
    return history


def save_refseq_history(history_path: Path | str, history: dict[str, Any]) -> None:
    resolved_history_path = Path(history_path)
    resolved_history_path.parent.mkdir(parents=True, exist_ok=True)
    history["format"] = REFSEQ_HISTORY_FORMAT
    history["updated_at"] = _utc_now_iso()
    resolved_history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def bootstrap_refseq_history(input_root: Path | str, history: dict[str, Any]) -> dict[str, Any]:
    resolved_input_root = Path(input_root)
    seen_keys: set[str] = set()
    for suffix in REFSEQ_ARCHIVE_SUFFIXES:
        for archive_path in resolved_input_root.rglob(f"*{suffix}"):
            entry = register_archive_file(history, resolved_input_root, archive_path)
            if not entry.get("last_download_status"):
                entry["last_download_status"] = "bootstrapped"
            seen_keys.add(_archive_key(resolved_input_root, archive_path))

    archives = history.setdefault("archives", {})
    for archive_key, entry in list(archives.items()):
        if not isinstance(entry, dict):
            archives[archive_key] = {}
            entry = archives[archive_key]
        if archive_key not in seen_keys:
            entry["present_on_disk"] = False

    history["input_root"] = str(resolved_input_root)
    return history


def register_archive_file(
    history: dict[str, Any],
    input_root: Path | str,
    archive_path: Path | str,
    *,
    url: str | None = None,
    expected_size: int | None = None,
    download_status: str | None = None,
) -> dict[str, Any]:
    resolved_input_root = Path(input_root)
    resolved_archive_path = Path(archive_path)
    archive_key = _archive_key(resolved_input_root, resolved_archive_path)
    archive_entry = _ensure_archive_entry(history, resolved_input_root, resolved_archive_path)

    stats = resolved_archive_path.stat() if resolved_archive_path.exists() else None
    if stats is not None:
        archive_entry["local_size"] = stats.st_size
        archive_entry["modified_time_ns"] = stats.st_mtime_ns
        archive_entry["present_on_disk"] = True
    else:
        archive_entry["present_on_disk"] = False

    if url:
        archive_entry["url"] = url
    if expected_size is not None:
        archive_entry["expected_size"] = expected_size
    if download_status:
        archive_entry["last_download_status"] = download_status
        archive_entry["last_download_at"] = _utc_now_iso()

    if stats is not None:
        if not _matches_compiled_snapshot(archive_entry, stats.st_size, stats.st_mtime_ns):
            archive_entry["build_status"] = "pending"
        elif archive_entry.get("build_status") == "compiled":
            archive_entry["build_status"] = "compiled"
    elif archive_entry.get("build_status") is None:
        archive_entry["build_status"] = "pending"

    history.setdefault("archives", {})[archive_key] = archive_entry
    return archive_entry


def mark_archive_compiled(
    history: dict[str, Any],
    input_root: Path | str,
    archive_path: Path | str,
    *,
    output_dir: Path | str,
) -> dict[str, Any]:
    resolved_input_root = Path(input_root)
    resolved_archive_path = Path(archive_path)
    archive_entry = _ensure_archive_entry(history, resolved_input_root, resolved_archive_path)
    if resolved_archive_path.exists():
        stats = resolved_archive_path.stat()
        archive_entry["local_size"] = stats.st_size
        archive_entry["modified_time_ns"] = stats.st_mtime_ns
        archive_entry["compiled_local_size"] = stats.st_size
        archive_entry["compiled_modified_time_ns"] = stats.st_mtime_ns
        archive_entry["present_on_disk"] = True
    archive_entry["build_status"] = "compiled"
    archive_entry["compiled_output_dir"] = str(Path(output_dir))
    archive_entry["last_compiled_at"] = _utc_now_iso()
    return archive_entry


def mark_archive_deleted_after_compile(
    history: dict[str, Any],
    input_root: Path | str,
    archive_path: Path | str,
) -> dict[str, Any]:
    resolved_input_root = Path(input_root)
    resolved_archive_path = Path(archive_path)
    archive_entry = _ensure_archive_entry(history, resolved_input_root, resolved_archive_path)
    archive_entry["present_on_disk"] = False
    archive_entry["deleted_after_compile"] = True
    archive_entry["deleted_at"] = _utc_now_iso()
    return archive_entry


def should_process_archive(
    history: dict[str, Any],
    input_root: Path | str,
    archive_path: Path | str,
) -> bool:
    resolved_input_root = Path(input_root)
    resolved_archive_path = Path(archive_path)
    archive_key = _archive_key(resolved_input_root, resolved_archive_path)
    archive_entry = history.get("archives", {}).get(archive_key)
    if not isinstance(archive_entry, dict):
        return True
    if archive_entry.get("build_status") != "compiled":
        return True
    if not resolved_archive_path.exists():
        return False
    stats = resolved_archive_path.stat()
    return not _matches_compiled_snapshot(archive_entry, stats.st_size, stats.st_mtime_ns)


def record_build_snapshot(
    history: dict[str, Any],
    *,
    output_dir: Path | str,
    summary: dict[str, Any],
) -> None:
    builds = history.setdefault("builds", {})
    builds[str(Path(output_dir))] = {
        **summary,
        "updated_at": _utc_now_iso(),
    }


def _empty_history_payload(input_root: Path) -> dict[str, Any]:
    return {
        "format": REFSEQ_HISTORY_FORMAT,
        "input_root": str(input_root),
        "updated_at": _utc_now_iso(),
        "archives": {},
        "builds": {},
    }


def _ensure_archive_entry(
    history: dict[str, Any],
    input_root: Path,
    archive_path: Path,
) -> dict[str, Any]:
    archive_key = _archive_key(input_root, archive_path)
    archives = history.setdefault("archives", {})
    existing = archives.get(archive_key)
    archive_entry = dict(existing) if isinstance(existing, dict) else {}
    relative_path = Path(archive_key)
    archive_entry.setdefault("file_name", archive_path.name)
    archive_entry.setdefault("relative_path", archive_key)
    archive_entry.setdefault(
        "group_name",
        relative_path.parent.name if relative_path.parent != Path(".") else input_root.name,
    )
    archive_entry.setdefault("kind", _archive_kind(archive_path))
    archive_entry.setdefault("build_status", "pending")
    archives[archive_key] = archive_entry
    return archive_entry


def _archive_key(input_root: Path, archive_path: Path) -> str:
    return archive_path.relative_to(input_root).as_posix()


def _archive_kind(archive_path: Path) -> str:
    archive_name = archive_path.name
    if archive_name.endswith(".gpff.gz"):
        return "gpff"
    if archive_name.endswith(".faa.gz"):
        return "faa"
    return "unknown"


def _matches_compiled_snapshot(archive_entry: dict[str, Any], size: int, modified_time_ns: int) -> bool:
    try:
        compiled_size = int(archive_entry.get("compiled_local_size"))
        compiled_mtime_ns = int(archive_entry.get("compiled_modified_time_ns"))
    except (TypeError, ValueError):
        return False
    return compiled_size == size and compiled_mtime_ns == modified_time_ns


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

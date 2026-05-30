from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def create_resume_state(
    *,
    mode: str,
    config_path: str,
    training_config_snapshot_path: str,
    checkpoint_path: str,
    best_checkpoint_path: str,
    final_checkpoint_path: str,
    model_info: dict[str, Any],
    optimizer_info: dict[str, Any],
    runtime_info: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": run_id or uuid.uuid4().hex[:12],
        "mode": mode,
        "config_path": str(config_path),
        "training_config_snapshot_path": str(training_config_snapshot_path),
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "model": dict(model_info),
        "optimizer": dict(optimizer_info),
        "runtime": dict(runtime_info),
        "progress": {
            "epoch": 0,
            "global_step": 0,
            "tokens_seen": 0,
            "current_part_index": 0,
            "current_part_uri": None,
            "current_batch_index": 0,
            "completed_parts": [],
        },
        "metrics": {
            "latest_train_loss": None,
            "latest_val_loss": None,
            "best_loss": None,
            "best_metric_name": "val_loss",
            "train_losses": [],
            "val_losses": [],
        },
        "timestamps": {
            "created_at": now,
            "updated_at": now,
        },
    }


def update_resume_state_progress(
    state: dict[str, Any],
    *,
    epoch: int | None = None,
    global_step: int | None = None,
    tokens_seen: int | None = None,
    current_part_index: int | None = None,
    current_part_uri: str | None = None,
    current_batch_index: int | None = None,
    completed_parts: list[str] | None = None,
) -> dict[str, Any]:
    progress = state.setdefault("progress", {})
    if epoch is not None:
        progress["epoch"] = epoch
    if global_step is not None:
        progress["global_step"] = global_step
    if tokens_seen is not None:
        progress["tokens_seen"] = tokens_seen
    if current_part_index is not None:
        progress["current_part_index"] = current_part_index
    if current_part_uri is not None:
        progress["current_part_uri"] = current_part_uri
    if current_batch_index is not None:
        progress["current_batch_index"] = current_batch_index
    if completed_parts is not None:
        progress["completed_parts"] = completed_parts
    state["timestamps"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    return state


def update_resume_state_metrics(
    state: dict[str, Any],
    *,
    train_loss: float | None = None,
    val_loss: float | None = None,
    best_loss: float | None = None,
    best_metric_name: str | None = None,
) -> dict[str, Any]:
    metrics = state.setdefault("metrics", {})
    if train_loss is not None:
        metrics["latest_train_loss"] = train_loss
        metrics.setdefault("train_losses", []).append(train_loss)
    if val_loss is not None:
        metrics["latest_val_loss"] = val_loss
        metrics.setdefault("val_losses", []).append(val_loss)
    if best_loss is not None:
        metrics["best_loss"] = best_loss
    if best_metric_name is not None:
        metrics["best_metric_name"] = best_metric_name
    state["timestamps"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    return state


def save_resume_state(state: dict[str, Any], path: Path | str) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    backup_path = resolved_path.with_suffix(".json.bak")
    if resolved_path.exists():
        shutil.copy2(resolved_path, backup_path)

    temp_fd, temp_path_str = tempfile.mkstemp(
        dir=str(resolved_path.parent),
        prefix=".resume_state_",
        suffix=".tmp",
    )
    temp_path = Path(temp_path_str)
    try:
        with open(temp_fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temp_path.replace(resolved_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    return resolved_path


def load_resume_state(path: Path | str) -> dict[str, Any] | None:
    resolved_path = Path(path)
    if not resolved_path.exists():
        return None
    text = resolved_path.read_text(encoding="utf-8")
    if not text.strip():
        return None
    return json.loads(text)

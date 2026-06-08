from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_OPENMM_SIMULATION = "openmm_simulation"
PROTEIN_SIMULATION_QUEUE = "protein_simulation_queue"
RUN_OPENMM_SIMULATION_TASK = "structure_predictor.simulation.run_openmm_simulation"

_SAFE_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class SimulationJobStore:
    def __init__(self, jobs_root: str | Path) -> None:
        self.jobs_root = Path(jobs_root).expanduser().resolve()

    def create_job(
        self,
        *,
        job_id: str,
        payload: dict[str, Any],
        run_target: str,
        external_job_id: str | None = None,
    ) -> dict[str, Any]:
        total_steps = int((payload.get("environment") or {}).get("steps") or 0)
        state = {
            "job_id": job_id,
            "task": TASK_OPENMM_SIMULATION,
            "status": "queued",
            "run_target": run_target,
            "step": "queued",
            "progress": {
                "current_step": 0,
                "total_steps": total_steps,
            },
            "error": None,
            "external_job_id": external_job_id,
            "request": payload,
            "result": None,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        self.write_job(state)
        return state

    def get_job(self, job_id: str) -> dict[str, Any]:
        path = self._job_path(job_id)
        if not path.is_file():
            raise FileNotFoundError(f"Simulation job not found: {job_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Simulation job state must be a JSON object: {path}")
        return payload

    def update_job(self, job_id: str, **updates: Any) -> dict[str, Any]:
        state = self.get_job(job_id)
        state.update(updates)
        state["updated_at"] = _utc_now()
        self.write_job(state)
        return state

    def write_job(self, state: dict[str, Any]) -> None:
        job_id = str(state["job_id"])
        path = self._job_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary_path, path)

    def _job_path(self, job_id: str) -> Path:
        normalized_job_id = str(job_id).strip()
        if not _SAFE_JOB_ID_PATTERN.fullmatch(normalized_job_id):
            raise ValueError(f"Invalid simulation job_id: {job_id!r}")
        return self.jobs_root / normalized_job_id / "job.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

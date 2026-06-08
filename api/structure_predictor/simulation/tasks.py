from __future__ import annotations

from typing import Any

from structure_predictor.config import load_config

from .celery_app import create_celery_app
from .jobs import RUN_OPENMM_SIMULATION_TASK, SimulationJobStore
from .openmm_runner import OpenMMSimulationRunner


celery_app = create_celery_app()


@celery_app.task(name=RUN_OPENMM_SIMULATION_TASK, bind=True)
def run_openmm_simulation_task(self, payload: dict[str, Any]) -> dict[str, Any]:
    _ = self
    settings = load_config()
    store = SimulationJobStore(settings.simulation.jobs_root)
    job_id = str(payload["job_id"])
    environment = payload.get("environment") or {}
    total_steps = int(environment.get("steps") or 0)
    store.update_job(
        job_id,
        status="running",
        step="setup",
        progress={"current_step": 0, "total_steps": total_steps},
        error=None,
    )

    def update_progress(current_step: int, total_steps: int, step: str) -> None:
        store.update_job(
            job_id,
            status="running",
            step=step,
            progress={
                "current_step": current_step,
                "total_steps": total_steps,
            },
            error=None,
        )

    try:
        runner = OpenMMSimulationRunner(settings.simulation.jobs_root)
        result = runner.run(payload, progress_callback=update_progress)
    except Exception as exc:
        store.update_job(
            job_id,
            status="failed",
            step="failed",
            error=str(exc),
        )
        raise

    store.update_job(
        job_id,
        status="completed",
        step="completed",
        progress={
            "current_step": int(result["steps"]),
            "total_steps": int(result["steps"]),
        },
        result=result,
        error=None,
    )
    return result

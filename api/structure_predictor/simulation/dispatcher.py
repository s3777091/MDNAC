from __future__ import annotations

from typing import Any

from structure_predictor.config import StructureAPISettings

from .celery_app import create_celery_app
from .jobs import RUN_OPENMM_SIMULATION_TASK


def enqueue_local_simulation_job(
    payload: dict[str, Any],
    *,
    settings: StructureAPISettings,
) -> None:
    app = create_celery_app(settings)
    app.send_task(
        RUN_OPENMM_SIMULATION_TASK,
        args=[payload],
        queue=settings.simulation.queue_name,
        task_id=str(payload["job_id"]),
    )

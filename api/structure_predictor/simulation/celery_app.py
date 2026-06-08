from __future__ import annotations

from typing import Any

from structure_predictor.config import StructureAPISettings, load_config

from .jobs import RUN_OPENMM_SIMULATION_TASK


def create_celery_app(settings: StructureAPISettings | None = None):
    try:
        from celery import Celery
    except ImportError as exc:
        raise RuntimeError(
            "Enqueuing OpenMM simulations requires `celery`. "
            "Install the api project with the simulation dependencies."
        ) from exc

    resolved_settings = settings or load_config()
    app = Celery(
        "mdnac_structure_simulation",
        broker=resolved_settings.simulation.rabbitmq_url,
    )
    app.conf.update(
        _celery_config(resolved_settings),
    )
    return app


def _celery_config(settings: StructureAPISettings) -> dict[str, Any]:
    return {
        "task_default_queue": settings.simulation.queue_name,
        "task_routes": {
            RUN_OPENMM_SIMULATION_TASK: {"queue": settings.simulation.queue_name},
        },
        "worker_prefetch_multiplier": settings.simulation.worker_prefetch_multiplier,
        "task_acks_late": settings.simulation.task_acks_late,
        "task_reject_on_worker_lost": settings.simulation.task_reject_on_worker_lost,
        "task_track_started": settings.simulation.task_track_started,
    }

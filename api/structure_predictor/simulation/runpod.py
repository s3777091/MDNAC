from __future__ import annotations

from typing import Any


def submit_runpod_simulation_job(payload: dict[str, Any]) -> str | None:
    """Submit an OpenMM simulation to RunPod when a dispatcher is available.

    The project does not currently include a RunPod simulation dispatcher. This
    interface keeps the API contract explicit without inventing a fake result.
    Return the external RunPod job id here once the dispatcher is implemented.
    """

    _ = payload
    return None

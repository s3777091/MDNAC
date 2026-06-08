"""Async OpenMM simulation support for the structure predictor API."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "PROTEIN_SIMULATION_QUEUE",
    "RUN_OPENMM_SIMULATION_TASK",
    "TASK_OPENMM_SIMULATION",
    "SimulationJobStore",
    "enqueue_local_simulation_job",
    "submit_runpod_simulation_job",
]


def __getattr__(name: str):
    if name in {
        "PROTEIN_SIMULATION_QUEUE",
        "RUN_OPENMM_SIMULATION_TASK",
        "TASK_OPENMM_SIMULATION",
        "SimulationJobStore",
    }:
        module = import_module("structure_predictor.simulation.jobs")
        return getattr(module, name)
    if name == "enqueue_local_simulation_job":
        module = import_module("structure_predictor.simulation.dispatcher")
        return getattr(module, name)
    if name == "submit_runpod_simulation_job":
        module = import_module("structure_predictor.simulation.runpod")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

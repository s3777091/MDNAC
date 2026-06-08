"""OpenFold-backed protein structure prediction API."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "OpenFoldPredictionResult",
    "OpenFoldRunner",
    "OpenFoldSettings",
    "SimulationSettings",
    "StructureAPISettings",
    "StructurePredictionRequest",
    "build_openfold_command",
    "load_config",
]


def __getattr__(name: str):
    if name in {"OpenFoldSettings", "SimulationSettings", "StructureAPISettings", "load_config"}:
        module = import_module("structure_predictor.config")
        return getattr(module, name)
    if name in {
        "OpenFoldPredictionResult",
        "OpenFoldRunner",
        "StructurePredictionRequest",
        "build_openfold_command",
    }:
        module = import_module("structure_predictor.openfold")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

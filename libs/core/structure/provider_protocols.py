"""Protocol for structure prediction providers."""

from __future__ import annotations

from typing import Protocol

from .types import StructurePrediction


class StructurePredictionProvider(Protocol):
    """Protocol that external structure prediction adapters must satisfy.

    Implementations should wrap tools like AlphaFold, Boltz-2, ESMFold, etc.
    If no provider is configured, validation must report 'missing_structure_provider'
    rather than silently passing.
    """

    model_name: str

    def predict(self, sequence: str) -> StructurePrediction: ...

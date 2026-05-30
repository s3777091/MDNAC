"""SQLite cache for 3Di structure predictions keyed by model + sequence hash."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


def _sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


class Sequence3DiCache:
    """Small SQLite cache keyed by model + protein sequence hash."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sequence_3di_cache (
                model_name TEXT NOT NULL,
                sequence_sha256 TEXT NOT NULL,
                sequence TEXT NOT NULL,
                structure_3di TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                PRIMARY KEY (model_name, sequence_sha256)
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def get(self, *, model_name: str, sequence: str) -> str | None:
        from .instruction_3di import normalize_3di_structure

        sequence_hash = _sequence_hash(sequence)
        row = self._connection.execute(
            """
            SELECT structure_3di
            FROM sequence_3di_cache
            WHERE model_name = ? AND sequence_sha256 = ? AND sequence = ?
            """,
            (model_name, sequence_hash, sequence),
        ).fetchone()
        if row is None:
            return None
        normalized = normalize_3di_structure(row[0])
        return normalized or None

    def set_many(self, *, model_name: str, values: Mapping[str, str]) -> None:
        if not values:
            return

        updated_at = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                model_name,
                _sequence_hash(sequence),
                sequence,
                structure_3di,
                updated_at,
            )
            for sequence, structure_3di in values.items()
        ]
        self._connection.executemany(
            """
            INSERT INTO sequence_3di_cache (
                model_name,
                sequence_sha256,
                sequence,
                structure_3di,
                updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(model_name, sequence_sha256) DO UPDATE SET
                sequence = excluded.sequence,
                structure_3di = excluded.structure_3di,
                updated_at_utc = excluded.updated_at_utc
            """,
            rows,
        )
        self._connection.commit()

    def __enter__(self) -> "Sequence3DiCache":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

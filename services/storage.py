"""SQLite results store with CSV export.

SQLite is used for robustness (concurrent testers, no lost rows); every row
can be exported to CSV for analysis at any time.
"""

from __future__ import annotations

import csv
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

logger = logging.getLogger("stream_tts.storage")

TABLE = "evaluations"

#: Column name -> SQLite type. The order defines the CSV column order.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("evaluation_id", "TEXT PRIMARY KEY"),
    ("timestamp", "TEXT NOT NULL"),
    ("tester_id", "TEXT NOT NULL"),
    ("session_id", "TEXT"),
    ("trial_id", "TEXT"),
    ("language", "TEXT NOT NULL"),
    ("accent", "TEXT"),
    ("gender", "TEXT"),
    ("speaker_id", "TEXT NOT NULL"),
    ("sentence_id", "TEXT NOT NULL"),
    ("sentence_text", "TEXT"),
    # INTERNAL decoding fields - researchers only, never shown to testers.
    ("sample_A_model", "TEXT NOT NULL"),
    ("sample_B_model", "TEXT NOT NULL"),
    ("preferred_sample", "TEXT NOT NULL"),
    ("preferred_model", "TEXT"),
    ("naturalness_A", "INTEGER"),
    ("naturalness_B", "INTEGER"),
    ("pronunciation_A", "INTEGER"),
    ("pronunciation_B", "INTEGER"),
    ("speaker_similarity_A", "INTEGER"),
    ("speaker_similarity_B", "INTEGER"),
    ("comments", "TEXT"),
    ("sample_A_cache_key", "TEXT"),
    ("sample_B_cache_key", "TEXT"),
    ("generation_mode", "TEXT"),
    ("reference_audio", "TEXT"),
    ("reference_text", "TEXT"),
    ("selection_mode", "TEXT"),
    # Per-trial, not per-process: 1 when either side came from a placeholder
    # system, so those rows can be excluded from the analysis.
    ("mock_mode", "INTEGER"),
    ("sample_A_mocked", "INTEGER"),
    ("sample_B_mocked", "INTEGER"),
    # 0 when the A/B mapping was pinned by configuration instead of drawn.
    ("ab_randomized", "INTEGER"),
    ("app_version", "TEXT"),
    ("listening_seconds", "REAL"),
)

COLUMN_NAMES: tuple[str, ...] = tuple(name for name, _ in COLUMNS)

#: Columns that decode the anonymised A/B mapping.
INTERNAL_COLUMNS: frozenset[str] = frozenset({"sample_A_model", "sample_B_model", "preferred_model"})


class StorageError(RuntimeError):
    pass


class ResultsStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    # -- connection --------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            raise StorageError(str(exc)) from exc
        finally:
            connection.close()

    def initialize(self) -> None:
        definition = ", ".join(f"{name} {sql_type}" for name, sql_type in COLUMNS)
        with self._connect() as connection:
            connection.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} ({definition})")
            connection.execute(f"CREATE INDEX IF NOT EXISTS idx_lang ON {TABLE}(language)")
            connection.execute(f"CREATE INDEX IF NOT EXISTS idx_tester ON {TABLE}(tester_id)")
            # Add columns introduced by later versions of the app.
            existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({TABLE})")}
            for name, sql_type in COLUMNS:
                if name not in existing:
                    base_type = sql_type.split()[0]
                    logger.info("Adding missing results column %s", name)
                    connection.execute(f"ALTER TABLE {TABLE} ADD COLUMN {name} {base_type}")

    # -- writes ------------------------------------------------------------
    def save(self, row: dict[str, Any]) -> None:
        unknown = set(row) - set(COLUMN_NAMES)
        if unknown:
            raise StorageError(f"Unknown result column(s): {', '.join(sorted(unknown))}")
        values = [row.get(name) for name in COLUMN_NAMES]
        placeholders = ", ".join("?" for _ in COLUMN_NAMES)
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO {TABLE} ({', '.join(COLUMN_NAMES)}) VALUES ({placeholders})",
                values,
            )
        logger.info(
            "Saved evaluation %s (%s/%s)",
            row.get("evaluation_id"),
            row.get("language"),
            row.get("sentence_id"),
        )

    # -- reads -------------------------------------------------------------
    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0])

    def rows(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        query = f"SELECT {', '.join(COLUMN_NAMES)} FROM {TABLE} ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query)]

    def dataframe(self, *, include_internal: bool = True):
        """Results as a pandas DataFrame (imported lazily)."""
        import pandas as pd

        frame = pd.DataFrame(self.rows(), columns=list(COLUMN_NAMES))
        if not include_internal:
            frame = frame.drop(columns=[c for c in INTERNAL_COLUMNS if c in frame.columns])
        return frame

    def counts_by(self, column: str) -> list[dict[str, Any]]:
        if column not in COLUMN_NAMES:
            raise StorageError(f"Unknown column {column!r}")
        with self._connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    f"SELECT COALESCE(NULLIF({column}, ''), '(none)') AS value, "
                    f"COUNT(*) AS evaluations FROM {TABLE} GROUP BY value ORDER BY evaluations DESC"
                )
            ]

    def preference_breakdown(self, group_by: str | None = None) -> list[dict[str, Any]]:
        """Preference counts, decoded to the internal systems.

        Researcher-only: the result mentions internal system names.
        """
        if group_by is not None and group_by not in COLUMN_NAMES:
            raise StorageError(f"Unknown column {group_by!r}")
        group_expression = f"COALESCE(NULLIF({group_by}, ''), '(none)')" if group_by else "'all'"
        query = f"""
            SELECT {group_expression} AS value,
                   COUNT(*) AS evaluations,
                   SUM(CASE WHEN preferred_model = 'individual' THEN 1 ELSE 0 END) AS individual_wins,
                   SUM(CASE WHEN preferred_model = 'combined' THEN 1 ELSE 0 END) AS combined_wins,
                   SUM(CASE WHEN preferred_model = 'tie' THEN 1 ELSE 0 END) AS ties,
                   ROUND(AVG(CASE WHEN sample_A_model = 'individual'
                                  THEN naturalness_A ELSE naturalness_B END), 2) AS individual_naturalness,
                   ROUND(AVG(CASE WHEN sample_A_model = 'combined'
                                  THEN naturalness_A ELSE naturalness_B END), 2) AS combined_naturalness
            FROM {TABLE}
            GROUP BY value
            ORDER BY evaluations DESC
        """
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query)]

    def covered_condition_keys(self) -> set[str]:
        with self._connect() as connection:
            return {
                "/".join(
                    [
                        row["language"],
                        row["accent"] or "-",
                        row["gender"] or "",
                        row["speaker_id"],
                        row["sentence_id"],
                    ]
                )
                for row in connection.execute(
                    f"SELECT language, accent, gender, speaker_id, sentence_id FROM {TABLE}"
                )
            }

    def coverage(self, total_conditions: int) -> dict[str, Any]:
        covered = len(self.covered_condition_keys())
        percent = round(100.0 * covered / total_conditions, 2) if total_conditions else 0.0
        return {
            "conditions_total": total_conditions,
            "conditions_evaluated": covered,
            "coverage_percent": percent,
            "evaluations": self.count(),
        }

    # -- export ------------------------------------------------------------
    def export_csv(self, path: Path | None = None, *, include_internal: bool = True) -> Path:
        target = Path(path) if path else self.db_path.parent / f"evaluations_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        columns: Sequence[str] = (
            COLUMN_NAMES
            if include_internal
            else tuple(c for c in COLUMN_NAMES if c not in INTERNAL_COLUMNS)
        )
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows())
        logger.info("Exported %s evaluations to %s", self.count(), target)
        return target

    def csv_bytes(self, *, include_internal: bool = True) -> bytes:
        import io

        columns: Sequence[str] = (
            COLUMN_NAMES
            if include_internal
            else tuple(c for c in COLUMN_NAMES if c not in INTERNAL_COLUMNS)
        )
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.rows())
        return buffer.getvalue().encode("utf-8")

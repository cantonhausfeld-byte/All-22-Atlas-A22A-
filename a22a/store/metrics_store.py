"""Append-only metrics store backed by DuckDB."""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import Any, Iterable

import duckdb

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metrics (
    recorded_at TIMESTAMP,
    run_id VARCHAR,
    namespace VARCHAR,
    metric VARCHAR,
    value DOUBLE,
    context JSON
)
"""


class MetricsStore:
    """Lightweight wrapper for persisting metrics to DuckDB."""

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA_SQL)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.path))

    # Write API -----------------------------------------------------------------------

    def append_metrics(
        self,
        namespace: str,
        metrics: dict[str, float],
        *,
        run_id: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        if not metrics:
            return
        payload_context = json.dumps(context or {})
        recorded_at = _dt.datetime.now(tz=_dt.timezone.utc)
        rows = [
            (recorded_at, run_id, namespace, metric, float(value), payload_context)
            for metric, value in metrics.items()
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    # Read API ------------------------------------------------------------------------

    def read_recent(self, namespace: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
        query = "SELECT * FROM metrics"
        params: list[Any] = []
        if namespace:
            query += " WHERE namespace = ?"
            params.append(namespace)
        query += " ORDER BY recorded_at DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            cursor = conn.execute(query, params)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in results]

    def namespaces(self) -> Iterable[str]:
        query = "SELECT DISTINCT namespace FROM metrics ORDER BY namespace"
        with self._connect() as conn:
            rows = conn.execute(query).fetchall()
        return [row[0] for row in rows]


__all__ = ["MetricsStore"]

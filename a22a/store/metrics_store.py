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

BACKTEST_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS backtest_metrics (
    date DATE,
    season INTEGER,
    week INTEGER,
    n_bets INTEGER,
    roi DOUBLE,
    win_pct DOUBLE,
    ece DOUBLE,
    clv_bps_mean DOUBLE,
    bankroll DOUBLE
)
"""


class MetricsStore:
    """Lightweight wrapper for persisting metrics to DuckDB."""

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA_SQL)
            conn.execute(BACKTEST_SCHEMA_SQL)

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

    def append_backtest_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        """Persist weekly backtest metrics into the structured table."""

        prepared: list[tuple[Any, ...]] = []
        for row in rows:
            prepared.append(
                (
                    row.get("date"),
                    int(row.get("season", 0)),
                    int(row.get("week", 0)),
                    int(row.get("n_bets", 0)),
                    float(row.get("roi", 0.0)),
                    float(row.get("win_pct", 0.0)),
                    float(row.get("ece", 0.0)),
                    float(row.get("clv_bps_mean", 0.0)),
                    float(row.get("bankroll", 0.0)),
                )
            )

        if not prepared:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO backtest_metrics
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                prepared,
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

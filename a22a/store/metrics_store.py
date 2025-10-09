"""DuckDB-backed metrics store used by the bootstrap backtester."""

from __future__ import annotations

import pathlib
from typing import Any, Iterable

import duckdb

DEFAULT_STORE_PATH = pathlib.Path("./artifacts/store/a22a_metrics.duckdb")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metrics (
  ts TIMESTAMP,
  season INTEGER,
  week INTEGER,
  n_bets INTEGER,
  win_pct DOUBLE,
  roi DOUBLE,
  ece DOUBLE,
  clv_bps_mean DOUBLE,
  drawdown DOUBLE,
  herfindahl DOUBLE,
  bankroll DOUBLE
)
"""


class MetricsStore:
    """Minimal append-only wrapper around DuckDB."""

    def __init__(self, path: str | pathlib.Path = DEFAULT_STORE_PATH):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.execute(SCHEMA_SQL)

    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.path))

    def append(self, record: dict[str, Any]) -> None:
        keys = (
            "ts",
            "season",
            "week",
            "n_bets",
            "win_pct",
            "roi",
            "ece",
            "clv_bps_mean",
            "drawdown",
            "herfindahl",
            "bankroll",
        )
        values = tuple(record.get(key) for key in keys)
        with self.connect() as con:
            con.execute(
                "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )

    def read_all(self) -> Iterable[tuple[Any, ...]]:
        with self.connect() as con:
            return con.execute("SELECT * FROM metrics").fetchall()


def ensure_store(path: str | pathlib.Path = DEFAULT_STORE_PATH) -> duckdb.DuckDBPyConnection:
    store = MetricsStore(path)
    return store.connect()


__all__ = ["MetricsStore", "ensure_store"]

"""Phase 2 ingestion scaffold."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import polars as pl

from a22a.data import contracts

STAGED_DIR = Path("staged")


def ensure_staged_dir() -> Path:
    STAGED_DIR.mkdir(parents=True, exist_ok=True)
    return STAGED_DIR


def build_empty_tables() -> Dict[str, pl.DataFrame]:
    """Create empty tables based on configured contracts."""

    tables: Dict[str, pl.DataFrame] = {}
    for name, contract in contracts.load_contracts().items():
        lazy = contract.lazy_frame()
        tables[name] = lazy.collect()
    return tables


def validate_tables(tables: Dict[str, pl.DataFrame]) -> None:
    for name, frame in tables.items():
        contracts.assert_contract(name, frame)


def main() -> None:
    ensure_staged_dir()
    tables = build_empty_tables()
    validate_tables(tables)
    print("[ingest] staged directory ready at", ensure_staged_dir())
    print("[ingest] no raw sources available yet; produced empty tables:")
    for name, frame in tables.items():
        print(f"    - {name}: {frame.shape[0]} rows, columns={frame.columns}")


if __name__ == "__main__":
    main()

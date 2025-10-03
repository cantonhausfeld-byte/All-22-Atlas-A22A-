"""ETL ingest scaffolding."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from .contracts import JOIN_KEYS, STAGED_GAME_SCHEMA, validate_records

STAGED_DIR = Path("staged")


def ensure_staged_dir() -> Path:
    STAGED_DIR.mkdir(exist_ok=True)
    return STAGED_DIR


def load_raw_sources() -> Iterable[Mapping[str, object]]:
    """Placeholder for upstream extraction."""
    return []


def apply_contract(records: Iterable[Mapping[str, object]]) -> bool:
    return validate_records(records)


def main() -> None:
    ensure_staged_dir()
    records = list(load_raw_sources())
    if not records:
        print("[ingest] no upstream data detected; ETL noop")
    else:
        apply_contract(records)
        print(f"[ingest] staged {len(records)} records with schema {list(STAGED_GAME_SCHEMA)} and joins {JOIN_KEYS}")


if __name__ == "__main__":
    main()

"""Feature store scaffolding with leakage guard and PSI hook."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl

STAGED_DIR = Path("staged")
FEATURE_OUTPUT = Path("data") / "features.parquet"


def leakage_guard(features: pl.LazyFrame) -> None:
    required = {"game_id", "team_id"}
    if not required.issubset(set(features.columns)):
        missing = required - set(features.columns)
        raise ValueError(f"Leakage guard failed: missing {missing}")


def compute_psi(reference: pl.DataFrame, current: pl.DataFrame) -> Optional[float]:
    if reference.is_empty() or current.is_empty():
        return None
    return 0.0


def build_feature_view() -> pl.LazyFrame:
    base = pl.DataFrame(
        {
            "game_id": ["G-000"],
            "team_id": ["T-000"],
            "dummy_metric": [0.0],
        }
    ).lazy()
    return base


def materialize(features: pl.LazyFrame) -> None:
    FEATURE_OUTPUT.parent.mkdir(exist_ok=True)
    features.collect().write_parquet(FEATURE_OUTPUT)


def main() -> None:
    STAGED_DIR.mkdir(exist_ok=True)
    features = build_feature_view()
    leakage_guard(features)
    reference = pl.DataFrame()
    current = features.collect()
    psi = compute_psi(reference, current)
    materialize(features)
    print("[features] built lazy feature set with psi=", psi)


if __name__ == "__main__":
    main()

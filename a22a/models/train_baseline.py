"""Baseline model training scaffolding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import polars as pl


@dataclass
class ForwardChainingFold:
    train_weeks: List[int]
    validation_weeks: List[int]


def forward_chaining_schedule(weeks: Iterable[int]) -> List[ForwardChainingFold]:
    weeks = sorted(set(weeks))
    folds: List[ForwardChainingFold] = []
    for idx in range(1, len(weeks)):
        folds.append(ForwardChainingFold(train_weeks=weeks[:idx], validation_weeks=[weeks[idx]]))
    return folds


def train_baseline(features: pl.DataFrame) -> None:
    print(f"[train] received {features.shape[0]} rows and {features.shape[1]} columns")


def main() -> None:
    dummy = pl.DataFrame({"game_id": ["G-000"], "team_id": ["T-000"], "week": [1], "target": [0.0]})
    folds = forward_chaining_schedule(dummy["week"].to_list())
    train_baseline(dummy)
    print(f"[train] generated {len(folds)} forward-chaining folds")
    print("[train] calibration plot scaffold pending data availability")


if __name__ == "__main__":
    main()

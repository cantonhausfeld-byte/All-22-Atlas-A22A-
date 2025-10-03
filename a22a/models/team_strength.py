"""Bayesian team strength scaffolding."""
from __future__ import annotations

from pathlib import Path
from typing import List

import polars as pl
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "team_strength.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def build_team_strength_table(teams: List[str]) -> pl.DataFrame:
    cfg = load_config()
    decay = cfg.get("recency_decay", 0.9)
    table = pl.DataFrame(
        {
            "team_id": teams,
            "theta": [0.0 for _ in teams],
            "ci_lower": [-0.1 for _ in teams],
            "ci_upper": [0.1 for _ in teams],
            "recency_decay": [decay for _ in teams],
        }
    )
    return table


def main() -> None:
    teams = ["T-000", "T-001"]
    table = build_team_strength_table(teams)
    print(table)


if __name__ == "__main__":
    main()

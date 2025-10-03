"""Bayesian team strength scaffold for Phase 5."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "defaults.yaml"
STAGED_DIR = REPO_ROOT / "staged"
OUTPUT_PATH = STAGED_DIR / "team_strength.parquet"


@dataclass(frozen=True)
class StrengthConfig:
    recency_decay: float
    min_games: int


def load_config() -> StrengthConfig:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        defaults = yaml.safe_load(fh) or {}
    cfg = defaults.get("models", {}).get("team_strength", {})
    return StrengthConfig(
        recency_decay=float(cfg.get("recency_decay", 0.85)),
        min_games=int(cfg.get("min_games", 3)),
    )


def run_stub(config: StrengthConfig, teams: Sequence[str]) -> pl.DataFrame:
    records = []
    for team in teams:
        records.append(
            {
                "season": 2023,
                "week": 1,
                "team_id": team,
                "theta_mean": 0.0,
                "theta_ci_lower": -0.1,
                "theta_ci_upper": 0.1,
                "samples": float(config.min_games),
            }
        )
    return pl.DataFrame(records, schema={
        "season": pl.Int64,
        "week": pl.Int64,
        "team_id": pl.Utf8,
        "theta_mean": pl.Float64,
        "theta_ci_lower": pl.Float64,
        "theta_ci_upper": pl.Float64,
        "samples": pl.Float64,
    })


def main(teams: Iterable[str] | None = None) -> None:
    config = load_config()
    STAGED_DIR.mkdir(parents=True, exist_ok=True)
    team_list = list(teams) if teams is not None else ["BUF", "KC"]
    table = run_stub(config, team_list)
    table.write_parquet(OUTPUT_PATH)
    print(
        f"[team-strength] stub complete | teams={len(team_list)} | decay={config.recency_decay}"
    )
    print(f"[team-strength] output -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

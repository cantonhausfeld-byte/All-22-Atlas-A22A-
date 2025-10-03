"""Bayesian-style team strength estimates with recency decay."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import polars as pl
import yaml

from a22a.data.contracts import validate_frame

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "team_strength.yaml"
STAGED_DIR = Path(__file__).resolve().parents[2] / "staged"


@dataclass
class TeamState:
    mean: float = 0.0
    samples: float = 1.0

    def decay(self, factor: float) -> None:
        self.mean *= factor
        self.samples = max(self.samples * factor, 1.0)

    def update(self, observation: float) -> None:
        weight = self.samples
        self.samples += 1.0
        self.mean = (self.mean * weight + observation) / self.samples


def load_config() -> Dict[str, float]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def _prepare_states(teams: Iterable[str]) -> Dict[str, TeamState]:
    return {team: TeamState() for team in teams}


def _scaled_margin(row: Mapping[str, object]) -> float:
    margin = float(row["margin"])
    total = max(float(row["total_points"]), 1.0)
    return margin / total * 10.0


def compute_team_strength(team_games: pl.DataFrame) -> pl.DataFrame:
    cfg = load_config()
    decay = float(cfg.get("recency_decay", 0.85))
    prior_scale = float(cfg.get("prior_scale", 1.0))

    team_games = team_games.sort(["season", "week", "game_datetime", "team_id"])
    teams = team_games.get_column("team_id").unique().to_list()
    states = _prepare_states(teams)

    records = []
    for row in team_games.iter_rows(named=True):
        state = states[row["team_id"]]
        state.decay(decay)
        std = prior_scale / (state.samples**0.5)
        records.append(
            {
                "season": row["season"],
                "week": row["week"],
                "team_id": row["team_id"],
                "theta_mean": state.mean,
                "theta_ci_lower": state.mean - 1.96 * std,
                "theta_ci_upper": state.mean + 1.96 * std,
                "samples": state.samples,
            }
        )
        observation = _scaled_margin(row)
        state.update(observation)

    frame = pl.DataFrame(records)
    if not frame.is_empty():
        validate_frame(frame, "team_strength")
    return frame


def main() -> None:
    team_games_path = STAGED_DIR / "team_games.parquet"
    if not team_games_path.exists():
        raise FileNotFoundError(f"Expected {team_games_path} – run `make ingest` first")
    team_games = pl.read_parquet(team_games_path)
    strength = compute_team_strength(team_games)
    output = STAGED_DIR / "team_strength.parquet"
    strength.write_parquet(output)
    print(f"[team-strength] computed {strength.height} rows -> {output}")


if __name__ == "__main__":
    main()

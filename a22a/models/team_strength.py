"""Bayesian-inspired recency-weighted team strength model (Phase 5)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl
import yaml


@dataclass
class StrengthConfig:
    seasons: list[int]
    decay_half_life_weeks: float = 6.0
    process_sigma: float = 6.0
    observation_sigma: float = 13.5


def _load_config(path: str | Path = "configs/defaults.yaml") -> dict:
    if Path(path).exists():
        return yaml.safe_load(Path(path).read_text())
    return {}


def _load_games(staged_dir: Path, seasons: Iterable[int]) -> pl.DataFrame:
    table = staged_dir / "games"
    files = list(table.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no games parquet found in {table}")
    lf = pl.scan_parquet(files)
    lf = lf.filter(pl.col("season").is_in(list(seasons)))
    df = lf.collect()
    if "kickoff_datetime" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Datetime).alias("kickoff_datetime"))
    return df


def _iter_games(df: pl.DataFrame):
    df = df.sort(["season", "week", "kickoff_datetime", "game_id"])
    for row in df.iter_rows(named=True):
        yield row


def _apply_decay(state: dict, team: str, delta: float, cfg: StrengthConfig) -> None:
    if delta <= 0:
        return
    decay = math.exp(-math.log(2) * delta / cfg.decay_half_life_weeks)
    mu, var = state[team]
    mu *= decay
    var = var * (decay**2) + cfg.process_sigma**2 * (1 - decay**2)
    state[team] = (mu, var)


def _update_team(state: dict, team: str, obs: float, cfg: StrengthConfig) -> None:
    mu, var = state[team]
    obs_var = cfg.observation_sigma**2
    k = var / (var + obs_var)
    mu = mu + k * (obs - mu)
    var = max(1e-6, (1 - k) * var)
    state[team] = (mu, var)


def _initial_state(teams: Iterable[str]) -> dict[str, tuple[float, float]]:
    return {t: (0.0, 25.0) for t in teams}


def compute_theta(games: pl.DataFrame, cfg: StrengthConfig) -> pl.DataFrame:
    teams = sorted(set(games["home_team"].to_list()) | set(games["away_team"].to_list()))
    state = _initial_state(teams)
    last_week_played: dict[str, float] = {t: 0.0 for t in teams}
    records: list[dict[str, object]] = []

    week_lookup: dict[int, dict[int, int]] = {}
    for season in games.select("season").unique().to_series().to_list():
        weeks = (
            games.filter(pl.col("season") == season)
            .select("week")
            .unique()
            .to_series()
            .to_list()
        )
        week_lookup[int(season)] = {int(week): idx + 1 for idx, week in enumerate(sorted(weeks))}

    for row in _iter_games(games):
        season = int(row["season"])
        week = int(row["week"])
        home = row["home_team"]
        away = row["away_team"]
        home_score = float(row.get("home_score", 0.0))
        away_score = float(row.get("away_score", 0.0))
        margin = home_score - away_score
        diff = week_lookup[season][week]

        for team in (home, away):
            delta = diff - last_week_played.get(team, 0.0)
            _apply_decay(state, team, delta, cfg)
            last_week_played[team] = diff

        _update_team(state, home, margin, cfg)
        _update_team(state, away, -margin, cfg)

        for team in (home, away):
            mu, var = state[team]
            stdev = math.sqrt(var)
            records.append(
                {
                    "season": season,
                    "week": week,
                    "team_id": team,
                    "theta_mean": mu,
                    "theta_lo": mu - 1.96 * stdev,
                    "theta_hi": mu + 1.96 * stdev,
                    "games_played": int(diff),
                }
            )

    return pl.DataFrame(records)


def run_model(staged_dir: Path, models_dir: Path, cfg: StrengthConfig) -> Path:
    games = _load_games(staged_dir, cfg.seasons)
    theta = compute_theta(games, cfg)
    models_dir.mkdir(parents=True, exist_ok=True)
    out = models_dir / "team_strength.parquet"
    theta.write_parquet(out)
    summary = {
        "rows": theta.height,
        "teams": sorted(theta["team_id"].unique()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (models_dir / "team_strength.json").write_text(json.dumps(summary, indent=2))
    return out


def main() -> None:
    cfg_dict = _load_config()
    seasons = cfg_dict.get("ingest", {}).get("seasons") or cfg_dict.get("model", {}).get("seasons")
    if not seasons:
        seasons = [cfg_dict.get("defaults", {}).get("season", 2023)]
    cfg = StrengthConfig(seasons=list(map(int, seasons)))
    staged = Path(cfg_dict.get("paths", {}).get("staged", "./data/staged"))
    models_dir = Path(cfg_dict.get("paths", {}).get("models", "./data/models"))
    print(f"[theta] computing strengths for seasons={cfg.seasons}")
    out = run_model(staged, models_dir, cfg)
    print(f"[theta] wrote {out}")


if __name__ == "__main__":
    main()

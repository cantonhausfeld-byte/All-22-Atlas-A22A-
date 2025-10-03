"""Drive-level hazard simulation with antithetic QMC."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl

SIM_DIR = Path("data") / "sim"
MODEL_PREDICTIONS = Path("data") / "model" / "win_probabilities.parquet"


@dataclass
class SimulationConfig:
    num_draws: int = 256
    ci_width: float = 0.05
    min_draws: int = 32


def _van_der_corput(n: int, base: int) -> np.ndarray:
    sequence = np.zeros(n)
    for i in range(n):
        denominator = 1
        index = i + 1
        value = 0.0
        while index:
            index, remainder = divmod(index, base)
            denominator *= base
            value += remainder / denominator
        sequence[i] = value
    return sequence


def quasi_monte_carlo(num_draws: int, dimension: int) -> np.ndarray:
    bases = [2, 3, 5, 7, 11, 13, 17, 19]
    usable = bases[:dimension]
    base_draws = np.column_stack([_van_der_corput(num_draws // 2, base) for base in usable])
    antithetic = 1.0 - base_draws
    draws = np.vstack([base_draws, antithetic])
    if draws.shape[0] < num_draws:
        repeats = np.tile(draws, (int(np.ceil(num_draws / draws.shape[0])), 1))
        draws = repeats[:num_draws, :]
    return draws


def _load_drives() -> pl.DataFrame:
    path = Path("staged") / "drives.parquet"
    if not path.exists():
        raise FileNotFoundError("Missing drives data – run `make ingest` first")
    return pl.read_parquet(path)


def _load_team_games() -> pl.DataFrame:
    path = Path("staged") / "team_games.parquet"
    if not path.exists():
        raise FileNotFoundError("Missing team games data – run `make ingest` first")
    return pl.read_parquet(path)


def _load_predictions() -> pl.DataFrame:
    if not MODEL_PREDICTIONS.exists():
        raise FileNotFoundError("Missing model predictions – run `make train` first")
    return pl.read_parquet(MODEL_PREDICTIONS)


def _drive_hazards(drives: pl.DataFrame) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
    hazards: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for row in drives.iter_rows(named=True):
        key = (row["game_id"], row["team_id"])
        prob = min(max(row["points"] / 7.0, 0.05), 0.95)
        hazards.setdefault(key, []).append((prob, row["points"]))
    return hazards


def _adjust_probabilities(hazards: List[Tuple[float, float]], win_prob: float) -> List[Tuple[float, float]]:
    adjusted = []
    for prob, points in hazards:
        shift = (win_prob - 0.5) * 0.2
        adjusted_prob = float(np.clip(prob + shift, 0.01, 0.99))
        adjusted.append((adjusted_prob, points))
    return adjusted


def _simulate_game(
    game_id: str,
    home_hazards: List[Tuple[float, float]],
    away_hazards: List[Tuple[float, float]],
    config: SimulationConfig,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    dimension = len(home_hazards) + len(away_hazards)
    draws = quasi_monte_carlo(config.num_draws, dimension)

    results = []
    home_wins = 0
    for iteration, draw in enumerate(draws, start=1):
        home_draws = draw[: len(home_hazards)]
        away_draws = draw[len(home_hazards) :]
        home_score = sum(points if uniform < prob else 0.0 for (prob, points), uniform in zip(home_hazards, home_draws))
        away_score = sum(points if uniform < prob else 0.0 for (prob, points), uniform in zip(away_hazards, away_draws))
        margin = home_score - away_score
        total = home_score + away_score
        if margin > 0:
            home_wins += 1
        results.append((iteration, home_score, away_score, margin, total))
        if iteration >= config.min_draws:
            p_hat = home_wins / iteration
            stderr = max(np.sqrt(p_hat * (1 - p_hat) / iteration), 1e-6)
            width = 3.92 * stderr
            if width <= config.ci_width:
                break

    result_frame = pl.DataFrame(results, schema=["iteration", "home_points", "away_points", "margin", "total_points"])
    summary = pl.DataFrame(
        {
            "game_id": [game_id],
            "simulations": [int(result_frame.height)],
            "home_win_rate": [home_wins / result_frame.height],
            "mean_margin": [result_frame["margin"].mean()],
            "mean_total": [result_frame["total_points"].mean()],
        }
    )
    return summary, result_frame


def _ladder(frame: pl.DataFrame, game_id: str) -> pl.DataFrame:
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    ladder_rows = []
    for stat in ("margin", "total_points"):
        series = frame[stat]
        for q in quantiles:
            ladder_rows.append((game_id, stat, q, series.quantile(q)))
    return pl.DataFrame(ladder_rows, schema=["game_id", "stat", "quantile", "value"])


def main() -> None:
    drives = _load_drives()
    team_games = _load_team_games()
    predictions = _load_predictions()

    hazards = _drive_hazards(drives)
    prediction_map = (
        team_games.join(predictions, on=["game_id", "team_id", "season", "week"], how="inner")
        .filter(pl.col("is_home"))
        .select(["game_id", "team_id", "opponent_id", "win_prob_calibrated"])
    )

    SIM_DIR.mkdir(parents=True, exist_ok=True)

    summaries: List[pl.DataFrame] = []
    ladders: List[pl.DataFrame] = []

    config = SimulationConfig()

    for row in prediction_map.iter_rows(named=True):
        game_id = row["game_id"]
        home_team = row["team_id"]
        away_team = row["opponent_id"]
        home_prob = float(row["win_prob_calibrated"])
        away_prob = 1.0 - home_prob
        home_hazard = _adjust_probabilities(hazards[(game_id, home_team)], home_prob)
        away_hazard = _adjust_probabilities(hazards[(game_id, away_team)], away_prob)
        summary, dist = _simulate_game(game_id, home_hazard, away_hazard, config)
        ladder = _ladder(dist, game_id)
        summaries.append(summary)
        ladders.append(ladder)
        dist.write_parquet(SIM_DIR / f"{game_id}_distribution.parquet")
        ladder.write_parquet(SIM_DIR / f"{game_id}_ladder.parquet")
        print(
            f"[sim] {game_id}: sims={summary['simulations'][0]} home_win={summary['home_win_rate'][0]:.3f}"
        )

    if summaries:
        pl.concat(summaries, how="vertical").write_parquet(SIM_DIR / "summary.parquet")
    if ladders:
        pl.concat(ladders, how="vertical").write_parquet(SIM_DIR / "ladder.parquet")


if __name__ == "__main__":
    main()

"""Simulation engine scaffold for Phase 6."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "defaults.yaml"
SIM_DIR = Path("staged") / "simulations"


@dataclass(frozen=True)
class SimulationConfig:
    sampler: str
    max_sims_per_game: int
    early_stop_margin: float


def load_config() -> SimulationConfig:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        defaults = yaml.safe_load(fh) or {}
    cfg = defaults.get("simulation", {})
    return SimulationConfig(
        sampler=str(cfg.get("qmc_sampler", "sobol")),
        max_sims_per_game=int(cfg.get("max_sims_per_game", 1024)),
        early_stop_margin=float(cfg.get("early_stop_margin", 0.01)),
    )


def quasi_monte_carlo(draws: int, dimension: int) -> np.ndarray:
    bases = [2, 3, 5, 7, 11, 13, 17, 19]
    usable = bases[: max(dimension, 1)]
    sequence = [np.arange(draws) / draws for _ in usable]
    return np.column_stack(sequence[:dimension]) if dimension else np.empty((draws, 0))


def simulate_game_stub(config: SimulationConfig, teams: Sequence[str]) -> Dict[str, object]:
    draws = quasi_monte_carlo(min(config.max_sims_per_game, 32), dimension=len(teams))
    summary = {
        "draws": draws.shape[0],
        "sampler": config.sampler,
        "early_stop_margin": config.early_stop_margin,
    }
    return summary


def main(games: Iterable[str] | None = None) -> None:
    config = load_config()
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    slate = list(games) if games is not None else ["BUF@KC"]
    results = {}
    for game_id in slate:
        summary = simulate_game_stub(config, teams=["home", "away"])
        results[game_id] = summary
    ladder_path = SIM_DIR / "fair_ladder_placeholder.parquet"
    pl.DataFrame(
        {
            "game_id": list(results.keys()),
            "sampler": [summary["sampler"] for summary in results.values()],
            "draws": [summary["draws"] for summary in results.values()],
        }
    ).write_parquet(ladder_path)
    print(
        f"[sim] stub complete | games={len(slate)} | ladder={ladder_path}"
    )


if __name__ == "__main__":
    main()

"""Synthetic ETL ingest for the Atlas Phase 1–2 deliverables."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import polars as pl

from .contracts import CONTRACTS, JOIN_KEYS, STAGED_GAME_SCHEMA, validate_frame

STAGED_DIR = Path("staged")


def ensure_staged_dir() -> Path:
    STAGED_DIR.mkdir(exist_ok=True)
    return STAGED_DIR


def _build_schedule(season: int) -> pl.DataFrame:
    games = [
        {
            "game_id": "2023WK01-BUF-KC",
            "season": season,
            "week": 1,
            "home_team": "KC",
            "away_team": "BUF",
            "home_points": 27,
            "away_points": 24,
            "game_datetime": datetime(2023, 9, 7, 20, 20),
            "venue": "GEHA Field",
        },
        {
            "game_id": "2023WK01-PHI-DAL",
            "season": season,
            "week": 1,
            "home_team": "DAL",
            "away_team": "PHI",
            "home_points": 21,
            "away_points": 17,
            "game_datetime": datetime(2023, 9, 10, 16, 25),
            "venue": "AT&T Stadium",
        },
        {
            "game_id": "2023WK02-KC-PHI",
            "season": season,
            "week": 2,
            "home_team": "PHI",
            "away_team": "KC",
            "home_points": 23,
            "away_points": 26,
            "game_datetime": datetime(2023, 9, 17, 13, 0),
            "venue": "Lincoln Financial Field",
        },
        {
            "game_id": "2023WK02-BUF-DAL",
            "season": season,
            "week": 2,
            "home_team": "BUF",
            "away_team": "DAL",
            "home_points": 31,
            "away_points": 20,
            "game_datetime": datetime(2023, 9, 17, 16, 5),
            "venue": "Highmark Stadium",
        },
    ]
    return pl.DataFrame(games)


def _build_roster(season: int) -> pl.DataFrame:
    players = [
        {"season": season, "team_id": team, "player_id": f"{team}-{idx:02d}", "position": pos, "full_name": name, "experience": exp}
        for team, roster in {
            "BUF": [("QB", "Josh Allen", 6), ("WR", "Stefon Diggs", 8)],
            "KC": [("QB", "Patrick Mahomes", 6), ("TE", "Travis Kelce", 10)],
            "PHI": [("QB", "Jalen Hurts", 4), ("WR", "A.J. Brown", 5)],
            "DAL": [("QB", "Dak Prescott", 8), ("WR", "CeeDee Lamb", 4)],
        }.items()
        for idx, (pos, name, exp) in enumerate(roster, start=1)
    ]
    return pl.DataFrame(players)


def _build_drives_and_pbp(schedule: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    pbp_rows = []
    drive_rows = []
    play_id = 1
    for game in schedule.iter_rows(named=True):
        for team_id, opponent_id, is_home in (
            (game["home_team"], game["away_team"], True),
            (game["away_team"], game["home_team"], False),
        ):
            for drive_idx in range(1, 4):
                drive_id = f"{team_id}-D{drive_idx}"
                yards = 25 + 10 * drive_idx + (3 if is_home else 0)
                points = 7.0 if drive_idx % 2 == 0 else 3.0
                result = "TD" if points >= 7 else "FG"
                drive_rows.append(
                    {
                        "game_id": game["game_id"],
                        "drive_id": drive_id,
                        "season": game["season"],
                        "week": game["week"],
                        "team_id": team_id,
                        "result": result,
                        "plays": 6 + drive_idx,
                        "yards": float(yards),
                        "points": points,
                    }
                )
                for play_offset in range(1, 4):
                    yards_gained = float(4 * play_offset + (2 if result == "TD" else 0))
                    scoring_margin = (points if result == "TD" else points - 3) + (drive_idx - 2)
                    pbp_rows.append(
                        {
                            "game_id": game["game_id"],
                            "play_id": play_id,
                            "drive_id": drive_id,
                            "season": game["season"],
                            "week": game["week"],
                            "posteam": team_id,
                            "defteam": opponent_id,
                            "yards_gained": yards_gained,
                            "scoring_margin": float(scoring_margin),
                            "success": yards_gained >= 4.0,
                        }
                    )
                    play_id += 1
    pbp = pl.DataFrame(pbp_rows)
    drives = pl.DataFrame(drive_rows)
    return {"pbp": pbp, "drives": drives}


def _build_team_games(schedule: pl.DataFrame) -> pl.DataFrame:
    home = schedule.select(
        [
            pl.col("game_id"),
            pl.col("season"),
            pl.col("week"),
            pl.col("home_team").alias("team_id"),
            pl.col("away_team").alias("opponent_id"),
            pl.lit(True).alias("is_home"),
            pl.col("home_points").alias("points_for"),
            pl.col("away_points").alias("points_against"),
            pl.col("game_datetime"),
        ]
    )
    away = schedule.select(
        [
            pl.col("game_id"),
            pl.col("season"),
            pl.col("week"),
            pl.col("away_team").alias("team_id"),
            pl.col("home_team").alias("opponent_id"),
            pl.lit(False).alias("is_home"),
            pl.col("away_points").alias("points_for"),
            pl.col("home_points").alias("points_against"),
            pl.col("game_datetime"),
        ]
    )
    team_games = pl.concat([home, away]).with_columns(
        (pl.col("points_for") - pl.col("points_against")).alias("margin"),
        (pl.col("points_for") + pl.col("points_against")).alias("total_points"),
        pl.when(pl.col("points_for") > pl.col("points_against")).then(1).otherwise(0).alias("win"),
    )
    return team_games.sort(["season", "week", "game_id", "team_id"])


def load_raw_sources() -> Dict[str, pl.DataFrame]:
    """Produce deterministic sample datasets that satisfy the data contracts."""

    season = 2023
    schedule = _build_schedule(season)
    roster = _build_roster(season)
    staged = _build_drives_and_pbp(schedule)
    staged["schedule"] = schedule
    staged["roster"] = roster
    staged["team_games"] = _build_team_games(schedule)
    return staged


def apply_contract(frames: Dict[str, pl.DataFrame]) -> None:
    for name, frame in frames.items():
        dataset_name = name if name in CONTRACTS else "team_games"
        validate_frame(frame, dataset_name)


def _stage_frame(name: str, frame: pl.DataFrame) -> Path:
    path = STAGED_DIR / f"{name}.parquet"
    frame.write_parquet(path)
    return path


def main() -> None:
    ensure_staged_dir()
    frames = load_raw_sources()
    apply_contract(frames)
    materialised = {name: _stage_frame(name, frame) for name, frame in frames.items()}
    summary = ", ".join(f"{name}={frame.height}" for name, frame in frames.items())
    print(
        "[ingest] staged datasets with schema keys",
        sorted(STAGED_GAME_SCHEMA),
        "join keys",
        sorted(JOIN_KEYS),
        f"records -> {summary}",
    )
    print("[ingest] materialised:")
    for name, path in materialised.items():
        print(f"    - {name}: {path}")


if __name__ == "__main__":
    main()

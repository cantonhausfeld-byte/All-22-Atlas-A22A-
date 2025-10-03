"""Utility sample data for offline-friendly pipelines.

The sample set mirrors a single regular-season NFL game with a handful of
plays.  It is intentionally small so that the Phase 1–6 pipelines can run in
seconds inside CI while still exercising schema, feature, and modelling paths.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl


GAME_ID = "2023_01_BUF_KC"
SEASON = 2023
WEEK = 1
HOME = "BUF"
AWAY = "KC"
KICKOFF = datetime(2023, 9, 10, 17, 25)


def _base_game_meta() -> dict[str, object]:
    return {
        "game_id": GAME_ID,
        "season": SEASON,
        "week": WEEK,
        "home_team": HOME,
        "away_team": AWAY,
        "kickoff_datetime": KICKOFF,
        "game_date": KICKOFF.date(),
        "home_coach": "S. McDermott",
        "away_coach": "A. Reid",
        "stadium": "Highmark Stadium",
        "surface": "turf",
        "home_score": 27,
        "away_score": 23,
    }


def sample_games() -> pl.DataFrame:
    meta = _base_game_meta()
    meta.update({
        "spread_line": -2.5,
        "total_line": 48.5,
        "weekday": "Sunday",
    })
    return pl.DataFrame(meta)


def sample_drives() -> pl.DataFrame:
    drives = [
        {
            "game_id": GAME_ID,
            "season": SEASON,
            "week": WEEK,
            "drive_id": f"{GAME_ID}_{i+1}",
            "posteam": HOME if i % 2 == 0 else AWAY,
            "start_period": 1 + i // 4,
            "end_period": 1 + (i + 1) // 4,
            "drive_number": i + 1,
            "drive_play_count": 6 + (i % 3),
            "drive_yards": 35 + 3 * i,
            "drive_time_seconds": 180 + 5 * i,
            "drive_result": result,
            "drive_points": pts,
            "drive_end_event": end_event,
            "posteam_score": pts_for,
            "defteam_score": pts_against,
        }
        for i, (result, pts, end_event, pts_for, pts_against) in enumerate(
            [
                ("touchdown", 7, "rush", 7, 0),
                ("field_goal", 3, "kick", 10, 7),
                ("punt", 0, "punt", 10, 10),
                ("touchdown", 7, "pass", 17, 13),
                ("field_goal", 3, "kick", 20, 16),
                ("interception", 0, "turnover", 20, 23),
                ("touchdown", 7, "pass", 27, 23),
                ("end_game", 0, "kneel", 27, 23),
            ]
        )
    ]
    return pl.DataFrame(drives)


def sample_pbp() -> pl.DataFrame:
    plays = []
    clock = 3600
    yards_line = 75
    score_home = 0
    score_away = 0
    drive = 1
    for play_id, (team, opp, yards, epa, play_type, gained_pts) in enumerate(
        [
            (HOME, AWAY, 12, 0.65, "pass", 0),
            (HOME, AWAY, 8, 0.42, "rush", 0),
            (HOME, AWAY, 10, 2.70, "pass", 7),
            (AWAY, HOME, 15, 0.80, "pass", 0),
            (AWAY, HOME, -3, -0.40, "rush", 0),
            (AWAY, HOME, 30, 3.10, "pass", 7),
            (HOME, AWAY, 18, 1.10, "pass", 0),
            (HOME, AWAY, 5, 0.30, "rush", 0),
            (HOME, AWAY, 7, 1.95, "pass", 7),
            (AWAY, HOME, 9, 0.55, "pass", 0),
            (AWAY, HOME, 6, 0.20, "rush", 0),
            (AWAY, HOME, 2, -0.05, "rush", 0),
            (HOME, AWAY, 3, -0.10, "rush", 0),
            (HOME, AWAY, 4, 0.35, "pass", 0),
            (HOME, AWAY, -2, -0.30, "rush", 0),
            (AWAY, HOME, 14, 0.75, "pass", 0),
            (AWAY, HOME, 20, 1.60, "pass", 7),
            (HOME, AWAY, 3, 0.10, "rush", 0),
            (HOME, AWAY, 17, 0.85, "pass", 0),
            (HOME, AWAY, 22, 2.40, "pass", 7),
        ],
        start=1,
    ):
        clock -= 32
        yards_line = max(0, yards_line - yards)
        is_pass = int(play_type == "pass")
        is_rush = int(play_type == "rush")
        if team == HOME:
            score_home += gained_pts
        else:
            score_away += gained_pts
        plays.append(
            {
                "game_id": GAME_ID,
                "play_id": play_id,
                "season": SEASON,
                "week": WEEK,
                "posteam": team,
                "defteam": opp,
                "drive": drive,
                "yardline_100": yards_line,
                "yards_gained": yards,
                "epa": epa,
                "play_type": play_type,
                "pass": is_pass,
                "rush": is_rush,
                "down": 1 + ((play_id - 1) % 3),
                "ydstogo": max(1, 10 - yards),
                "game_seconds_remaining": clock,
                "score_differential": score_home - score_away,
                "posteam_score": score_home,
                "defteam_score": score_away,
                "half_seconds_remaining": clock % 1800,
                "success": int(epa > 0),
            }
        )
        if gained_pts:
            drive += 1
    return pl.DataFrame(plays)


def sample_roster() -> pl.DataFrame:
    base_ts = KICKOFF - timedelta(days=3)
    rows = []
    for team, players in {
        HOME: [
            ("QB1", "Josh Allen", "QB"),
            ("WR1", "Stefon Diggs", "WR"),
            ("RB1", "James Cook", "RB"),
        ],
        AWAY: [
            ("QB2", "Patrick Mahomes", "QB"),
            ("TE1", "Travis Kelce", "TE"),
            ("RB2", "Isiah Pacheco", "RB"),
        ],
    }.items():
        for idx, (pid, name, pos) in enumerate(players, start=1):
            rows.append(
                {
                    "season": SEASON,
                    "team": team,
                    "player_id": pid,
                    "gsis_id": f"{pid}{idx:02d}",
                    "player_name": name,
                    "position": pos,
                    "depth_chart_order": idx,
                    "status": "ACT",
                    "last_updated": base_ts,
                }
            )
    return pl.DataFrame(rows)


def load_all() -> dict[str, pl.DataFrame]:
    return {
        "pbp": sample_pbp(),
        "drives": sample_drives(),
        "games": sample_games(),
        "roster": sample_roster(),
    }

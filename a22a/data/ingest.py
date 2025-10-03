"""Ingest nflverse-style data into staged parquet datasets (Phases 1–2)."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import polars as pl
import yaml

from . import sample_data
from .contracts import CONTRACTS, ContractViolation, validate_all


def _load_config(path: str | os.PathLike = "configs/defaults.yaml") -> dict:
    if Path(path).exists():
        return yaml.safe_load(Path(path).read_text())
    return {}


def _ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "pbp").mkdir(exist_ok=True)
    (path / "drives").mkdir(exist_ok=True)
    (path / "games").mkdir(exist_ok=True)
    (path / "roster").mkdir(exist_ok=True)


def _try_import(module: str):
    try:
        return __import__(module)
    except Exception:
        return None


def _load_from_nflverse(seasons: Iterable[int]) -> dict[str, pl.DataFrame] | None:
    nfl = _try_import("nfl_data_py")
    if nfl is None:
        return None
    try:
        pbp = pl.from_pandas(nfl.import_pbp_data(list(seasons), downcast=True))
        drives = pl.from_pandas(nfl.import_drives(list(seasons)))
        schedules = pl.from_pandas(nfl.import_schedules(list(seasons)))
        roster = pl.from_pandas(nfl.import_rosters(list(seasons)))
    except Exception:
        return None
    games = schedules.rename({"team": "home_team"})
    return {
        "pbp": pbp,
        "drives": drives,
        "games": games,
        "roster": roster,
    }


def _load_sources(seasons: Iterable[int]) -> dict[str, pl.DataFrame]:
    data = _load_from_nflverse(seasons)
    if data:
        return data
    # Offline-friendly fallback: bundled sample data
    return sample_data.load_all()


def _write_partitioned(df: pl.DataFrame, target: Path, *, partition_col: str = "season") -> list[Path]:
    written: list[Path] = []
    if partition_col not in df.columns:
        out = target / "part-0.parquet"
        df.write_parquet(out)
        written.append(out)
        return written
    partitions = df.partition_by(partition_col, maintain_order=True, as_dict=True)
    for season_key, part in partitions.items():
        season = season_key[0] if isinstance(season_key, tuple) else season_key
        out = target / f"season={season}" / "data.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        part.write_parquet(out)
        written.append(out)
    return written


def _enrich_games(games: pl.DataFrame) -> pl.DataFrame:
    if "kickoff_datetime" in games.columns and games["kickoff_datetime"].dtype != pl.Datetime:
        games = games.with_columns(pl.col("kickoff_datetime").str.strptime(pl.Datetime, strict=False))
    if "game_date" not in games.columns and "kickoff_datetime" in games.columns:
        games = games.with_columns(pl.col("kickoff_datetime").dt.date().alias("game_date"))
    return games


def ingest(seasons: Iterable[int], staged_dir: Path) -> dict[str, list[Path]]:
    staged_dir = staged_dir.resolve()
    _ensure_dirs(staged_dir)
    start = time.time()
    tables = _load_sources(seasons)
    tables["games"] = _enrich_games(tables["games"])

    validate_all(tables, CONTRACTS)

    written_paths: dict[str, list[Path]] = defaultdict(list)
    for name, df in tables.items():
        target = staged_dir / name
        written_paths[name] = _write_partitioned(df, target)

    registry = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seasons": list(seasons),
        "tables": {k: [str(p) for p in v] for k, v in written_paths.items()},
        "duration_seconds": round(time.time() - start, 3),
    }
    (staged_dir / "run_registry.json").write_text(json.dumps(registry, indent=2))
    return written_paths


def main() -> None:
    cfg = _load_config()
    seasons = cfg.get("ingest", {}).get("seasons", [sample_data.SEASON])
    staged_root = Path(cfg.get("paths", {}).get("staged", "./data/staged"))
    print(f"[ingest] starting seasons={seasons} -> {staged_root}")
    try:
        written = ingest(seasons, staged_root)
    except ContractViolation as err:
        raise SystemExit(f"[ingest] schema validation failed: {err}")
    print("[ingest] wrote:")
    for name, paths in written.items():
        for path in paths:
            print(f"  - {name}: {path}")
    print("[ingest] completed successfully")


if __name__ == "__main__":
    main()

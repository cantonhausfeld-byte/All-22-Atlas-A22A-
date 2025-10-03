"""Baseline model training scaffold for Phase 4."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "defaults.yaml"
FEATURE_REFERENCE = Path("data") / "features" / "reference.parquet"
MODEL_DIR = Path("data") / "models"
PLOTS_DIR = Path("reports") / "baseline"
CALIBRATION_PLOT = PLOTS_DIR / "calibration_placeholder.png"


@dataclass(frozen=True)
class Fold:
    train_weeks: Sequence[int]
    validation_weeks: Sequence[int]


def load_config() -> Dict[str, object]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_features() -> pl.DataFrame:
    if FEATURE_REFERENCE.exists():
        return pl.read_parquet(FEATURE_REFERENCE)
    return pl.DataFrame(
        {
            "season": [],
            "week": [],
            "game_id": [],
            "team_id": [],
            "target": [],
        }
    )


def forward_chaining_schedule(weeks: Iterable[int], purge_gap: int) -> List[Fold]:
    ordered = sorted(set(int(week) for week in weeks))
    folds: List[Fold] = []
    for idx in range(1, len(ordered)):
        validation_week = ordered[idx]
        train_cutoff = validation_week - purge_gap
        train_weeks = [week for week in ordered[:idx] if week <= train_cutoff]
        if not train_weeks:
            continue
        folds.append(Fold(train_weeks=train_weeks, validation_weeks=[validation_week]))
    return folds


def calibration_stub(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("calibration placeholder", encoding="utf-8")
    return path


def train_stub(features: pl.DataFrame, folds: Sequence[Fold]) -> Dict[str, object]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "baseline_model_placeholder.npz"
    np.savez(model_path, metadata="baseline placeholder")
    return {
        "folds": len(folds),
        "model_path": model_path,
    }


def main() -> None:
    config = load_config()
    feature_table = load_features()
    baseline_cfg = config.get("models", {}).get("baseline", {})
    purge_gap = int(baseline_cfg.get("purging_weeks", 1))
    folds = forward_chaining_schedule(feature_table.get_column("week") if "week" in feature_table.columns else [], purge_gap)
    artefacts = train_stub(feature_table, folds)
    calibration_stub(CALIBRATION_PLOT)
    print(
        "[train] baseline stub complete | folds=%s | model=%s"
        % (artefacts["folds"], artefacts["model_path"])
    )
    print(f"[train] calibration placeholder -> {CALIBRATION_PLOT}")


if __name__ == "__main__":
    main()

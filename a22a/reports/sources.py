"""Helpers for resolving report artifacts in a defensive way."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Callable, Iterable

import pandas as pd
import yaml

_DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parents[2] / "configs" / "defaults.yaml"


def latest(path_glob: str) -> pathlib.Path | None:
    """Return the lexicographically latest path matching ``path_glob``."""

    matches = sorted(pathlib.Path(".").glob(path_glob))
    return matches[-1] if matches else None


def _load_with(loader: Callable[[pathlib.Path], Any], candidates: Iterable[pathlib.Path]) -> tuple[Any | None, pathlib.Path | None]:
    for path in sorted(candidates):
        try:
            return loader(path), path
        except Exception:
            continue
    return None, None


def load_latest_parquet_or_csv(glob_pat: str) -> tuple[pd.DataFrame | None, pathlib.Path | None]:
    """Load the most recent parquet/csv file for a glob pattern."""

    paths = sorted(pathlib.Path(".").glob(glob_pat))
    if not paths:
        return None, None

    def _loader(path: pathlib.Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    return _load_with(_loader, paths[::-1])


def load_latest_json(glob_pat: str) -> tuple[dict[str, Any] | list[Any] | None, pathlib.Path | None]:
    paths = sorted(pathlib.Path(".").glob(glob_pat))
    if not paths:
        return None, None

    def _loader(path: pathlib.Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    return _load_with(_loader, paths[::-1])


def reports_out_dir(config_path: str | pathlib.Path = _DEFAULT_CONFIG) -> pathlib.Path:
    """Resolve the configured reports output directory (default: ``./reports``)."""

    cfg_path = pathlib.Path(config_path)
    if not cfg_path.exists():
        fallback = pathlib.Path("configs/defaults.yaml")
        if fallback.exists():
            cfg_path = fallback
    out_dir = pathlib.Path("reports")
    if cfg_path.exists():
        try:
            config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            reports_cfg = config.get("reports", {}) if isinstance(config, dict) else {}
            configured = reports_cfg.get("out_dir") if isinstance(reports_cfg, dict) else None
            if configured:
                out_dir = pathlib.Path(configured)
        except Exception:
            pass
    return out_dir


__all__ = [
    "latest",
    "load_latest_parquet_or_csv",
    "load_latest_json",
    "reports_out_dir",
]

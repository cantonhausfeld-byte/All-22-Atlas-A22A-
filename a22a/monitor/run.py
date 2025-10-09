"""Monitoring orchestrator for A22A health checks."""

from __future__ import annotations

import datetime as _dt
import json
import math
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml

from .alerts import AlertsClient

DEFAULT_CONFIG_PATH = "configs/defaults.yaml"
ARTIFACT_DIR = pathlib.Path("artifacts/monitor")
META_DIR = pathlib.Path("artifacts/meta")
PORTFOLIO_DIR = pathlib.Path("artifacts/portfolio")
MARKET_DIR = pathlib.Path("artifacts/market")


@dataclass(slots=True)
class MetricRecord:
    name: str
    value: Any
    target: Any | None
    status: str


STATUS_ORDER = {"ok": 0, "warn": 1, "fail": 2}


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.timezone.utc)


def _load_config(path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    cfg_path = pathlib.Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _latest_path(directory: pathlib.Path, pattern: str) -> pathlib.Path | None:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]


def _load_json(path: pathlib.Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _reliability_slope(bins: Iterable[dict[str, Any]]) -> float | None:
    df = pd.DataFrame(bins)
    if df.empty:
        return None
    if "confidence" not in df.columns or "accuracy" not in df.columns:
        return None
    if df["confidence"].nunique(dropna=True) < 2:
        return None
    x = df["confidence"].astype(float).to_numpy()
    y = df["accuracy"].astype(float).to_numpy()
    weights = df.get("count")
    try:
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            slope = np.polyfit(x, y, 1, w=w)[0]
        else:
            slope = np.polyfit(x, y, 1)[0]
    except Exception:
        return None
    return float(slope)


def _status_from_values(*statuses: str) -> str:
    if not statuses:
        return "ok"
    return max(statuses, key=lambda s: STATUS_ORDER.get(s, 1))


def _compare_lower(value: float | None, target: float, warn_ratio: float = 1.25) -> str:
    if value is None or math.isnan(value):
        return "warn"
    if value <= target:
        return "ok"
    if value <= target * warn_ratio:
        return "warn"
    return "fail"


def _compare_range(value: float | None, lower: float, upper: float, slack: float = 0.1) -> str:
    if value is None or math.isnan(value):
        return "warn"
    if lower <= value <= upper:
        return "ok"
    if (lower - slack) <= value <= (upper + slack):
        return "warn"
    return "fail"


def _load_calibration_report() -> tuple[dict[str, Any] | None, pathlib.Path | None]:
    path = _latest_path(META_DIR, "calibration_report_*.json")
    return _load_json(path), path


def _load_portfolio_picks() -> tuple[pd.DataFrame, pathlib.Path | None]:
    picks_path = _latest_path(PORTFOLIO_DIR, "picks_week_*.parquet")
    if picks_path is None:
        picks_path = _latest_path(PORTFOLIO_DIR, "picks_week_*.csv")
    if picks_path is None:
        return pd.DataFrame(), None
    try:
        if picks_path.suffix == ".csv":
            df = pd.read_csv(picks_path)
        else:
            df = pd.read_parquet(picks_path)
    except Exception:
        return pd.DataFrame(), picks_path
    return df, picks_path


def _load_clv_table() -> tuple[pd.DataFrame, pathlib.Path | None]:
    clv_path = _latest_path(MARKET_DIR, "clv_*.parquet")
    if clv_path is None:
        return pd.DataFrame(), None
    try:
        df = pd.read_parquet(clv_path)
    except Exception:
        return pd.DataFrame(), clv_path
    return df, clv_path


def _calibration_details(report: dict[str, Any] | None, cfg: dict[str, Any]) -> dict[str, Any]:
    ece_val = float(report.get("ece")) if report and "ece" in report else None
    brier_val = float(report.get("brier")) if report and "brier" in report else None
    slope_val = _reliability_slope(report.get("reliability_bins", [])) if report else None

    ece_target = float(cfg.get("ece_target", 0.03) or 0.03)
    brier_target = float(cfg.get("brier_target", 0.25) or 0.25)
    slope_min = float(cfg.get("reliability_slope_min", 0.85) or 0.85)
    slope_max = float(cfg.get("reliability_slope_max", 1.15) or 1.15)

    ece_status = _compare_lower(ece_val, ece_target)
    brier_status = _compare_lower(brier_val, brier_target)
    slope_status = _compare_range(slope_val, slope_min, slope_max)

    metrics = {
        "ece": {"value": ece_val, "target": ece_target, "status": ece_status},
        "brier": {"value": brier_val, "target": brier_target, "status": brier_status},
        "reliability_slope": {
            "value": slope_val,
            "target": {"min": slope_min, "max": slope_max},
            "status": slope_status,
        },
    }

    status = _status_from_values(ece_status, brier_status, slope_status)
    detail = {"status": status, "metrics": metrics}
    if report is None:
        detail["message"] = "calibration report not found"
    return detail


def _coverage_details(report: dict[str, Any] | None, cfg: dict[str, Any]) -> dict[str, Any]:
    nominal = float(cfg.get("min_coverage", 0.9) or 0.9)
    tolerance = float(cfg.get("coverage_tolerance", 0.02) or 0.02)
    empirical = None
    if report:
        binary = (report.get("conformal") or {}).get("binary") or {}
        value = binary.get("empirical")
        if value is not None:
            empirical = float(value)

    if empirical is None:
        status = "warn"
    else:
        diff = abs(empirical - nominal)
        if diff <= tolerance:
            status = "ok"
        elif diff <= tolerance * 2:
            status = "warn"
        else:
            status = "fail"

    metrics = {
        "empirical": {
            "value": empirical,
            "target": nominal,
            "tolerance": tolerance,
            "status": status,
        }
    }
    detail = {"status": status, "metrics": metrics}
    if empirical is None:
        detail["message"] = "empirical coverage unavailable"
    return detail


def _clv_book_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "clv_bps" not in df.columns:
        return []
    records: list[dict[str, Any]] = []
    grouped = df.groupby("book", dropna=False)
    for book, group in grouped:
        clv = group["clv_bps"].astype(float)
        records.append(
            {
                "book": None if pd.isna(book) else str(book),
                "mean_bps": float(clv.mean()),
                "median_bps": float(clv.median()),
                "positive_rate": float((clv > 0).mean()),
                "count": int(len(group)),
            }
        )
    return records


def _clv_details(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    target = float(cfg.get("clv_bps_target", 0.0) or 0.0)
    positive_target = float(cfg.get("clv_positive_rate_target", 0.5) or 0.5)

    if df.empty or "clv_bps" not in df.columns:
        return {
            "status": "warn",
            "metrics": {
                "mean_bps": {"value": None, "target": target, "status": "warn"},
                "median_bps": {"value": None, "target": target, "status": "warn"},
                "positive_rate": {
                    "value": None,
                    "target": positive_target,
                    "status": "warn",
                },
            },
            "message": "no CLV artifacts available",
            "books": [],
        }

    clv_series = df["clv_bps"].astype(float)
    mean_val = float(clv_series.mean())
    median_val = float(clv_series.median())
    positive_rate = float((clv_series > 0).mean())

    def _status(value: float, tgt: float, warn_band: float) -> str:
        if value >= tgt:
            return "ok"
        if value >= tgt - warn_band:
            return "warn"
        return "fail"

    mean_status = _status(mean_val, target, 10.0)
    median_status = _status(median_val, target, 10.0)
    pos_status = _status(positive_rate, positive_target, 0.1)

    metrics = {
        "mean_bps": {"value": mean_val, "target": target, "status": mean_status},
        "median_bps": {"value": median_val, "target": target, "status": median_status},
        "positive_rate": {
            "value": positive_rate,
            "target": positive_target,
            "status": pos_status,
        },
    }

    status = _status_from_values(mean_status, median_status, pos_status)
    return {"status": status, "metrics": metrics, "books": _clv_book_summary(df)}


def _slo_details(runtime_s: float, cfg: dict[str, Any]) -> dict[str, Any]:
    budget = float(cfg.get("max_runtime_s", 60) or 60)
    if runtime_s <= budget:
        status = "ok"
    elif runtime_s <= budget * 1.5:
        status = "warn"
    else:
        status = "fail"
    metrics = {
        "runtime_s": {"value": runtime_s, "target": budget, "status": status}
    }
    return {"status": status, "metrics": metrics}


def _gather_summary_rows(details: dict[str, Any]) -> list[MetricRecord]:
    rows: list[MetricRecord] = []
    for section, info in details.items():
        metrics = info.get("metrics", {})
        for name, meta in metrics.items():
            value = meta.get("value")
            if isinstance(value, float):
                value = round(value, 4)
            target = meta.get("target")
            rows.append(
                MetricRecord(
                    name=f"{section}.{name}",
                    value=value,
                    target=target,
                    status=str(meta.get("status", "warn")),
                )
            )
    return rows


def _print_summary(details: dict[str, Any]) -> None:
    rows = _gather_summary_rows(details)
    if not rows:
        return
    print("[monitor] summary")
    header = ("metric", "value", "target", "status")
    formatted = [header]
    for row in rows:
        target = row.target
        if isinstance(target, dict):
            target = json.dumps(target, sort_keys=True)
        formatted.append((row.name, str(row.value), str(target), row.status))

    widths = [max(len(col[idx]) for col in formatted) for idx in range(4)]
    for idx, line in enumerate(formatted):
        prefix = "[monitor]" if idx == 0 else "          "
        print(
            f"{prefix} {line[0].ljust(widths[0])}  "
            f"{line[1].ljust(widths[1])}  {line[2].ljust(widths[2])}  {line[3]}"
        )


def run_monitor(config_path: str = DEFAULT_CONFIG_PATH) -> Tuple[dict[str, Any], pathlib.Path]:
    start = time.perf_counter()

    config = _load_config(config_path)
    monitoring_cfg = config.get("monitoring", {}) or {}
    alerts_cfg = config.get("alerts", {}) or {}

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    calibration_report, calibration_path = _load_calibration_report()
    picks_df, picks_path = _load_portfolio_picks()
    clv_df, clv_path = _load_clv_table()

    calibration_details = _calibration_details(calibration_report, monitoring_cfg)
    coverage_details = _coverage_details(calibration_report, monitoring_cfg)
    clv_details = _clv_details(clv_df, monitoring_cfg)

    runtime_s = time.perf_counter() - start
    slo_details = _slo_details(runtime_s, monitoring_cfg)

    details = {
        "calibration": calibration_details,
        "coverage": coverage_details,
        "clv": clv_details,
        "slo": slo_details,
    }

    overall_status = _status_from_values(*(info["status"] for info in details.values()))

    now_utc = _utcnow()
    timestamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
    payload: dict[str, Any] = {
        "generated_at": now_utc.isoformat().replace("+00:00", "Z"),
        "status": overall_status,
        "details": details,
        "artifacts": {
            "calibration_report": calibration_path.name if calibration_path else None,
            "portfolio_picks": picks_path.name if picks_path else None,
            "clv_table": clv_path.name if clv_path else None,
        },
        "portfolio_snapshot": {
            "rows": int(len(picks_df)),
            "active": int(picks_df.get("stake_amount", pd.Series(dtype=float)).gt(0).sum())
            if not picks_df.empty
            else 0,
        },
    }

    output_path = ARTIFACT_DIR / f"health_{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    _print_summary(details)

    alert_channels = monitoring_cfg.get("alert_channels", []) or []
    alerts_client = AlertsClient(alerts_cfg, alert_channels)
    if overall_status != "ok":
        alerts_client.notify({"kind": "monitoring.health", **payload})
    else:
        print("[monitor] all checks within thresholds — no alerts sent")

    print(f"[monitor] wrote {output_path}")
    return payload, output_path


def main() -> None:
    run_monitor()


if __name__ == "__main__":  # pragma: no cover - CLI hook
    main()


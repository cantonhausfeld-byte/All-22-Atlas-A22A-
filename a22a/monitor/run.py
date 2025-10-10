"""Phase 18 monitoring implementation with metric aggregation and alerting."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

from a22a.metrics.calibration import brier_score as compute_brier
from a22a.metrics.calibration import ece as compute_ece
from a22a.reports.sources import (
    load_latest_json,
    load_latest_parquet_or_csv,
    reports_out_dir,
)

from .alerts import AlertPayload, AlertsClient, send_slack

DEFAULT_CONFIG_PATH = pathlib.Path("configs/defaults.yaml")
ARTIFACT_DIR = pathlib.Path("artifacts/monitor")
BACKTEST_DIR = pathlib.Path("artifacts/backtest")
DOCTOR_LOG = pathlib.Path("artifacts/logs/doctor_runs.jsonl")
HEALTH_PREFIX = "health_"

Status = str


@dataclass(slots=True)
class MetricCheck:
    name: str
    status: Status
    value: Any
    target: Any | None = None
    reason: str | None = None
    extra: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "value": self.value,
        }
        if self.target is not None:
            payload["target"] = self.target
        if self.reason:
            payload["reason"] = self.reason
        if self.extra:
            payload.update(self.extra)
        return payload


def _load_config(path: pathlib.Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_reports_summary() -> dict[str, Any] | None:
    summary_path = reports_out_dir() / "summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _latest_backtest_summary() -> dict[str, Any] | None:
    if not BACKTEST_DIR.exists():
        return None
    candidates = sorted(BACKTEST_DIR.glob("summary_*.json"))
    for path in reversed(candidates):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return None


def _load_meta_probabilities() -> pd.DataFrame | None:
    df, _ = load_latest_parquet_or_csv("artifacts/meta/final_probs_*.parquet")
    return df


def _load_calibration_report() -> dict[str, Any] | None:
    report, _ = load_latest_json("artifacts/meta/calibration_report_*.json")
    if isinstance(report, dict):
        return report
    return None


def _load_clv_snapshot() -> pd.DataFrame | None:
    df, _ = load_latest_parquet_or_csv("artifacts/market/clv_*.parquet")
    return df


def _load_portfolio_snapshot() -> pd.DataFrame | None:
    df, _ = load_latest_parquet_or_csv("artifacts/portfolio/picks_week_*.parquet")
    return df


def _synthetic_outcomes(probs: pd.Series) -> pd.Series:
    outcomes: List[int] = []
    for idx, value in probs.fillna(0.5).clip(0.0, 1.0).items():
        key = f"{idx}-{float(value):.6f}".encode("utf-8")
        hashed = hashlib.blake2s(key, digest_size=8).digest()
        draw = int.from_bytes(hashed, "big") / float(1 << 64)
        outcomes.append(1 if draw < float(value) else 0)
    return pd.Series(outcomes, index=probs.index)


def _calibration_metrics(
    calibration_report: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    meta_df: pd.DataFrame | None,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    source = None
    ece_val: Optional[float] = None
    brier_val: Optional[float] = None

    if calibration_report:
        ece_val = float(calibration_report.get("ece", math.nan))
        brier_val = float(calibration_report.get("brier", math.nan))
        source = "calibration_report"

    if (ece_val is None or math.isnan(ece_val)) and summary:
        calib = summary.get("calibration") or {}
        ece_val = float(calib.get("ece", math.nan))
        brier_val = float(calib.get("brier", math.nan))
        source = source or "reports_summary"

    if (ece_val is None or math.isnan(ece_val)) and meta_df is not None and not meta_df.empty:
        probs = pd.to_numeric(meta_df.get("p_home"), errors="coerce").fillna(0.5).clip(0.0, 1.0)
        outcomes = _synthetic_outcomes(probs)
        try:
            ece_val = float(compute_ece(probs, outcomes, bins=bins))
            brier_val = float(compute_brier(probs, outcomes))
            source = source or "synthetic"
        except Exception:
            ece_val = math.nan
            brier_val = math.nan

    return {
        "ece": ece_val,
        "brier": brier_val,
        "source": source or "unknown",
        "samples": int(meta_df.shape[0]) if isinstance(meta_df, pd.DataFrame) else None,
    }


def _coverage_metrics(
    calibration_report: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    monitoring_cfg: dict[str, Any],
    meta_df: pd.DataFrame | None,
) -> dict[str, Any]:
    target = float(monitoring_cfg.get("min_coverage", 0.9))
    empirical: Optional[float] = None
    nominal: Optional[float] = None
    source = None

    binary = (
        (calibration_report or {})
        .get("conformal", {})
        .get("binary", {})
        if calibration_report
        else {}
    )
    empirical = float(binary.get("empirical", math.nan)) if binary else None
    nominal = float(binary.get("nominal", math.nan)) if binary else None
    if binary:
        source = "calibration_report"

    if (empirical is None or math.isnan(empirical)) and summary:
        cal = summary.get("calibration", {})
        conf = cal.get("conformal", {}) if isinstance(cal, dict) else {}
        binary = conf.get("binary", {}) if isinstance(conf, dict) else {}
        if binary:
            empirical = float(binary.get("empirical", math.nan))
            nominal = float(binary.get("nominal", math.nan))
            source = source or "reports_summary"

    if (empirical is None or math.isnan(empirical)) and meta_df is not None and not meta_df.empty:
        probs = pd.to_numeric(meta_df.get("p_home"), errors="coerce").fillna(0.5).clip(0.0, 1.0)
        outcomes = _synthetic_outcomes(probs)
        residuals = (probs - outcomes).abs()
        radius = float(monitoring_cfg.get("synthetic_radius", 0.15))
        empirical = float((residuals <= radius).mean())
        nominal = float(1.0 - radius)
        source = source or "synthetic"

    return {
        "target": target,
        "empirical": empirical,
        "nominal": nominal,
        "source": source or "unknown",
    }


def _clv_metrics(clv_df: pd.DataFrame | None) -> dict[str, Any]:
    if clv_df is None or clv_df.empty:
        return {
            "status": "missing",
            "samples": 0,
            "mean_bps": None,
            "median_bps": None,
            "positive_pct": None,
            "by_book": [],
        }

    working = clv_df.copy()
    series = pd.to_numeric(working.get("clv_bps"), errors="coerce").dropna()
    if series.empty:
        return {
            "status": "missing",
            "samples": 0,
            "mean_bps": None,
            "median_bps": None,
            "positive_pct": None,
            "by_book": [],
        }

    by_book: List[dict[str, Any]] = []
    if "book" in working.columns:
        grouped = (
            working.assign(clv_bps=series)
            .dropna(subset=["clv_bps"])
            .groupby(working["book"].astype(str), dropna=True)
            ["clv_bps"]
        )
        for book, values in grouped:
            vals = pd.to_numeric(values, errors="coerce").dropna()
            if vals.empty:
                continue
            by_book.append(
                {
                    "book": str(book),
                    "mean_bps": float(vals.mean()),
                    "median_bps": float(vals.median()),
                    "count": int(vals.size),
                }
            )

    return {
        "status": "ok",
        "samples": int(series.size),
        "mean_bps": float(series.mean()),
        "median_bps": float(series.median()),
        "positive_pct": float((series > 0).mean()),
        "by_book": sorted(by_book, key=lambda row: row.get("book", "")),
    }


def _slo_metrics(
    monitoring_cfg: dict[str, Any],
    runtime_s: float,
) -> dict[str, Any]:
    target = float(monitoring_cfg.get("max_runtime_s", 600.0))
    doctor_runtime: Optional[float] = None
    if DOCTOR_LOG.exists():
        for line in DOCTOR_LOG.read_text(encoding="utf-8").splitlines()[::-1]:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("task") == "monitor":
                doctor_runtime = float(payload.get("runtime_s", math.nan))
                break

    observed = doctor_runtime if doctor_runtime and not math.isnan(doctor_runtime) else runtime_s
    source = "doctor" if doctor_runtime else "local"
    return {
        "target": target,
        "recent_runtime_s": float(observed),
        "source": source,
    }


def _evaluate_checks(monitoring_cfg: dict[str, Any], metrics: dict[str, Any]) -> Tuple[Status, List[str], Dict[str, MetricCheck]]:
    checks: Dict[str, MetricCheck] = {}
    reasons: List[str] = []

    def _status_from_delta(value: Optional[float], target: float, *, higher_is_better: bool) -> MetricCheck:
        if value is None or math.isnan(value):
            return MetricCheck(
                name="missing",
                status="warn",
                value=value,
                target=target,
                reason="metric missing",
            )
        delta = value - target if higher_is_better else target - value
        if delta >= 0:
            status: Status = "ok"
        elif delta > -0.05 if higher_is_better else delta > -0.01:
            status = "warn"
        else:
            status = "fail"
        reason = None
        if status != "ok":
            direction = ">=" if higher_is_better else "<="
            reason = f"expected {direction} {target:.3f} got {value:.3f}"
        return MetricCheck(name="", status=status, value=value, target=target, reason=reason)

    calib = metrics.get("calibration", {})
    ece_target = float(monitoring_cfg.get("ece_target", 0.03))
    ece_val = calib.get("ece") if isinstance(calib, dict) else None
    check = _status_from_delta(ece_val, ece_target, higher_is_better=False)
    check.name = "calibration_ece"
    check.extra = {"source": calib.get("source"), "brier": calib.get("brier")}
    checks[check.name] = check
    if check.reason:
        reasons.append(f"ECE {check.reason}")

    coverage = metrics.get("coverage", {})
    coverage_target = float(coverage.get("target", monitoring_cfg.get("min_coverage", 0.9)))
    coverage_val = coverage.get("empirical") if isinstance(coverage, dict) else None
    check = _status_from_delta(coverage_val, coverage_target, higher_is_better=True)
    check.name = "coverage"
    check.extra = {"source": coverage.get("source"), "nominal": coverage.get("nominal")}
    checks[check.name] = check
    if check.reason:
        reasons.append(f"Coverage {check.reason}")

    clv = metrics.get("clv", {})
    clv_target = float(monitoring_cfg.get("clv_bps_target", 0.0))
    clv_val = clv.get("mean_bps") if isinstance(clv, dict) else None
    check = _status_from_delta(clv_val, clv_target, higher_is_better=True)
    check.name = "clv"
    check.extra = {"samples": clv.get("samples"), "positive_pct": clv.get("positive_pct")}
    checks[check.name] = check
    if check.reason:
        reasons.append(f"CLV {check.reason}")

    slo = metrics.get("slo", {})
    slo_target = float(slo.get("target", monitoring_cfg.get("max_runtime_s", 600.0)))
    slo_val = slo.get("recent_runtime_s") if isinstance(slo, dict) else None
    check = _status_from_delta(slo_val, slo_target, higher_is_better=False)
    check.name = "slo"
    check.extra = {"source": slo.get("source")}
    checks[check.name] = check
    if check.reason:
        reasons.append(f"Runtime {check.reason}")

    status_order = {"ok": 0, "warn": 1, "fail": 2}
    overall = "ok"
    for metric in checks.values():
        if status_order[metric.status] > status_order[overall]:
            overall = metric.status

    return overall, reasons, checks


def run_monitor(config_path: pathlib.Path | None = None) -> Tuple[dict[str, Any], pathlib.Path]:
    start = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(config_path or DEFAULT_CONFIG_PATH)
    monitoring_cfg = cfg.get("monitoring", {}) or {}
    alerts_cfg = cfg.get("alerts", {}) or {}

    meta_df = _load_meta_probabilities()
    calibration_report = _load_calibration_report()
    reports_summary = _load_reports_summary()
    clv_df = _load_clv_snapshot()

    metrics: dict[str, Any] = {}
    metrics["calibration"] = _calibration_metrics(calibration_report, reports_summary, meta_df)
    metrics["coverage"] = _coverage_metrics(calibration_report, reports_summary, monitoring_cfg, meta_df)
    metrics["clv"] = _clv_metrics(clv_df)

    backtest_summary = _latest_backtest_summary()
    portfolio_df = _load_portfolio_snapshot()
    metrics["backtest"] = backtest_summary
    metrics["portfolio"] = {"has_picks": bool(portfolio_df is not None and not portfolio_df.empty)}

    runtime_s = time.time() - start
    metrics["slo"] = _slo_metrics(monitoring_cfg, runtime_s)

    status, reasons, checks = _evaluate_checks(monitoring_cfg, metrics)

    artifacts: Dict[str, Any] = {}
    if isinstance(calibration_report, dict):
        artifacts["calibration_report"] = True
    if isinstance(backtest_summary, dict):
        artifacts["backtest_summary"] = True
    if meta_df is not None:
        artifacts["meta_final_probs"] = True
    if clv_df is not None and not clv_df.empty:
        artifacts["market_clv"] = True
    if portfolio_df is not None and not portfolio_df.empty:
        artifacts["portfolio_picks"] = True

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reasons": reasons,
        "metrics": {name: metric.as_dict() for name, metric in checks.items()},
        "details": metrics,
        "artifacts": artifacts,
        "timings": {"runtime_s": round(runtime_s, 3)},
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_path = ARTIFACT_DIR / f"{HEALTH_PREFIX}{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    channels: Iterable[str] = monitoring_cfg.get("alert_channels", []) or []
    if status != "ok" and channels:
        message = {
            "title": "A22A Monitor",
            "status": status,
            "reasons": reasons,
            "metrics": {name: metric.as_dict() for name, metric in checks.items()},
        }
        if "slack" in [channel.lower() for channel in channels]:
            send_slack(message, webhook_env=alerts_cfg.get("slack_webhook_env", "SLACK_WEBHOOK_URL"))
        remaining = [ch for ch in channels if ch.lower() != "slack"]
        if remaining:
            client = AlertsClient(alerts_cfg)
            payload_obj = AlertPayload(title="A22A Monitor", status=status, body=message)
            for channel in remaining:
                client.send(channel, payload_obj)

    print(f"[monitor] status={status} reasons={len(reasons)} runtime={runtime_s:.2f}s")
    print(f"[monitor] wrote {output_path.relative_to(pathlib.Path('.'))}")
    return payload, output_path


def main() -> None:
    run_monitor()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()

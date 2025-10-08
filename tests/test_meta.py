import json
import pathlib
import subprocess
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from a22a.meta.conformal import split_conformal_binary
from a22a.metrics.calibration import ece


def _run_meta_module() -> Tuple[pathlib.Path, pathlib.Path]:
    meta_dir = pathlib.Path("artifacts/meta")
    meta_dir.mkdir(parents=True, exist_ok=True)
    before_probs = set(meta_dir.glob("final_probs_*.parquet"))
    before_reports = set(meta_dir.glob("calibration_report_*.json"))

    result = subprocess.run(
        [sys.executable, "-m", "a22a.meta.run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "conformal coverage" in result.stdout

    after_probs = set(meta_dir.glob("final_probs_*.parquet"))
    after_reports = set(meta_dir.glob("calibration_report_*.json"))
    diff_probs = sorted(after_probs - before_probs)
    diff_reports = sorted(after_reports - before_reports)
    assert diff_probs and diff_reports, "meta run must write new artifacts"
    new_prob = diff_probs[-1]
    new_report = diff_reports[-1]
    return new_prob, new_report


def test_meta_pipeline_outputs():
    prob_path, report_path = _run_meta_module()

    final_df = pd.read_parquet(prob_path)
    assert list(final_df.columns) == ["game_id", "p_home", "p_away"]
    assert final_df["p_home"].between(0, 1).all()
    assert final_df["p_away"].between(0, 1).all()
    assert np.allclose(final_df["p_home"] + final_df["p_away"], 1.0, atol=1e-6)

    report = json.loads(report_path.read_text())
    for key in ["ece", "brier", "log_loss", "calibration_method", "conformal"]:
        assert key in report
    assert "binary" in report["conformal"]
    assert "nominal" in report["conformal"]["binary"]


def test_calibration_metrics_sanity():
    probs = np.array([0.0, 1.0])
    labels = np.array([0, 1])
    assert ece(probs, labels, bins=2) == pytest.approx(0.0, abs=1e-8)


def test_conformal_binary_coverage():
    probs = pd.Series(np.linspace(0.1, 0.9, 200))
    labels = pd.Series((probs > 0.5).astype(int))
    result = split_conformal_binary(probs, labels, coverage=0.9)
    residuals = np.abs(labels - probs)
    empirical = float((residuals <= result["q"]).mean())
    assert empirical >= 0.85

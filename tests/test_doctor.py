from __future__ import annotations

from pathlib import Path

import pytest

from a22a.tools import doctor


def test_validate_env_missing():
    with pytest.raises(EnvironmentError):
        doctor.validate_env(env={})


def test_check_secrets_not_hardcoded(tmp_path: Path):
    module = tmp_path / "module.py"
    module.write_text("OPENAI_API_KEY = 'abc'", encoding="utf-8")
    with pytest.raises(RuntimeError):
        doctor.check_secrets_not_hardcoded(tmp_path)


def test_scan_feature_code_for_odds(tmp_path: Path):
    feature = tmp_path / "feature.py"
    feature.write_text("# using theoddsapi.com", encoding="utf-8")
    with pytest.raises(RuntimeError):
        doctor.scan_feature_code_for_odds(tmp_path)

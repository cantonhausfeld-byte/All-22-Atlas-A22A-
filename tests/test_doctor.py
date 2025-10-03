from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from a22a.data import contracts
from a22a.tools import doctor


def test_validate_env_missing():
    with pytest.raises(EnvironmentError):
        doctor.validate_env(env={})


def test_check_secrets_not_hardcoded(tmp_path: Path):
    module = tmp_path / "module.py"
    module.write_text("OPENAI_API_KEY = 'abc'", encoding="utf-8")
    with pytest.raises(RuntimeError):
        doctor.check_secrets_not_hardcoded(tmp_path)


def test_scan_code_for_odds(tmp_path: Path):
    feature = tmp_path / "feature.py"
    feature.write_text("# using theoddsapi.com", encoding="utf-8")
    with pytest.raises(RuntimeError):
        doctor.scan_code_for_odds([tmp_path])


def test_contract_join_key_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "defaults.yaml"
    cfg.write_text(
        """
contracts:
  datasets:
    sample:
      join_keys: ["season", "week"]
      schema:
        season: int64
        week: int64
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(contracts, "CONFIG_PATH", cfg)
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    frame = pl.DataFrame({"season": [2023], "week": [1]})
    contracts.assert_contract("sample", frame)
    bad = pl.DataFrame({"season": [2023]})
    with pytest.raises(ValueError):
        contracts.assert_contract("sample", bad)

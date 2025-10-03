"""Data contracts and schema stubs for staged datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "defaults.yaml"


@dataclass(frozen=True)
class DatasetContract:
    """Lightweight dataset contract capturing join keys and expected schema."""

    name: str
    join_keys: Sequence[str]
    schema: Mapping[str, str]

    def validate_join_keys(self) -> None:
        if not self.join_keys:
            raise ValueError(f"Contract '{self.name}' missing join keys")
        if len(set(self.join_keys)) != len(tuple(self.join_keys)):
            raise ValueError(f"Contract '{self.name}' has duplicate join keys")

    def lazy_frame(self) -> pl.LazyFrame:
        """Return an empty lazy frame using the declared schema."""

        dtype_mapping = {
            "int64": pl.Int64,
            "float64": pl.Float64,
            "str": pl.Utf8,
            "bool": pl.Boolean,
        }
        schema = {
            column: dtype_mapping.get(dtype, pl.Utf8)
            for column, dtype in self.schema.items()
        }
        return pl.LazyFrame(schema=schema)

    def enforce(self, frame: pl.DataFrame) -> None:
        missing = set(self.join_keys) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{self.name}: missing join key columns -> {sorted(missing)}"
            )


def load_contracts(config_path: Path | None = None) -> Dict[str, DatasetContract]:
    config_path = CONFIG_PATH if config_path is None else config_path
    with config_path.open("r", encoding="utf-8") as fh:
        defaults = yaml.safe_load(fh) or {}
    datasets = (
        defaults.get("contracts", {}).get("datasets", {})
        if isinstance(defaults, dict)
        else {}
    )
    contracts: Dict[str, DatasetContract] = {}
    for name, payload in datasets.items():
        join_keys = tuple(payload.get("join_keys", ()))
        schema = payload.get("schema", {}) or {}
        contracts[name] = DatasetContract(name=name, join_keys=join_keys, schema=schema)
        contracts[name].validate_join_keys()
    return contracts


def assert_contract(dataset: str, frame: pl.DataFrame, *, config_path: Path | None = None) -> None:
    contract_map = load_contracts(config_path=config_path)
    if dataset not in contract_map:
        raise KeyError(f"Unknown dataset '{dataset}'")
    contract_map[dataset].enforce(frame)


def assert_records(dataset: str, records: Iterable[Mapping[str, object]], *, config_path: Path | None = None) -> None:
    frame = pl.DataFrame(list(records)) if records else pl.DataFrame()
    assert_contract(dataset, frame, config_path=config_path)


def main() -> None:
    contracts = load_contracts()
    print(f"Contracts initialised -> {', '.join(sorted(contracts))}")


if __name__ == "__main__":
    main()

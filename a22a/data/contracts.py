"""Schema contracts and validation utilities for staged tables."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import polars as pl


@dataclass(frozen=True)
class FieldSpec:
    """Specification for a single column."""

    dtype: pl.DataType | type | str
    nullable: bool = True

    def matches(self, series: pl.Series) -> bool:
        if self.nullable is False and series.null_count() > 0:
            return False
        actual = series.dtype
        if isinstance(self.dtype, pl.DataType):
            expected = self.dtype
        elif isinstance(self.dtype, type) and issubclass(self.dtype, pl.DataType):
            expected = self.dtype()
        elif isinstance(self.dtype, str):
            expected = getattr(pl, self.dtype, None)
            if isinstance(expected, type) and issubclass(expected, pl.DataType):
                expected = expected()
        else:
            expected = None
        if isinstance(expected, pl.DataType) and isinstance(actual, pl.DataType):
            return str(actual) == str(expected)
        return True


@dataclass(frozen=True)
class TableContract:
    name: str
    columns: dict[str, FieldSpec]
    primary_key: tuple[str, ...]
    foreign_keys: dict[str, tuple[str, str]] = field(default_factory=dict)

    @property
    def required_cols(self) -> set[str]:
        return set(self.columns)


class ContractViolation(RuntimeError):
    pass


def _ensure_columns(table: pl.DataFrame, contract: TableContract) -> None:
    missing = contract.required_cols - set(table.columns)
    if missing:
        raise ContractViolation(f"{contract.name}: missing columns {sorted(missing)}")
    for name, spec in contract.columns.items():
        if name not in table.columns:
            continue
        if not spec.matches(table[name]):
            raise ContractViolation(
                f"{contract.name}.{name}: dtype/nullability mismatch (expected {spec.dtype}, nullable={spec.nullable})"
            )


def _ensure_primary_key(table: pl.DataFrame, contract: TableContract) -> None:
    if not contract.primary_key:
        return
    subset = list(contract.primary_key)
    if table.select(subset).null_count().sum_horizontal().item() > 0:
        raise ContractViolation(f"{contract.name}: primary key {subset} contains nulls")
    uniq = table.unique(subset=subset)
    if uniq.height != table.height:
        raise ContractViolation(
            f"{contract.name}: primary key {subset} not unique ({table.height - uniq.height} duplicate rows)"
        )


def _ensure_foreign_keys(
    table: pl.DataFrame, contract: TableContract, foreign_tables: dict[str, pl.DataFrame]
) -> None:
    for col, (foreign_table_name, foreign_col) in contract.foreign_keys.items():
        if col not in table.columns:
            raise ContractViolation(f"{contract.name}: foreign key column {col} missing")
        if foreign_table_name not in foreign_tables:
            raise ContractViolation(f"{contract.name}: foreign table {foreign_table_name} unavailable for FK check")
        fk_table = foreign_tables[foreign_table_name]
        if foreign_col not in fk_table.columns:
            raise ContractViolation(
                f"{contract.name}: foreign table {foreign_table_name} missing key column {foreign_col}"
            )
        missing = (
            table.join(
                fk_table.select(pl.col(foreign_col).alias("__fk")),
                left_on=col,
                right_on="__fk",
                how="anti",
            )
            .select(col)
            .unique()
        )
        if missing.height:
            raise ContractViolation(
                f"{contract.name}: {missing.height} values in {col} missing from {foreign_table_name}.{foreign_col}"
            )


def validate_contract(
    table: pl.DataFrame,
    contract: TableContract,
    *,
    foreign_tables: dict[str, pl.DataFrame] | None = None,
) -> None:
    """Validate ``table`` against ``contract`` raising ``ContractViolation`` if invalid."""

    foreign_tables = foreign_tables or {}
    _ensure_columns(table, contract)
    _ensure_primary_key(table, contract)
    _ensure_foreign_keys(table, contract, foreign_tables)


def validate_all(
    tables: dict[str, pl.DataFrame],
    contracts: Iterable[TableContract],
) -> None:
    lookup = {c.name: c for c in contracts}
    for name, table in tables.items():
        if name not in lookup:
            continue
        validate_contract(table, lookup[name], foreign_tables=tables)


PBP_CONTRACT = TableContract(
    name="pbp",
    columns={
        "game_id": FieldSpec(pl.Utf8, nullable=False),
        "play_id": FieldSpec(pl.Int64, nullable=False),
        "season": FieldSpec(pl.Int64, nullable=False),
        "week": FieldSpec(pl.Int64, nullable=False),
        "posteam": FieldSpec(pl.Utf8, nullable=False),
        "defteam": FieldSpec(pl.Utf8, nullable=False),
        "drive": FieldSpec(pl.Int64),
        "yards_gained": FieldSpec(pl.Int64),
        "epa": FieldSpec(pl.Float64),
        "play_type": FieldSpec(pl.Utf8),
        "pass": FieldSpec(pl.Int64),
        "rush": FieldSpec(pl.Int64),
        "down": FieldSpec(pl.Int64),
        "ydstogo": FieldSpec(pl.Int64),
        "game_seconds_remaining": FieldSpec(pl.Int64),
        "score_differential": FieldSpec(pl.Int64),
        "posteam_score": FieldSpec(pl.Int64),
        "defteam_score": FieldSpec(pl.Int64),
        "success": FieldSpec(pl.Int64),
    },
    primary_key=("game_id", "play_id"),
    foreign_keys={"game_id": ("games", "game_id")},
)

GAMES_CONTRACT = TableContract(
    name="games",
    columns={
        "game_id": FieldSpec(pl.Utf8, nullable=False),
        "season": FieldSpec(pl.Int64, nullable=False),
        "week": FieldSpec(pl.Int64, nullable=False),
        "kickoff_datetime": FieldSpec(pl.Datetime),
        "home_team": FieldSpec(pl.Utf8, nullable=False),
        "away_team": FieldSpec(pl.Utf8, nullable=False),
        "home_score": FieldSpec(pl.Int64, nullable=False),
        "away_score": FieldSpec(pl.Int64, nullable=False),
    },
    primary_key=("game_id",),
)

DRIVES_CONTRACT = TableContract(
    name="drives",
    columns={
        "game_id": FieldSpec(pl.Utf8, nullable=False),
        "drive_id": FieldSpec(pl.Utf8, nullable=False),
        "season": FieldSpec(pl.Int64, nullable=False),
        "week": FieldSpec(pl.Int64, nullable=False),
        "posteam": FieldSpec(pl.Utf8, nullable=False),
        "drive_result": FieldSpec(pl.Utf8, nullable=False),
        "drive_points": FieldSpec(pl.Int64, nullable=False),
        "drive_play_count": FieldSpec(pl.Int64),
        "drive_yards": FieldSpec(pl.Int64),
        "drive_time_seconds": FieldSpec(pl.Int64),
    },
    primary_key=("drive_id",),
    foreign_keys={"game_id": ("games", "game_id")},
)

ROSTER_CONTRACT = TableContract(
    name="roster",
    columns={
        "season": FieldSpec(pl.Int64, nullable=False),
        "team": FieldSpec(pl.Utf8, nullable=False),
        "player_id": FieldSpec(pl.Utf8, nullable=False),
        "player_name": FieldSpec(pl.Utf8, nullable=False),
        "position": FieldSpec(pl.Utf8, nullable=False),
        "status": FieldSpec(pl.Utf8, nullable=False),
    },
    primary_key=("season", "team", "player_id"),
)

CONTRACTS = (PBP_CONTRACT, GAMES_CONTRACT, DRIVES_CONTRACT, ROSTER_CONTRACT)

"""Data contracts for staged inputs.

The Phase 1–2 ingest stack produces a family of staged Parquet artefacts that
mirror a trimmed-down nflverse schema. This module provides reusable contracts
that guard schema drift both for the in-memory records used during tests as
well as the Polars frames that back the persisted parquet files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import polars as pl

__all__ = [
    "JOIN_KEYS",
    "STAGED_GAME_SCHEMA",
    "DatasetContract",
    "ContractViolation",
    "validate_records",
    "validate_frame",
]


@dataclass(frozen=True)
class Field:
    """Represents a required column in the staged schema."""

    name: str
    dtype: pl.DataType
    optional: bool = False

    def validate(self, frame: pl.DataFrame, dataset: str) -> None:
        if self.name not in frame.columns:
            raise ContractViolation(
                dataset=dataset,
                message=f"Missing required column '{self.name}'",
                record_index=-1,
            )
        series = frame.get_column(self.name)
        if series.null_count() > 0 and not self.optional:
            raise ContractViolation(
                dataset=dataset,
                message=f"Column '{self.name}' contains null values",
                record_index=-1,
            )
        if series.dtype != self.dtype:
            try:
                series.cast(self.dtype)
            except Exception as exc:  # noqa: BLE001
                raise ContractViolation(
                    dataset=dataset,
                    message=(
                        f"Column '{self.name}' expected dtype {self.dtype}, "
                        f"received {series.dtype}"
                    ),
                    record_index=-1,
                ) from exc


@dataclass(frozen=True)
class DatasetContract:
    """Contract definition for a staged dataset."""

    name: str
    fields: Sequence[Field]
    primary_key: Sequence[str]

    def validate_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return frame
        for field in self.fields:
            field.validate(frame, dataset=self.name)
        missing_keys = set(self.primary_key) - set(frame.columns)
        if missing_keys:
            raise ContractViolation(
                dataset=self.name,
                message=f"Missing primary key columns: {sorted(missing_keys)}",
                record_index=-1,
            )
        key_counts = frame.group_by(list(self.primary_key)).len()
        duplicated_keys = key_counts.filter(pl.col("len") > 1)
        if duplicated_keys.height > 0:
            duplicated_key = duplicated_keys.row(0)
            mask = pl.all_horizontal(
                [pl.col(column) == value for column, value in zip(self.primary_key, duplicated_key)]
            )
            index = (
                frame
                .with_row_count()
                .filter(mask)
                .select("row_nr")
                .to_series()
                .to_list()[0]
            )
            raise ContractViolation(
                dataset=self.name,
                message="Primary key uniqueness violated",
                record_index=int(index),
            )
        return frame


@dataclass
class ContractViolation(Exception):
    """Raised when staged data fails to meet the expected schema."""

    dataset: str
    message: str
    record_index: int

    def __str__(self) -> str:  # pragma: no cover - repr convenience
        return f"{self.dataset}: {self.message} (record={self.record_index})"


STAGED_GAME_SCHEMA = {
    "game_id": pl.Utf8,
    "season": pl.Int64,
    "week": pl.Int64,
    "team_id": pl.Utf8,
    "opponent_id": pl.Utf8,
}

JOIN_KEYS = {"game_id", "team_id"}


CONTRACTS: Mapping[str, DatasetContract] = {
    "pbp": DatasetContract(
        name="pbp",
        primary_key=("game_id", "play_id"),
        fields=(
            Field("game_id", pl.Utf8),
            Field("play_id", pl.Int64),
            Field("drive_id", pl.Utf8),
            Field("season", pl.Int64),
            Field("week", pl.Int64),
            Field("posteam", pl.Utf8),
            Field("defteam", pl.Utf8),
            Field("yards_gained", pl.Float64),
            Field("scoring_margin", pl.Float64),
            Field("success", pl.Boolean),
        ),
    ),
    "drives": DatasetContract(
        name="drives",
        primary_key=("game_id", "drive_id"),
        fields=(
            Field("game_id", pl.Utf8),
            Field("drive_id", pl.Utf8),
            Field("season", pl.Int64),
            Field("week", pl.Int64),
            Field("team_id", pl.Utf8),
            Field("result", pl.Utf8),
            Field("plays", pl.Int64),
            Field("yards", pl.Float64),
            Field("points", pl.Float64),
        ),
    ),
    "schedule": DatasetContract(
        name="schedule",
        primary_key=("game_id",),
        fields=(
            Field("game_id", pl.Utf8),
            Field("season", pl.Int64),
            Field("week", pl.Int64),
            Field("home_team", pl.Utf8),
            Field("away_team", pl.Utf8),
            Field("home_points", pl.Int64),
            Field("away_points", pl.Int64),
            Field("game_datetime", pl.Datetime("ns")),
            Field("venue", pl.Utf8),
        ),
    ),
    "roster": DatasetContract(
        name="roster",
        primary_key=("season", "team_id", "player_id"),
        fields=(
            Field("season", pl.Int64),
            Field("team_id", pl.Utf8),
            Field("player_id", pl.Utf8),
            Field("position", pl.Utf8),
            Field("full_name", pl.Utf8),
            Field("experience", pl.Int64, optional=True),
        ),
    ),
    "team_games": DatasetContract(
        name="team_games",
        primary_key=("game_id", "team_id"),
        fields=(
            Field("game_id", pl.Utf8),
            Field("season", pl.Int64),
            Field("week", pl.Int64),
            Field("team_id", pl.Utf8),
            Field("opponent_id", pl.Utf8),
            Field("is_home", pl.Boolean),
            Field("points_for", pl.Int64),
            Field("points_against", pl.Int64),
            Field("margin", pl.Int64),
            Field("total_points", pl.Int64),
            Field("win", pl.Int64),
            Field("game_datetime", pl.Datetime("ns")),
        ),
    ),
    "team_strength": DatasetContract(
        name="team_strength",
        primary_key=("season", "week", "team_id"),
        fields=(
            Field("season", pl.Int64),
            Field("week", pl.Int64),
            Field("team_id", pl.Utf8),
            Field("theta_mean", pl.Float64),
            Field("theta_ci_lower", pl.Float64),
            Field("theta_ci_upper", pl.Float64),
            Field("samples", pl.Float64),
        ),
    ),
}


def _contract(dataset: str) -> DatasetContract:
    try:
        return CONTRACTS[dataset]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise KeyError(f"Unknown dataset '{dataset}'") from exc


def _ensure_frame(records: Iterable[Mapping[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(list(records))


def validate_records(records: Iterable[Mapping[str, object]], dataset: str = "team_games") -> bool:
    """Validate staged records against a named contract.

    Parameters
    ----------
    records:
        Iterable of mapping-like records.
    dataset:
        Name of the dataset contract to evaluate.
    """

    records = list(records)
    if not records:
        return True
    frame = _ensure_frame(records)
    validate_frame(frame, dataset)
    return True


def validate_frame(frame: pl.DataFrame, dataset: str) -> pl.DataFrame:
    """Validate a Polars frame against a dataset contract."""

    contract = _contract(dataset)
    return contract.validate_frame(frame)


def main() -> None:
    print("Data contracts module provides schema validation for staged inputs.")


if __name__ == "__main__":
    main()

"""Data contracts for staged inputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

STAGED_GAME_SCHEMA = {
    "game_id": str,
    "season": int,
    "week": int,
    "team_id": str,
    "opponent_id": str,
}

JOIN_KEYS = {"game_id", "team_id"}


@dataclass
class ContractViolation(Exception):
    message: str
    record_index: int


def assert_schema(records: Sequence[Mapping[str, object]]) -> None:
    for idx, record in enumerate(records):
        missing = STAGED_GAME_SCHEMA.keys() - record.keys()
        if missing:
            raise ContractViolation(f"Missing fields: {sorted(missing)}", idx)
        for column, expected_type in STAGED_GAME_SCHEMA.items():
            value = record[column]
            if value is None:
                raise ContractViolation(f"Column '{column}' cannot be None", idx)
            if not isinstance(value, expected_type):
                raise ContractViolation(
                    f"Column '{column}' expected {expected_type.__name__}, got {type(value).__name__}",
                    idx,
                )
        if not JOIN_KEYS.issubset(record.keys()):
            raise ContractViolation("Missing join keys", idx)


def validate_records(records: Iterable[Mapping[str, object]]) -> bool:
    """Validate staged records against the contract schema."""
    records = list(records)
    if not records:
        return True
    assert_schema(records)
    return True


def main() -> None:
    print("Data contracts module provides schema validation for staged inputs.")


if __name__ == "__main__":
    main()

"""Preprocessing helpers for simulation-generated tabular data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def validate_required_columns(rows: Iterable[Mapping[str, Any]], required_columns: set[str]) -> None:
    """Validate that every row contains the required columns."""

    for index, row in enumerate(rows):
        missing = required_columns.difference(row)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Row {index} is missing required columns: {missing_list}")


def drop_rows_with_missing_values(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return rows that have no empty-string or None values."""

    cleaned: list[dict[str, Any]] = []
    for row in rows:
        if all(value is not None and value != "" for value in row.values()):
            cleaned.append(dict(row))
    return cleaned


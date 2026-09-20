"""CSV loading helpers for NS-3 generated datasets."""

from __future__ import annotations

import csv
from pathlib import Path


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Load a CSV file as a list of dictionaries."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries to CSV, creating parent directories as needed."""

    if not rows:
        raise ValueError("rows cannot be empty")

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


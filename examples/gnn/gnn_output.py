"""Output helpers for the GNN example."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict


class PredictionRow(TypedDict):
    """Serialized prediction output."""

    node_id: str
    actual_flow_loss: float
    predicted_flow_loss: float


def write_predictions(rows: list[PredictionRow], csv_path: Path) -> None:
    """Write predictions to CSV."""
    fieldnames = ["node_id", "actual_flow_loss", "predicted_flow_loss"]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

"""CSV writer and row schema for per-node prediction output."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict


class PredictionRow(TypedDict):
    """One row in the predictions CSV.

    Attributes:
        node_id: IP address (graph node identifier).
        actual_importance: Ground-truth importance score (raw scale, not log1p).
        predicted_importance: Model-predicted importance score (raw scale).
        failure_probability: Normalised MC-Dropout variance in ``[0, 1]``;
            higher means the model is less certain about this node.
        composite_risk: ``actual_importance * failure_probability`` —
            ranking metric used to surface the most critical nodes.
    """

    node_id: str
    actual_importance: float
    predicted_importance: float
    failure_probability: float
    composite_risk: float


def write_predictions(rows: list[PredictionRow], csv_path: Path) -> None:
    """Write prediction rows to ``csv_path`` as UTF-8 CSV with a header line.

    Args:
        rows: List of :class:`PredictionRow` dicts (one per test node).
        csv_path: Destination CSV file. Any existing file at this path is overwritten.
    """
    fieldnames = [
        "node_id",
        "actual_importance",
        "predicted_importance",
        "failure_probability",
        "composite_risk",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

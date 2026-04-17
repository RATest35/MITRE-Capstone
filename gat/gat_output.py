 
from __future__ import annotations
 
import csv
from pathlib import Path
from typing import TypedDict
 
 
class PredictionRow(TypedDict):
 
    node_id: str
    actual_importance: float
    predicted_importance: float
    failure_probability: float
    composite_risk: float
 
 
def write_predictions(rows: list[PredictionRow], csv_path: Path) -> None:
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
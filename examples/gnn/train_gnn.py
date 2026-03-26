"""Entry point for GNN training."""

from __future__ import annotations

from gnn_config import GRAPHML_PATH, PREDICTION_CSV_PATH
from gnn_data import build_data
from gnn_output import PredictionRow, write_predictions
from gnn_training import MetricRow, train_and_evaluate


def main() -> None:
    """Run training."""
    data, node_ids = build_data(GRAPHML_PATH)
    metrics: MetricRow
    prediction_rows: list[PredictionRow]
    metrics, prediction_rows = train_and_evaluate(data, node_ids)
    write_predictions(prediction_rows, PREDICTION_CSV_PATH)

    print(
        "Metrics: "
        f"MAE={metrics['mae']:.2f}, "
        f"RMSE={metrics['rmse']:.2f}, "
        f"log_MAE={metrics['log_mae']:.4f}, "
        f"log_RMSE={metrics['log_rmse']:.4f}, "
        f"best_val_loss={metrics['best_val_loss']:.4f}"
    )
    print(f"Composite score predictions written to: {PREDICTION_CSV_PATH}")


if __name__ == "__main__":
    main()

"""Training and evaluation for the GNN example."""

from __future__ import annotations

from copy import deepcopy
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from gnn_config import DROPOUT, HIDDEN_CHANNELS, LEARNING_RATE, NUM_EPOCHS, PATIENCE, WEIGHT_DECAY
from gnn_data import build_masks, standardize_features, standardize_targets
from gnn_model import CompositeScoreGNN
from gnn_output import PredictionRow


class MetricRow(TypedDict):
    """Aggregate metrics for one training run."""

    mae: float
    rmse: float
    log_mae: float
    log_rmse: float
    best_val_loss: float


def train_and_evaluate(base_data: Data, node_ids: list[str]) -> tuple[MetricRow, list[PredictionRow]]:
    """Train the model and evaluate on the test split."""
    data = deepcopy(base_data)
    train_mask, val_mask, test_mask = build_masks(data.num_nodes)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.x = standardize_features(data.x, train_mask)
    data.y, target_mean, target_std = standardize_targets(data.y, train_mask)

    model = CompositeScoreGNN(
        input_channels=data.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_state = _train_model(model, data, optimizer, train_mask, val_mask)

    model.load_state_dict(best_state["state_dict"])
    model.eval()

    with torch.no_grad():
        predictions = model(data)
        predictions = predictions * target_std + target_mean

    metrics = _build_metrics(
        predictions,
        data.composite_score,
        test_mask,
        best_state["best_val_loss"],
    )
    prediction_rows = _build_prediction_rows(
        predictions,
        data.composite_score,
        test_mask,
        node_ids,
    )
    return metrics, prediction_rows


def _train_model(
    model: CompositeScoreGNN,
    data: Data,
    optimizer: torch.optim.Optimizer,
    train_mask: Tensor,
    val_mask: Tensor,
) -> dict[str, dict[str, Tensor] | float]:
    """Train the model with early stopping."""
    best_state: dict[str, Tensor] | None = None
    best_val_loss = float("inf")
    stale_epochs = 0

    for _ in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        predictions = model(data)
        train_loss = F.mse_loss(predictions[train_mask], data.y[train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(data)
            val_loss = F.mse_loss(val_predictions[val_mask], data.y[val_mask]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
            continue

        stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    return {"state_dict": best_state, "best_val_loss": best_val_loss}


def _build_metrics(
    predictions: Tensor,
    composite_scores: Tensor,
    test_mask: Tensor,
    best_val_loss: float,
) -> MetricRow:
    """Build aggregate metrics."""
    actual = composite_scores[test_mask]
    predicted = torch.expm1(predictions[test_mask]).clamp_min(0.0)
    actual_log = torch.log1p(actual)
    predicted_log = torch.log1p(predicted)

    return {
        "mae": torch.mean(torch.abs(predicted - actual)).item(),
        "rmse": torch.sqrt(torch.mean((predicted - actual) ** 2)).item(),
        "log_mae": torch.mean(torch.abs(predicted_log - actual_log)).item(),
        "log_rmse": torch.sqrt(torch.mean((predicted_log - actual_log) ** 2)).item(),
        "best_val_loss": best_val_loss,
    }


def _build_prediction_rows(
    predictions: Tensor,
    composite_scores: Tensor,
    test_mask: Tensor,
    node_ids: list[str],
) -> list[PredictionRow]:
    """Build per-node prediction rows."""
    rows: list[PredictionRow] = []
    test_indices = test_mask.nonzero(as_tuple=False).view(-1).tolist()

    for node_position in test_indices:
        rows.append(
            {
                "node_id": node_ids[node_position],
                "actual_composite_score": float(composite_scores[node_position].item()),
                "predicted_composite_score": float(torch.expm1(predictions[node_position]).clamp_min(0.0).item()),
            }
        )

    return rows

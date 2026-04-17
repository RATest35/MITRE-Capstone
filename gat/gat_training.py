"""Training loop and evaluation for the GATv2 pipeline with K-fold CV."""
 
from __future__ import annotations
 
from copy import deepcopy
from typing import TypedDict
 
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
 
from gat_config import (
    DEVICE,
    DROP_EDGE_PROB,
    DROPOUT,
    EDGE_DIM,
    HIDDEN_CHANNELS,
    K_FOLDS,
    LEARNING_RATE,
    MC_DROPOUT_SAMPLES,
    NUM_EPOCHS,
    NUM_HEADS,
    PATIENCE,
    RANKING_LOSS_WEIGHT,
    RANKING_MARGIN,
    RANKING_PAIRS,
    SELECTION_METRIC,
    WEIGHT_DECAY,
)
from gat_data import build_kfold_masks, standardize_features
from gat_metrics import regression_metrics
from gat_model import GATv2Regressor
from gat_output import PredictionRow
 
# Metrics where higher values are better (used for early stopping direction).
_HIGHER_IS_BETTER: set[str] = {
    "pearson", "spearman",
    "top_1pct_recall", "top_5pct_recall",
    "ndcg_1pct", "ndcg_5pct",
}
 
 
def train_and_evaluate(
    base_data: Data, node_ids: list[str],
) -> tuple[list[dict[str, float]], list[PredictionRow]]:
    """Train and evaluate a GATv2Regressor using K-fold cross-validation.
 
    For each fold the function:
      1. Applies fold-specific train/val/test masks
      2. Standardises node features on the training split
      3. Trains with early stopping (Huber loss + sample weights + optional ranking loss)
      4. Runs MC Dropout inference for failure probability
      5. Collects per-fold metrics and test-node predictions
 
    """
    fold_masks = build_kfold_masks(base_data.num_nodes)
    all_metrics: list[dict[str, float]] = []
    all_predictions: list[PredictionRow] = []
 
    for fold_idx, (train_mask, val_mask, test_mask) in enumerate(fold_masks):
        print(f"\n-- Fold {fold_idx + 1}/{K_FOLDS} ---------------------------------")
 
        data = deepcopy(base_data)
        data.x = standardize_features(data.x, train_mask)
 
        model = GATv2Regressor(
            input_channels=data.num_node_features,
            hidden_channels=HIDDEN_CHANNELS,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            drop_edge_prob=DROP_EDGE_PROB,
            edge_dim=EDGE_DIM,
        ).to(DEVICE)
 
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
        best_state = _train_model(model, data, optimizer, train_mask, val_mask)
 
        # --- MC Dropout inference -----------------------------------------------
        model.load_state_dict(best_state["state_dict"])
        model.train()  # keep dropout active
        saved_drop_edge_prob = model.drop_edge_prob
        model.drop_edge_prob = 0.0  # disable DropEdge during MC passes
 
        mc_preds: list[Tensor] = []
        with torch.no_grad():
            for _ in range(MC_DROPOUT_SAMPLES):
                mc_preds.append(model(data))
 
        model.drop_edge_prob = saved_drop_edge_prob
 
        stacked = torch.stack(mc_preds, dim=0)       # [MC, N]
        mean_predictions = stacked.mean(dim=0)        # [N]
        variance = stacked.var(dim=0)                 # [N]
 
        # Normalise variance → failure probability [0, 1]
        var_range = variance.max() - variance.min()
        if var_range.item() < 1e-9:
            failure_probability = torch.zeros_like(variance)
        else:
            failure_probability = (variance - variance.min()) / var_range
 
        # --- Metrics (full suite) -----------------------------------------------
        actual_log = data.y[test_mask].cpu().numpy()
        predicted_log = mean_predictions[test_mask].cpu().numpy()
        fold_metrics = regression_metrics(actual_log, predicted_log)
        fold_metrics["fold"] = float(fold_idx + 1)
        fold_metrics["best_val_loss"] = best_state["best_val_loss"]
        all_metrics.append(fold_metrics)
 
        fold_rows = _build_prediction_rows(
            mean_predictions, data.y, failure_probability, test_mask, node_ids,
        )
        all_predictions.extend(fold_rows)
 
        print(f"  RMSE       : {fold_metrics['rmse']:.4f}")
        print(f"  Log-MAE    : {fold_metrics['log_mae']:.4f}")
        print(f"  Spearman   : {fold_metrics['spearman']:.4f}")
        print(f"  NDCG@5%    : {fold_metrics['ndcg_5pct']:.4f}")
        print(f"  Top-5% Rec : {fold_metrics['top_5pct_recall']:.4f}")
 
    return all_metrics, all_predictions
 
 

# Private helpers
# ---------------------------------------------------------------------------
 
def _train_model(
    model: GATv2Regressor,
    data: Data,
    optimizer: torch.optim.Optimizer,
    train_mask: Tensor,
    val_mask: Tensor,
) -> dict:
    """Run the training loop with early stopping.
 
    Uses Huber loss weighted by sample weights.  Optionally adds a pairwise
    ranking loss (controlled by RANKING_LOSS_WEIGHT).
 
    Early stopping is based on SELECTION_METRIC computed on the val split.
    """
    higher_is_better = SELECTION_METRIC in _HIGHER_IS_BETTER
    best_state_dict: dict | None = None
    best_val_score = float("-inf") if higher_is_better else float("inf")
    stale_epochs = 0
 
    sample_weights = data.sample_weights
 
    for epoch in range(NUM_EPOCHS):
        # --- Training step ---
        model.train()
        optimizer.zero_grad()
        preds = model(data)
 
        # Weighted Huber loss on training nodes
        per_sample = F.huber_loss(
            preds[train_mask], data.y[train_mask], reduction="none",
        )
        weights = sample_weights[train_mask]
        regression_loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)
 
        # Optional pairwise ranking loss
        if RANKING_LOSS_WEIGHT > 0.0:
            ranking_loss = _pairwise_ranking_loss(
                preds[train_mask], data.y[train_mask], weights,
                margin=RANKING_MARGIN, max_pairs=RANKING_PAIRS,
            )
            loss = regression_loss + RANKING_LOSS_WEIGHT * ranking_loss
        else:
            loss = regression_loss
 
        loss.backward()
        optimizer.step()
 
        # --- Validation step ---
        model.eval()
        with torch.no_grad():
            val_preds = model(data)
 
        actual_log = data.y[val_mask].cpu().numpy()
        predicted_log = val_preds[val_mask].cpu().numpy()
        val_metrics = regression_metrics(actual_log, predicted_log)
        val_score = val_metrics[SELECTION_METRIC]
 
        improved = (val_score > best_val_score) if higher_is_better else (val_score < best_val_score)
        if improved:
            best_val_score = val_score
            best_state_dict = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
 
    if best_state_dict is None:
        raise RuntimeError("Training did not produce a valid model state.")
 
    return {"state_dict": best_state_dict, "best_val_loss": best_val_score}
 
 
def _pairwise_ranking_loss(
    predictions: Tensor,
    targets: Tensor,
    weights: Tensor,
    margin: float,
    max_pairs: int,
) -> Tensor:
    """Pairwise ranking hinge loss (matches composite pipeline).
 
    Encourages the model to rank higher-importance nodes above
    lower-importance nodes by at least *margin*.
    """
    if predictions.numel() < 2 or max_pairs <= 0:
        return predictions.new_zeros(())
 
    order = torch.argsort(targets)
    pair_count = min(max_pairs, predictions.numel() // 2)
    if pair_count == 0:
        return predictions.new_zeros(())
 
    low_indices = order[:pair_count]
    high_indices = order[-pair_count:]
    valid_mask = targets[high_indices] > targets[low_indices]
    if not torch.any(valid_mask):
        return predictions.new_zeros(())
 
    gaps = predictions[high_indices][valid_mask] - predictions[low_indices][valid_mask]
    pair_weights = 0.5 * (weights[high_indices][valid_mask] + weights[low_indices][valid_mask])
    pair_losses = torch.relu(margin - gaps)
    return (pair_losses * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)
 
 
def _build_prediction_rows(
    predictions: Tensor,
    labels: Tensor,
    failure_probability: Tensor,
    test_mask: Tensor,
    node_ids: list[str],
) -> list[PredictionRow]:
    """Build one ``PredictionRow`` per test node."""
    test_indices = test_mask.nonzero(as_tuple=False).view(-1).tolist()
    rows: list[PredictionRow] = []
 
    for i in test_indices:
        actual_imp = float(torch.expm1(labels[i]).item())
        pred_imp = float(torch.expm1(predictions[i].clamp(max=88.0)).clamp_min(0.0).item())
        fail_prob = float(failure_probability[i].item())
 
        rows.append({
            "node_id": node_ids[i],
            "actual_importance": actual_imp,
            "predicted_importance": pred_imp,
            "failure_probability": fail_prob,
            "composite_risk": pred_imp * fail_prob,
        })
 
    return rows
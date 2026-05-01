"""K-fold training, MC-Dropout inference, and evaluation for the GATv2 pipeline.

Public entry point is :func:`train_and_evaluate`. The training loss is a
sample-weighted Huber loss, optionally combined with a pairwise ranking hinge
loss (controlled by ``RANKING_LOSS_WEIGHT``). Best model per fold is selected
by ``SELECTION_METRIC`` on the validation split, then MC-Dropout is used to
estimate per-node failure probability (uncertainty proxy).
"""
 
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
    """Run K-fold cross-validation training and inference.

    For each fold:
        1. Apply the fold's train/val/test masks.
        2. Z-score-standardise node features using train-split statistics.
        3. Train a fresh :class:`GATv2Regressor` with early stopping on
           ``SELECTION_METRIC``.
        4. Run :data:`MC_DROPOUT_SAMPLES` stochastic forward passes to compute
           a mean prediction and a per-node failure probability (normalised
           variance in ``[0, 1]``).
        5. Compute the full metric suite on the test split and build per-node
           prediction rows.

    Args:
        base_data: PyG ``Data`` produced by :func:`gat_data.build_data`.
        node_ids: Ordered list of node IDs aligned with ``base_data.x``.

    Returns:
        Tuple ``(all_metrics, all_predictions)``:

        - ``all_metrics`` — list of per-fold metric dicts (length ``K_FOLDS``)
          with keys from :func:`regression_metrics` plus ``fold``,
          ``train_loss``, ``val_loss``.
        - ``all_predictions`` — flat list of :class:`PredictionRow` covering
          every test node across all folds.
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
        fold_metrics["train_loss"] = best_state["train_loss"]
        fold_metrics["val_loss"] = best_state["val_loss"]
        all_metrics.append(fold_metrics)
 
        fold_rows = _build_prediction_rows(
            mean_predictions, data.y, failure_probability, test_mask, node_ids,
        )
        all_predictions.extend(fold_rows)
 
        print(f"  Train loss : {fold_metrics['train_loss']:.6f}")
        print(f"  Val loss   : {fold_metrics['val_loss']:.6f}")
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

    Loss is sample-weighted Huber on the training split; if
    ``RANKING_LOSS_WEIGHT > 0`` a pairwise ranking hinge term is added.
    The best model state (snapshotted at the epoch with the best validation
    ``SELECTION_METRIC``) is restored before returning.

    Args:
        model: Untrained :class:`GATv2Regressor` instance on ``DEVICE``.
        data: PyG ``Data`` with ``x``, ``edge_index``, ``edge_attr``, ``y``,
            and ``sample_weights``.
        optimizer: Pre-built optimiser bound to ``model.parameters()``.
        train_mask: Boolean mask selecting training nodes.
        val_mask: Boolean mask selecting validation nodes.

    Returns:
        Dict with keys ``state_dict`` (best model weights),
        ``best_val_score`` (validation metric value at that epoch),
        ``train_loss`` and ``val_loss`` (loss values at the best epoch).

    Raises:
        RuntimeError: If no epoch ever improved over the initial best score
            (only happens for pathological data or zero ``NUM_EPOCHS``).
    """
    higher_is_better = SELECTION_METRIC in _HIGHER_IS_BETTER
    best_state_dict: dict | None = None
    best_val_score = float("-inf") if higher_is_better else float("inf")
    best_train_loss = float("nan")
    best_val_loss = float("nan")
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
        train_loss_value = float(loss.detach().item())

        # --- Validation step ---
        model.eval()
        with torch.no_grad():
            val_preds = model(data)
            val_loss_value = _compute_loss(val_preds, data, val_mask, sample_weights)

        actual_log = data.y[val_mask].cpu().numpy()
        predicted_log = val_preds[val_mask].cpu().numpy()
        val_metrics = regression_metrics(actual_log, predicted_log)
        val_score = val_metrics[SELECTION_METRIC]

        improved = (val_score > best_val_score) if higher_is_better else (val_score < best_val_score)
        if improved:
            best_val_score = val_score
            best_train_loss = train_loss_value
            best_val_loss = val_loss_value
            best_state_dict = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a valid model state.")

    return {
        "state_dict": best_state_dict,
        "best_val_score": best_val_score,
        "train_loss": best_train_loss,
        "val_loss": best_val_loss,
    }


def _compute_loss(preds: Tensor, data: Data, mask: Tensor, sample_weights: Tensor) -> float:
    """Compute the same training loss formulation on an arbitrary node split.

    Used to report a comparable validation loss value alongside the metric.

    Args:
        preds: Per-node predictions for the entire graph.
        data: PyG ``Data`` (only ``data.y`` is read).
        mask: Boolean mask selecting which nodes to score.
        sample_weights: Per-node weights (matches the training-time weights).

    Returns:
        Scalar loss value (Python float).
    """
    per_sample = F.huber_loss(preds[mask], data.y[mask], reduction="none")
    weights = sample_weights[mask]
    regression_loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)

    if RANKING_LOSS_WEIGHT > 0.0:
        ranking_loss = _pairwise_ranking_loss(
            preds[mask], data.y[mask], weights,
            margin=RANKING_MARGIN, max_pairs=RANKING_PAIRS,
        )
        return float((regression_loss + RANKING_LOSS_WEIGHT * ranking_loss).item())
    return float(regression_loss.item())
 
 
def _pairwise_ranking_loss(
    predictions: Tensor,
    targets: Tensor,
    weights: Tensor,
    margin: float,
    max_pairs: int,
) -> Tensor:
    """Sample-weighted pairwise ranking hinge loss.

    Encourages the model to rank the highest-target node above the lowest-target
    node by at least ``margin`` in prediction space, for each of the top
    ``max_pairs`` (high, low) pairs sorted by target.

    Args:
        predictions: Per-node predictions for the masked split.
        targets: Per-node targets for the same masked split.
        weights: Per-node sample weights for the same masked split.
        margin: Minimum prediction gap required between high and low pairs.
        max_pairs: Maximum number of (high, low) pairs to score.

    Returns:
        Scalar tensor (loss value). Returns zero if there are too few elements
        or if no valid (high > low) pairs exist.
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
    """Materialise per-test-node :class:`PredictionRow` dicts.

    Predictions are clamped to ``log_pred <= 88`` before ``expm1`` and then to
    ``>= 0`` to keep the raw-scale importance non-negative.

    Args:
        predictions: Per-node predictions in ``log1p`` space (full graph).
        labels: Per-node ground-truth labels in ``log1p`` space (full graph).
        failure_probability: Normalised MC-Dropout variance in ``[0, 1]``.
        test_mask: Boolean mask selecting test nodes.
        node_ids: Ordered list of node IDs (aligned with ``predictions``).

    Returns:
        List of :class:`PredictionRow` dicts (one per test node).
    """
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
            "composite_risk": actual_imp * fail_prob,
        })
 
    return rows
"""Evaluation metrics for the GATv2 pipeline.
 
"""
 
from __future__ import annotations
 
from typing import Any
 
import numpy as np
from scipy.stats import pearsonr, spearmanr
 
 
def _safe_statistic(function: Any, actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return a finite correlation value, 0.0 on degenerate input."""
    if len(actual) < 2:
        return 0.0
    value = function(actual, predicted).statistic
    if np.isnan(value):
        return 0.0
    return float(value)
 
 
def regression_metrics(actual_log: np.ndarray, predicted_log: np.ndarray) -> dict[str, float]:
    """Compute regression and ranking metrics on log and raw scales.
 
    Parameters
    ----------
    actual_log:    Ground-truth values in log-space (log1p).
    predicted_log: Predicted values in log-space (log1p).
 
    Returns
    -------
    Dict with keys: mae, rmse, log_mae, log_rmse, pearson, spearman,
    top_1pct_recall, top_5pct_recall, ndcg_1pct, ndcg_5pct.
    """
    actual = np.expm1(actual_log)
    predicted = np.expm1(np.clip(predicted_log, None, 88.0))
    predicted = np.clip(predicted, 0.0, None)
 
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    log_mae = float(np.mean(np.abs(actual_log - predicted_log)))
    log_rmse = float(np.sqrt(np.mean((actual_log - predicted_log) ** 2)))
    pearson = _safe_statistic(pearsonr, actual, predicted)
    spearman = _safe_statistic(spearmanr, actual, predicted)
 
    return {
        "mae": mae,
        "rmse": rmse,
        "log_mae": log_mae,
        "log_rmse": log_rmse,
        "pearson": pearson,
        "spearman": spearman,
        "top_1pct_recall": top_k_recall(actual, predicted, 0.01),
        "top_5pct_recall": top_k_recall(actual, predicted, 0.05),
        "ndcg_1pct": ndcg_at_ratio(actual, predicted, 0.01),
        "ndcg_5pct": ndcg_at_ratio(actual, predicted, 0.05),
    }
 
 
def top_k_recall(actual: np.ndarray, predicted: np.ndarray, ratio: float) -> float:
    """Fraction of the true top-k nodes that appear in the predicted top-k.
 
    A recall of 1.0 means the model perfectly identifies which nodes are the
    most critical; 0.0 means it misses all of them.
    """
    count = max(1, int(len(actual) * ratio))
    actual_top = set(np.argsort(actual)[-count:].tolist())
    predicted_top = set(np.argsort(predicted)[-count:].tolist())
    return float(len(actual_top & predicted_top) / count)
 
 
def ndcg_at_ratio(actual: np.ndarray, predicted: np.ndarray, ratio: float) -> float:
    """Normalised Discounted Cumulative Gain at a given ratio.
 
    Uses raw target values as relevance scores.  NDCG of 1.0 means the
    predicted ranking of the top-k nodes perfectly matches the ideal ranking.
    """
    count = max(1, int(len(actual) * ratio))
    predicted_order = np.argsort(predicted)[::-1][:count]
    ideal_order = np.argsort(actual)[::-1][:count]
    predicted_gain = actual[predicted_order]
    ideal_gain = actual[ideal_order]
    discounts = 1.0 / np.log2(np.arange(2, count + 2))
    dcg = float(np.sum(predicted_gain * discounts))
    idcg = float(np.sum(ideal_gain * discounts))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
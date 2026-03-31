from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _safe_statistic(function: Any, actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return a finite correlation value."""
    if len(actual) < 2:
        return 0.0

    value = function(actual, predicted).statistic
    if np.isnan(value):
        return 0.0
    return float(value)


def regression_metrics(actual_log: np.ndarray, predicted_log: np.ndarray) -> dict[str, float]:
    """Compute regression metrics on log and raw scales."""
    actual = np.expm1(actual_log)
    predicted = np.expm1(predicted_log)

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
    """Compute recall of true top-k nodes."""
    count = max(1, int(len(actual) * ratio))
    actual_top = set(np.argsort(actual)[-count:].tolist())
    predicted_top = set(np.argsort(predicted)[-count:].tolist())
    return float(len(actual_top & predicted_top) / count)


def ndcg_at_ratio(actual: np.ndarray, predicted: np.ndarray, ratio: float) -> float:
    """Compute NDCG at one ratio using raw target values as relevance."""
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

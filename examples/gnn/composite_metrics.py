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
    }

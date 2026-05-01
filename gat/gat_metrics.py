"""Evaluation metrics for the GATv2 pipeline.

Provides the metric suite used by both the training loop and the Optuna
hyperparameter search:

- Regression: ``mae``, ``rmse``, ``log_mae``, ``log_rmse``
- Correlation: ``pearson``, ``spearman``
- Ranking: ``top_1pct_recall``, ``top_5pct_recall``, ``ndcg_1pct``, ``ndcg_5pct``

All metrics expect inputs in **log-space** (``log1p``-transformed values);
:func:`regression_metrics` applies ``expm1`` internally for the raw-scale metrics.
"""
 
from __future__ import annotations
 
from collections.abc import Callable
from typing import Any
 
import numpy as np
from scipy.stats import pearsonr, spearmanr
 
 
def _is_constant(values: np.ndarray) -> bool:
    """Return whether all values are identical."""
    return bool(np.all(values == values[0]))


def _safe_statistic(
    function: Callable[[np.ndarray, np.ndarray], Any],
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Return a finite correlation value, 0.0 on degenerate input."""
    if len(actual) < 2 or _is_constant(actual) or _is_constant(predicted):
        return 0.0
    value = function(actual, predicted).statistic
    if np.isnan(value):
        return 0.0
    return float(value)
 
 
def regression_metrics(actual_log: np.ndarray, predicted_log: np.ndarray) -> dict[str, float]:
    """Compute the full regression + ranking metric suite.

    Predicted values are clipped to ``log_pred <= 88`` before ``expm1`` to
    avoid floating-point overflow on extreme outputs.

    Args:
        actual_log: Ground-truth importance scores in ``log1p`` space.
        predicted_log: Model-predicted importance scores in ``log1p`` space.

    Returns:
        Dict mapping metric name → float value, with keys
        ``mae``, ``rmse``, ``log_mae``, ``log_rmse``, ``pearson``,
        ``spearman``, ``top_1pct_recall``, ``top_5pct_recall``,
        ``ndcg_1pct``, ``ndcg_5pct``.
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

    Args:
        actual: Ground-truth scores (any monotonic-with-importance scale).
        predicted: Predicted scores in the same ordering convention as ``actual``.
        ratio: Top-k as a fraction of the population (e.g. ``0.05`` for top 5%).

    Returns:
        Recall in ``[0, 1]``. ``1.0`` means perfect identification of critical
        nodes; ``0.0`` means none of the predicted top-k are truly critical.
    """
    count = max(1, int(len(actual) * ratio))
    actual_top = set(np.argsort(actual)[-count:].tolist())
    predicted_top = set(np.argsort(predicted)[-count:].tolist())
    return float(len(actual_top & predicted_top) / count)
 
 
def ndcg_at_ratio(actual: np.ndarray, predicted: np.ndarray, ratio: float) -> float:
    """Normalised Discounted Cumulative Gain at a given top-k ratio.

    Uses raw target values as relevance scores. Unlike :func:`top_k_recall`
    this also rewards correctly ordering the top-k nodes (not just selecting them).

    Args:
        actual: Ground-truth scores used as relevance values.
        predicted: Predicted scores used to define the candidate ranking.
        ratio: Top-k as a fraction of the population.

    Returns:
        NDCG in ``[0, 1]``. ``1.0`` means the predicted top-k ranking
        perfectly matches the ideal top-k ranking.
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

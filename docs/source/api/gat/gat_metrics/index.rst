gat.gat_metrics
===============

.. py:module:: gat.gat_metrics

.. autoapi-nested-parse::

   Evaluation metrics for the GATv2 pipeline.

   Provides the metric suite used by both the training loop and the Optuna
   hyperparameter search:

   - Regression: ``mae``, ``rmse``, ``log_mae``, ``log_rmse``
   - Correlation: ``pearson``, ``spearman``
   - Ranking: ``top_1pct_recall``, ``top_5pct_recall``, ``ndcg_1pct``, ``ndcg_5pct``

   All metrics expect inputs in **log-space** (``log1p``-transformed values);
   :func:`regression_metrics` applies ``expm1`` internally for the raw-scale metrics.



Functions
---------

.. autoapisummary::

   gat.gat_metrics.regression_metrics
   gat.gat_metrics.top_k_recall
   gat.gat_metrics.ndcg_at_ratio


Module Contents
---------------

.. py:function:: regression_metrics(actual_log: numpy.ndarray, predicted_log: numpy.ndarray) -> dict[str, float]

   Compute the full regression + ranking metric suite.

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


.. py:function:: top_k_recall(actual: numpy.ndarray, predicted: numpy.ndarray, ratio: float) -> float

   Fraction of the true top-k nodes that appear in the predicted top-k.

   Args:
       actual: Ground-truth scores (any monotonic-with-importance scale).
       predicted: Predicted scores in the same ordering convention as ``actual``.
       ratio: Top-k as a fraction of the population (e.g. ``0.05`` for top 5%).

   Returns:
       Recall in ``[0, 1]``. ``1.0`` means perfect identification of critical
       nodes; ``0.0`` means none of the predicted top-k are truly critical.


.. py:function:: ndcg_at_ratio(actual: numpy.ndarray, predicted: numpy.ndarray, ratio: float) -> float

   Normalised Discounted Cumulative Gain at a given top-k ratio.

   Uses raw target values as relevance scores. Unlike :func:`top_k_recall`
   this also rewards correctly ordering the top-k nodes (not just selecting them).

   Args:
       actual: Ground-truth scores used as relevance values.
       predicted: Predicted scores used to define the candidate ranking.
       ratio: Top-k as a fraction of the population.

   Returns:
       NDCG in ``[0, 1]``. ``1.0`` means the predicted top-k ranking
       perfectly matches the ideal top-k ranking.



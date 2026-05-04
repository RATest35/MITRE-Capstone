gat.gat_training
================

.. py:module:: gat.gat_training

.. autoapi-nested-parse::

   K-fold training, MC-Dropout inference, and evaluation for the GATv2 pipeline.

   Public entry point is :func:`train_and_evaluate`. The training loss is a
   sample-weighted Huber loss, optionally combined with a pairwise ranking hinge
   loss (controlled by ``RANKING_LOSS_WEIGHT``). Best model per fold is selected
   by ``SELECTION_METRIC`` on the validation split, then MC-Dropout is used to
   estimate per-node failure probability (uncertainty proxy).



Functions
---------

.. autoapisummary::

   gat.gat_training.train_and_evaluate


Module Contents
---------------

.. py:function:: train_and_evaluate(base_data: torch_geometric.data.Data, node_ids: list[str]) -> tuple[list[dict[str, float]], list[gat_output.PredictionRow]]

   Run K-fold cross-validation training and inference.

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



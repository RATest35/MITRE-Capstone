gat.train
=========

.. py:module:: gat.train

.. autoapi-nested-parse::

   Train a GATv2Regressor on the graph data and evaluate across K folds.
    
   Run from this directory:
       python train.py
    
   The script loads the GraphML graph, trains a GATv2Regressor across K folds
   with DropEdge and early stopping, runs MC Dropout inference per fold, prints
   cross-validated metrics and writes per-node predictions to CSV.



Functions
---------

.. autoapisummary::

   gat.train.main


Module Contents
---------------

.. py:function:: main() -> None

   Run the full training pipeline end-to-end.

   Steps:
       1. Load the GraphML graph via :func:`gat_data.build_data`.
       2. Run K-fold training + MC-Dropout inference via
          :func:`gat_training.train_and_evaluate`.
       3. Print a per-fold metric table and aggregated mean ± std summary.
       4. Print the top-5 test nodes ranked by composite risk.
       5. Write all per-node predictions to :data:`PREDICTION_CSV_PATH`.



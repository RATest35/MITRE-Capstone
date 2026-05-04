gat.tune
========

.. py:module:: gat.tune

.. autoapi-nested-parse::

   Hyperparameter search for the GATv2 pipeline using Optuna.

   Run from this directory:
       python tune.py                     # 50 trials (default)
       python tune.py --n-trials 100      # more trials
       python tune.py --n-trials 20       # quick smoke test

   Each trial trains one GATv2 configuration on a fixed 60/20 train/val split
   and returns val top_5pct_recall as the objective.  After the search finishes,
   best parameters are written back into gat_config.py automatically and saved
   to experiments/best_params.json.



Functions
---------

.. autoapisummary::

   gat.tune.objective
   gat.tune.main


Module Contents
---------------

.. py:function:: objective(trial: optuna.Trial, base_data: object, train_mask: torch.Tensor, val_mask: torch.Tensor) -> float

   Train one GATv2 configuration and return val top_5pct_recall.


.. py:function:: main() -> None

   Run hyperparameter search, then apply best params to gat_config.py.



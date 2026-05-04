gat.gat_output
==============

.. py:module:: gat.gat_output

.. autoapi-nested-parse::

   CSV writer and row schema for per-node prediction output.



Classes
-------

.. autoapisummary::

   gat.gat_output.PredictionRow


Functions
---------

.. autoapisummary::

   gat.gat_output.write_predictions


Module Contents
---------------

.. py:class:: PredictionRow

   Bases: :py:obj:`TypedDict`


   One row in the predictions CSV.

   - node_id: IP address graph node identifier.
   - actual_importance: Ground-truth importance score on the raw scale.
   - predicted_importance: Model-predicted importance score on the raw scale.
   - failure_probability: Normalised MC-Dropout variance in ``[0, 1]``.
   - composite_risk: ``actual_importance * failure_probability`` ranking metric.


   .. py:attribute:: node_id
      :type:  str


   .. py:attribute:: actual_importance
      :type:  float


   .. py:attribute:: predicted_importance
      :type:  float


   .. py:attribute:: failure_probability
      :type:  float


   .. py:attribute:: composite_risk
      :type:  float


.. py:function:: write_predictions(rows: list[PredictionRow], csv_path: pathlib.Path) -> None

   Write prediction rows to ``csv_path`` as UTF-8 CSV with a header line.

   Args:
       rows: List of :class:`PredictionRow` dicts (one per test node).
       csv_path: Destination CSV file. Any existing file at this path is overwritten.



gat.gat_config
==============

.. py:module:: gat.gat_config

.. autoapi-nested-parse::

   Configuration constants for the GATv2 training pipeline.



Attributes
----------

.. autoapisummary::

   gat.gat_config.DEVICE
   gat.gat_config.BASE_DIR
   gat.gat_config.GRAPHML_PATH
   gat.gat_config.PREDICTION_CSV_PATH
   gat.gat_config.NODE_FEATURE_KEYS
   gat.gat_config.IMPORTANCE_COMPONENT_KEYS
   gat.gat_config.EDGE_FEATURE_KEYS
   gat.gat_config.EDGE_DIM
   gat.gat_config.K_FOLDS
   gat.gat_config.NUM_HEADS
   gat.gat_config.HIDDEN_CHANNELS
   gat.gat_config.NUM_EPOCHS
   gat.gat_config.LEARNING_RATE
   gat.gat_config.WEIGHT_DECAY
   gat.gat_config.PATIENCE
   gat.gat_config.DROPOUT
   gat.gat_config.DROP_EDGE_PROB
   gat.gat_config.MC_DROPOUT_SAMPLES
   gat.gat_config.WEIGHT_MODE
   gat.gat_config.WEIGHT_SCALE
   gat.gat_config.RANKING_LOSS_WEIGHT
   gat.gat_config.RANKING_MARGIN
   gat.gat_config.RANKING_PAIRS
   gat.gat_config.SELECTION_METRIC


Module Contents
---------------

.. py:data:: DEVICE
   :type:  torch.device

.. py:data:: BASE_DIR
   :type:  pathlib.Path

.. py:data:: GRAPHML_PATH
   :type:  pathlib.Path

.. py:data:: PREDICTION_CSV_PATH
   :type:  pathlib.Path

.. py:data:: NODE_FEATURE_KEYS
   :type:  list[str]
   :value: ['in_flow_norm', 'out_flow_norm', 'flow_loss_norm', 'weighted_betweenness_norm', 'pagerank_norm']


.. py:data:: IMPORTANCE_COMPONENT_KEYS
   :type:  list[str]
   :value: ['weighted_betweenness_norm', 'pagerank_norm', 'flow_loss_norm']


.. py:data:: EDGE_FEATURE_KEYS
   :type:  list[str]
   :value: ['flow', 'bytes_per_sec', 'distance']


.. py:data:: EDGE_DIM
   :type:  int
   :value: 3


.. py:data:: K_FOLDS
   :type:  int
   :value: 10


.. py:data:: NUM_HEADS
   :type:  int
   :value: 2


.. py:data:: HIDDEN_CHANNELS
   :type:  int
   :value: 128


.. py:data:: NUM_EPOCHS
   :type:  int
   :value: 600


.. py:data:: LEARNING_RATE
   :type:  float
   :value: 0.0005981364320010053


.. py:data:: WEIGHT_DECAY
   :type:  float
   :value: 1.0969671773921393e-05


.. py:data:: PATIENCE
   :type:  int
   :value: 200


.. py:data:: DROPOUT
   :type:  float
   :value: 0.21621159087579556


.. py:data:: DROP_EDGE_PROB
   :type:  float
   :value: 0.30061413828907585


.. py:data:: MC_DROPOUT_SAMPLES
   :type:  int
   :value: 50


.. py:data:: WEIGHT_MODE
   :type:  str
   :value: 'quadratic'


.. py:data:: WEIGHT_SCALE
   :type:  float
   :value: 4.769499605360235


.. py:data:: RANKING_LOSS_WEIGHT
   :type:  float
   :value: 0.1


.. py:data:: RANKING_MARGIN
   :type:  float
   :value: 0.04568004906727538


.. py:data:: RANKING_PAIRS
   :type:  int
   :value: 128


.. py:data:: SELECTION_METRIC
   :type:  str
   :value: 'top_5pct_recall'



gat.gat_data
============

.. py:module:: gat.gat_data

.. autoapi-nested-parse::

   GraphML loading, feature engineering, and CV-mask construction.

   This module is the bridge between the raw GraphML file and the PyTorch Geometric
   ``Data`` object consumed by training. It computes node features, edge features,
   log-transformed labels, and rank-based sample weights, then splits the node set
   into K folds for cross-validation.



Functions
---------

.. autoapisummary::

   gat.gat_data.build_data
   gat.gat_data.build_kfold_masks
   gat.gat_data.standardize_features


Module Contents
---------------

.. py:function:: build_data(graphml_path: pathlib.Path) -> tuple[torch_geometric.data.Data, list[str]]

   Load a GraphML file and build the PyG ``Data`` object used for training.

   Pipeline:
       1. Read GraphML into a directed NetworkX graph.
       2. Build node feature vectors from ``NODE_FEATURE_KEYS`` plus in/out degree.
       3. Compute the importance label as ``log1p(sum(IMPORTANCE_COMPONENT_KEYS))``.
       4. Build ``[E, EDGE_DIM]`` edge feature tensor from ``EDGE_FEATURE_KEYS``.
       5. Compute rank-based sample weights via :func:`_build_sample_weights`.
       6. Move all tensors to ``DEVICE``.

   Args:
       graphml_path: Path to the input ``.graphml`` file.

   Returns:
       Tuple ``(data, node_ids)`` where ``data`` is the PyG ``Data`` (with
       ``x``, ``edge_index``, ``edge_attr``, ``y``, ``sample_weights``)
       and ``node_ids`` is the list of node IDs in the same order as ``data.x``.


.. py:function:: build_kfold_masks(num_nodes: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

   Build K-fold cross-validation masks over a node set.

   Nodes are randomly shuffled then split into ``K_FOLDS`` partitions.
   For fold *k*: partition *k* is the **test** set, partition *(k+1) % K*
   is the **validation** set, and the remaining partitions form the
   **training** set.

   Args:
       num_nodes: Total number of nodes in the graph.

   Returns:
       List of length ``K_FOLDS``, each entry a tuple of three boolean
       masks ``(train_mask, val_mask, test_mask)`` of shape ``[num_nodes]``,
       all placed on ``DEVICE``.


.. py:function:: standardize_features(features: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor

   Z-score normalise features using train-split-only mean and std.

   Computing statistics from ``features[train_mask]`` only (not the full set)
   prevents validation/test information from leaking into the normalisation.

   Args:
       features: Float tensor of shape ``[num_nodes, num_features]``.
       train_mask: Boolean mask of length ``num_nodes`` selecting training nodes.

   Returns:
       Normalised tensor of the same shape and device as ``features``.
       Std values smaller than ``1e-6`` are clamped to avoid division blow-ups.



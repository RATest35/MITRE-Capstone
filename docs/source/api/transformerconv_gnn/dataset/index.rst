transformerconv_gnn.dataset
===========================

.. py:module:: transformerconv_gnn.dataset


Attributes
----------

.. autoapisummary::

   transformerconv_gnn.dataset.EPSILON


Classes
-------

.. autoapisummary::

   transformerconv_gnn.dataset.GraphDataset
   transformerconv_gnn.dataset.SamplerConfig
   transformerconv_gnn.dataset.RootedSubgraphDataset


Functions
---------

.. autoapisummary::

   transformerconv_gnn.dataset.load_graph_dataset
   transformerconv_gnn.dataset.standardize_features
   transformerconv_gnn.dataset.subnet_group_split


Module Contents
---------------

.. py:data:: EPSILON
   :type:  float
   :value: 1e-06


.. py:class:: GraphDataset

   Store graph tensors and edge-attributed adjacency lists.

   :ivar node_ids: Ordered node identifiers from the graph.
   :ivar node_id_to_index: Mapping from node identifier to tensor index.
   :ivar group_keys: Split group keys for each node.
   :ivar features: Node feature matrix.
   :ivar targets: Log-scaled target values.
   :ivar raw_targets: Original target values from GraphML.
   :ivar sample_weights: Per-node sample weights.
   :ivar in_neighbors: Incoming neighbors, flow, and bytes-per-second values.
   :ivar out_neighbors: Outgoing neighbors, flow, and bytes-per-second values.


   .. py:attribute:: node_ids
      :type:  list[str]


   .. py:attribute:: node_id_to_index
      :type:  dict[str, int]


   .. py:attribute:: group_keys
      :type:  list[str]


   .. py:attribute:: features
      :type:  torch.Tensor


   .. py:attribute:: targets
      :type:  torch.Tensor


   .. py:attribute:: raw_targets
      :type:  torch.Tensor


   .. py:attribute:: sample_weights
      :type:  torch.Tensor


   .. py:attribute:: in_neighbors
      :type:  list[list[tuple[int, float, float]]]


   .. py:attribute:: out_neighbors
      :type:  list[list[tuple[int, float, float]]]


.. py:class:: SamplerConfig

   Configure rooted subgraph neighbor sampling.

   :ivar num_hops: Number of sampling hops from the root node.
   :ivar max_in_neighbors: Maximum incoming neighbors per node.
   :ivar max_out_neighbors: Maximum outgoing neighbors per node.


   .. py:attribute:: num_hops
      :type:  int


   .. py:attribute:: max_in_neighbors
      :type:  int


   .. py:attribute:: max_out_neighbors
      :type:  int


.. py:function:: load_graph_dataset(graphml_path: pathlib.Path, split_prefix_len: int = 2, sample_weight_max: float = 5.0) -> GraphDataset

   Load GraphML data into tensors and edge-attributed neighbors.

   :param graphml_path: Path to the GraphML file.
   :param split_prefix_len: Number of IPv4 octets used for split groups.
   :param sample_weight_max: Maximum target-rank sample weight.
   :return: Loaded graph dataset.


.. py:function:: standardize_features(features: torch.Tensor, train_indices: numpy.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

   Standardize features using train split statistics.

   :param features: Full node feature matrix.
   :param train_indices: Node indices used to compute statistics.
   :return: Standardized features, feature mean, and feature standard deviation.


.. py:function:: subnet_group_split(group_keys: list[str], train_ratio: float, val_ratio: float, seed: int) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]

   Split node indices by subnet group.

   :param group_keys: Split group key for each node.
   :param train_ratio: Ratio of groups assigned to training.
   :param val_ratio: Ratio of groups assigned to validation.
   :param seed: Random seed for group shuffling.
   :return: Train, validation, and test node indices.


.. py:class:: RootedSubgraphDataset(graph_dataset: GraphDataset, node_indices: numpy.ndarray, allowed_node_indices: numpy.ndarray, sampler_config: SamplerConfig)

   Bases: :py:obj:`torch.utils.data.Dataset`\ [\ :py:obj:`torch_geometric.data.Data`\ ]


   Create PyG rooted subgraph samples with edge attributes.

   :param graph_dataset: Full graph dataset.
   :param node_indices: Root node indices exposed by this dataset.
   :param allowed_node_indices: Node indices allowed during sampling.
   :param sampler_config: Rooted subgraph sampling configuration.


   .. py:attribute:: graph_dataset


   .. py:attribute:: node_indices


   .. py:attribute:: allowed_mask


   .. py:attribute:: sampler_config


   .. py:method:: __len__() -> int

      Return the number of root nodes.

      :return: Number of root nodes.



   .. py:method:: __getitem__(item: int) -> torch_geometric.data.Data

      Build one rooted subgraph sample.

      :param item: Position of the root node in this dataset.
      :return: PyG data object for the sampled rooted subgraph.




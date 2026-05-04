pnaconv_gnn.train
=================

.. py:module:: pnaconv_gnn.train


Attributes
----------

.. autoapisummary::

   pnaconv_gnn.train.BASE_DIR


Functions
---------

.. autoapisummary::

   pnaconv_gnn.train.parse_args
   pnaconv_gnn.train.prepare_dataset
   pnaconv_gnn.train.build_loader
   pnaconv_gnn.train.build_degree_histogram
   pnaconv_gnn.train.run_epoch
   pnaconv_gnn.train.evaluate_ranking
   pnaconv_gnn.train.main


Module Contents
---------------

.. py:data:: BASE_DIR

.. py:function:: parse_args() -> argparse.Namespace

   Parse command-line options for PNA training.

   :return: Parsed command-line arguments.


.. py:function:: prepare_dataset(args: argparse.Namespace) -> tuple[dataset.GraphDataset, torch.Tensor, torch.Tensor, numpy.ndarray, numpy.ndarray, numpy.ndarray]

   Load, split, and standardize the graph dataset.

   :param args: Parsed training arguments.
   :return: Dataset, feature mean, feature standard deviation, and split indices.


.. py:function:: build_loader(dataset: dataset.GraphDataset, node_indices: numpy.ndarray, allowed_node_indices: numpy.ndarray, sampler_config: dataset.SamplerConfig, batch_size: int, shuffle: bool) -> torch_geometric.loader.DataLoader

   Build a rooted-subgraph data loader.

   :param dataset: Full graph dataset.
   :param node_indices: Root node indices exposed by the loader.
   :param allowed_node_indices: Node indices allowed during sampling.
   :param sampler_config: Rooted subgraph sampling configuration.
   :param batch_size: Number of samples per batch.
   :param shuffle: Whether to shuffle the dataset.
   :return: PyG data loader for rooted subgraphs.


.. py:function:: build_degree_histogram(dataset: dataset.GraphDataset, allowed_node_indices: numpy.ndarray) -> torch.Tensor

   Build the degree histogram for PNA scalers.

   :param dataset: Full graph dataset.
   :param allowed_node_indices: Node indices allowed in the histogram.
   :return: Degree histogram tensor.


.. py:function:: run_epoch(model: model.PNAGNN, loader: torch_geometric.loader.DataLoader, optimizer: torch.optim.Adam | None, device: torch.device) -> float

   Run one training or evaluation epoch.

   :param model: PNA model.
   :param loader: Data loader for a split.
   :param optimizer: Optimizer for training, or ``None`` for evaluation.
   :param device: Device used for tensor computation.
   :return: Average masked loss for the epoch.


.. py:function:: evaluate_ranking(model: model.PNAGNN, loader: torch_geometric.loader.DataLoader, device: torch.device) -> dict[str, float]

   Compute top-risk retrieval metrics.

   :param model: PNA model.
   :param loader: Data loader for a split.
   :param device: Device used for tensor computation.
   :return: Precision and NDCG metrics at the top five percent.


.. py:function:: main() -> None

   Train and save the PNA model.

   :return: None.



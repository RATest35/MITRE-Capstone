transformerconv_gnn.train
=========================

.. py:module:: transformerconv_gnn.train


Attributes
----------

.. autoapisummary::

   transformerconv_gnn.train.DEFAULT_LAYER_DIMS


Functions
---------

.. autoapisummary::

   transformerconv_gnn.train.parse_args
   transformerconv_gnn.train.prepare_dataset
   transformerconv_gnn.train.build_loader
   transformerconv_gnn.train.compute_weighted_mse_loss
   transformerconv_gnn.train.compute_pairwise_ranking_loss
   transformerconv_gnn.train.compute_batch_loss
   transformerconv_gnn.train.run_epoch
   transformerconv_gnn.train.evaluate_ranking
   transformerconv_gnn.train.main


Module Contents
---------------

.. py:data:: DEFAULT_LAYER_DIMS
   :value: [128, 96, 64, 32]


.. py:function:: parse_args() -> argparse.Namespace

   Parse command-line options for TransformerConv training.

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


.. py:function:: compute_weighted_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor

   Compute sample-weighted mean squared error.

   :param predictions: Predicted target values.
   :param targets: Expected target values.
   :param weights: Sample weights for each prediction.
   :return: Weighted MSE loss.


.. py:function:: compute_pairwise_ranking_loss(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor

   Compute weighted pairwise ranking loss.

   :param predictions: Predicted target values.
   :param targets: Expected target values.
   :param weights: Sample weights for each prediction.
   :return: Weighted pairwise ranking loss.


.. py:function:: compute_batch_loss(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, ranking_alpha: float) -> torch.Tensor

   Combine regression and ranking losses.

   :param predictions: Predicted target values.
   :param targets: Expected target values.
   :param weights: Sample weights for each prediction.
   :param ranking_alpha: Multiplier applied to ranking loss.
   :return: Combined batch loss.


.. py:function:: run_epoch(model: model.TransformerConvGNN, loader: torch_geometric.loader.DataLoader, optimizer: torch.optim.Adam | None, device: torch.device, ranking_alpha: float) -> float

   Run one training or evaluation epoch.

   :param model: TransformerConv model.
   :param loader: Data loader for a split.
   :param optimizer: Optimizer for training, or ``None`` for evaluation.
   :param device: Device used for tensor computation.
   :param ranking_alpha: Multiplier applied to ranking loss.
   :return: Average masked loss for the epoch.


.. py:function:: evaluate_ranking(model: model.TransformerConvGNN, loader: torch_geometric.loader.DataLoader, device: torch.device) -> dict[str, float]

   Compute top-risk retrieval metrics.

   :param model: TransformerConv model.
   :param loader: Data loader for a split.
   :param device: Device used for tensor computation.
   :return: Precision and NDCG metrics at the top five percent.


.. py:function:: main() -> None

   Train and save the TransformerConv model.

   :return: None.



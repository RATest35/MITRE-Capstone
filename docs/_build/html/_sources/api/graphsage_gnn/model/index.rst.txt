graphsage_gnn.model
===================

.. py:module:: graphsage_gnn.model


Classes
-------

.. autoapisummary::

   graphsage_gnn.model.GraphSAGEGNN


Module Contents
---------------

.. py:class:: GraphSAGEGNN(input_dim: int, hidden_dim: int, num_layers: int, dropout: float)

   Bases: :py:obj:`torch.nn.Module`


   GraphSAGE regressor for composite score prediction.

   :param input_dim: Number of input node features.
   :param hidden_dim: Hidden representation size.
   :param num_layers: Number of GraphSAGE message-passing layers.
   :param dropout: Dropout probability.


   .. py:attribute:: encoder


   .. py:attribute:: layers


   .. py:attribute:: norms


   .. py:attribute:: activation


   .. py:attribute:: dropout


   .. py:attribute:: head


   .. py:method:: forward(x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor

      Predict log composite score for each node.

      :param x: Node feature matrix.
      :param edge_index: Graph connectivity in COO format.
      :param edge_weight: Edge weights accepted for interface compatibility.
      :return: Predicted log composite score for each node.




pnaconv_gnn.model
=================

.. py:module:: pnaconv_gnn.model


Classes
-------

.. autoapisummary::

   pnaconv_gnn.model.PNAGNN


Module Contents
---------------

.. py:class:: PNAGNN(input_dim: int, hidden_dim: int, num_layers: int, dropout: float, degree_histogram: torch.Tensor)

   Bases: :py:obj:`torch.nn.Module`


   PNA regressor for composite score prediction.

   :param input_dim: Number of input node features.
   :param hidden_dim: Hidden representation size.
   :param num_layers: Number of PNA message-passing layers.
   :param dropout: Dropout probability.
   :param degree_histogram: Degree histogram used by PNA scalers.


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
      :param edge_weight: Edge flow weights.
      :return: Predicted log composite score for each node.




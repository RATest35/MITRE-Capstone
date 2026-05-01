transformerconv_gnn.model
=========================

.. py:module:: transformerconv_gnn.model


Classes
-------

.. autoapisummary::

   transformerconv_gnn.model.TransformerConvGNN


Module Contents
---------------

.. py:class:: TransformerConvGNN(input_dim: int, layer_dims: list[int], dropout: float)

   Bases: :py:obj:`torch.nn.Module`


   TransformerConv regressor for composite score prediction.

   :param input_dim: Number of input node features.
   :param layer_dims: Hidden dimensions for TransformerConv layers.
   :param dropout: Dropout probability.


   .. py:attribute:: encoder


   .. py:attribute:: layers


   .. py:attribute:: residual_projections


   .. py:attribute:: norms


   .. py:attribute:: output_dim


   .. py:attribute:: activation


   .. py:attribute:: dropout


   .. py:attribute:: head


   .. py:method:: forward(x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor

      Predict log composite score for each node.

      :param x: Node feature matrix.
      :param edge_index: Graph connectivity in COO format.
      :param edge_attr: Edge feature matrix.
      :return: Predicted log composite score for each node.




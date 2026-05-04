gat.gat_model
=============

.. py:module:: gat.gat_model

.. autoapi-nested-parse::

   GATv2 regressor model for node-level flow-loss prediction.



Classes
-------

.. autoapisummary::

   gat.gat_model.GATv2Regressor


Module Contents
---------------

.. py:class:: GATv2Regressor(input_channels: int, hidden_channels: int = 128, num_heads: int = 4, dropout: float = 0.1569455125389924, drop_edge_prob: float = 0.22203374123224115, edge_dim: int | None = None)

   Bases: :py:obj:`torch.nn.Module`


   Two-layer GATv2 network that predicts a scalar value per node.

   Architecture
   ------------
   Layer 1 : GATv2Conv(in → hidden, heads=H, concat=True)  → LayerNorm → ELU → Dropout
   Layer 2 : GATv2Conv(hidden*H → hidden, heads=H, concat=False) → LayerNorm → ELU → Dropout
   Layer 3 : GATv2Conv(hidden*H → hidden, heads=H, concat=False) → LayerNorm → ELU
   Head    : Linear(hidden → 1) → squeeze

   Edge features are forwarded through both conv layers via *edge_dim*.


   .. py:attribute:: dropout
      :value: 0.1569455125389924



   .. py:attribute:: drop_edge_prob
      :value: 0.22203374123224115



   .. py:attribute:: conv1


   .. py:attribute:: norm1


   .. py:attribute:: conv2


   .. py:attribute:: norm2


   .. py:attribute:: conv3


   .. py:attribute:: norm3


   .. py:attribute:: head


   .. py:method:: forward(data: torch_geometric.data.Data) -> torch.Tensor

      Run the forward pass and return per-node scalar predictions.

      During training, applies DropEdge to ``edge_index`` (and slices
      ``edge_attr`` accordingly). The output is the squeezed scalar
      head (no activation), suitable for regression.

      Args:
          data: PyG ``Data`` object with ``x``, ``edge_index``, and optional
              ``edge_attr``.

      Returns:
          ``Tensor`` of shape ``[num_nodes]`` containing one prediction per node.




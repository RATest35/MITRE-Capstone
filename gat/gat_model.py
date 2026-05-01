"""GATv2 regressor model for node-level flow-loss prediction."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dropout_edge


class GATv2Regressor(nn.Module):
    """Two-layer GATv2 network that predicts a scalar value per node.

    Architecture
    ------------
    Layer 1 : GATv2Conv(in → hidden, heads=H, concat=True)  → LayerNorm → ELU → Dropout
    Layer 2 : GATv2Conv(hidden*H → hidden, heads=H, concat=False) → LayerNorm → ELU → Dropout
    Layer 3 : GATv2Conv(hidden*H → hidden, heads=H, concat=False) → LayerNorm → ELU
    Head    : Linear(hidden → 1) → squeeze

    Edge features are forwarded through both conv layers via *edge_dim*.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1569455125389924,
        drop_edge_prob: float = 0.22203374123224115,
        edge_dim: int | None = None,
    ) -> None:
        """Build the three GATv2Conv layers, layer-norms, and the linear head.

        Args:
            input_channels: Number of input node features.
            hidden_channels: Per-head hidden width inside each GATv2Conv layer.
            num_heads: Number of attention heads per GATv2Conv layer.
            dropout: Dropout probability applied to attention weights and after
                each conv layer's activation.
            drop_edge_prob: Probability of dropping each edge during training
                (DropEdge regularisation). Set to ``0`` to disable.
            edge_dim: Dimensionality of edge features. ``None`` disables edge
                features entirely.
        """
        super().__init__()
        self.dropout = dropout
        self.drop_edge_prob = drop_edge_prob

        self.conv1 = GATv2Conv(
            in_channels=input_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True,
        )
        self.norm1 = nn.LayerNorm(hidden_channels * num_heads)

        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True,
        )
        self.norm2 = nn.LayerNorm(hidden_channels * num_heads)

        self.conv3 = GATv2Conv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=False,
        )
        self.norm3 = nn.LayerNorm(hidden_channels)

        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, data: Data) -> Tensor:
        """Run the forward pass and return per-node scalar predictions.

        During training, applies DropEdge to ``edge_index`` (and slices
        ``edge_attr`` accordingly). The output is the squeezed scalar
        head (no activation), suitable for regression.

        Args:
            data: PyG ``Data`` object with ``x``, ``edge_index``, and optional
                ``edge_attr``.

        Returns:
            ``Tensor`` of shape ``[num_nodes]`` containing one prediction per node.
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        if self.drop_edge_prob > 0 and self.training:
            edge_index, edge_mask = dropout_edge(
                edge_index, p=self.drop_edge_prob, training=self.training,)
            
            if edge_attr is not None:
                edge_attr = edge_attr[edge_mask]

        # Layer 1
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=True)

        # Layer 2
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.norm2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=True)

        # Layer 3
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.norm3(x)
        x = F.elu(x)
    
        return self.head(x).squeeze(-1)

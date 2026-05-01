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

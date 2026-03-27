"""Model definition for the GNN example."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class CompositeScoreGNN(nn.Module):
    """GraphSAGE regressor for node composite score adjustment."""

    def __init__(self, input_channels: int, hidden_channels: int, dropout: float) -> None:
        """Initialize the model."""
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(input_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, data: Data) -> Tensor:
        """Return one prediction per node."""
        hidden_1 = self.conv1(data.x, data.edge_index)
        hidden_1 = F.relu(hidden_1)
        hidden_1 = F.dropout(hidden_1, p=self.dropout, training=self.training)

        hidden_2 = self.conv2(hidden_1, data.edge_index)
        hidden_2 = F.relu(hidden_2)
        hidden_2 = F.dropout(hidden_2, p=self.dropout, training=self.training)

        hidden_3 = self.conv3(hidden_2, data.edge_index)
        hidden_3 = F.relu(hidden_3)
        hidden = hidden_1 + hidden_2 + hidden_3
        return self.head(hidden).squeeze(-1)

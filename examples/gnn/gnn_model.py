"""Model definition for the GNN example."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class FlowLossGNN(nn.Module):
    """GraphSAGE regressor for node flow loss."""

    def __init__(self, input_channels: int, hidden_channels: int, dropout: float) -> None:
        """Initialize the model."""
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(input_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, data: Data) -> Tensor:
        """Return one prediction per node."""
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        return self.head(x).squeeze(-1)

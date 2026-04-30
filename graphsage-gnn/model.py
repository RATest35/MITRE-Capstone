from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGEGNN(nn.Module):
    """GraphSAGE regressor for composite score prediction.

    :param input_dim: Number of input node features.
    :param hidden_dim: Hidden representation size.
    :param num_layers: Number of GraphSAGE message-passing layers.
    :param dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Initialize the encoder, GraphSAGE layers, and head.

        :param input_dim: Number of input node features.
        :param hidden_dim: Hidden representation size.
        :param num_layers: Number of GraphSAGE message-passing layers.
        :param dropout: Dropout probability.
        :return: None.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Predict log composite score for each node.

        :param x: Node feature matrix.
        :param edge_index: Graph connectivity in COO format.
        :param edge_weight: Edge weights accepted for interface compatibility.
        :return: Predicted log composite score for each node.
        """
        del edge_weight
        hidden = self.encoder(x)
        for conv, norm in zip(self.layers, self.norms):
            updated = self.activation(conv(hidden, edge_index))
            hidden = norm(self.dropout(updated) + hidden)
        return self.head(hidden).squeeze(-1)

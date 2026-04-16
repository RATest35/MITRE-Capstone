from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import TransformerConv


class TransformerConvGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Project raw node features into the hidden representation.
        # ---------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ---------------------------------------------------------
        # Build edge-aware TransformerConv blocks with residual norms.
        # ---------------------------------------------------------
        self.layers = nn.ModuleList(
            TransformerConv(hidden_dim, hidden_dim, edge_dim=2, dropout=dropout)
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # ---------------------------------------------------------
        # Map the hidden state to one regression score per node.
        # ---------------------------------------------------------
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
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Encode node features before graph message passing.
        # ---------------------------------------------------------
        hidden = self.encoder(x)

        # ---------------------------------------------------------
        # Update node states with edge attributes for each layer.
        # ---------------------------------------------------------
        for conv, norm in zip(self.layers, self.norms):
            updated = self.activation(conv(hidden, edge_index, edge_attr=edge_attr))
            hidden = norm(self.dropout(updated) + hidden)

        # ---------------------------------------------------------
        # Predict the final node-level regression target.
        # ---------------------------------------------------------
        return self.head(hidden).squeeze(-1)

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import TransformerConv


class TransformerConvGNN(nn.Module):
    """TransformerConv regressor for composite score prediction."""

    def __init__(
        self,
        input_dim: int,
        layer_dims: list[int],
        dropout: float,
    ) -> None:
        """Initialize edge-aware TransformerConv layers and head."""
        super().__init__()
        first_hidden_dim = layer_dims[0]
        layer_pairs = list(zip(layer_dims[:-1], layer_dims[1:]))

        # ---------------------------------------------------------
        # Project raw node features into the hidden representation.
        # ---------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, first_hidden_dim),
            nn.LayerNorm(first_hidden_dim),
            nn.ReLU(),
        )

        # ---------------------------------------------------------
        # Build edge-aware TransformerConv blocks with shrinking widths.
        # ---------------------------------------------------------
        self.layers = nn.ModuleList([
            TransformerConv(in_dim, out_dim, edge_dim=2, dropout=dropout)
            for in_dim, out_dim in [(first_hidden_dim, first_hidden_dim), *layer_pairs]
        ])
        self.residual_projections = nn.ModuleList([
            nn.Identity(),
            *[nn.Linear(in_dim, out_dim) for in_dim, out_dim in layer_pairs],
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim)
            for dim in layer_dims
        ])
        self.output_dim = layer_dims[-1]
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # ---------------------------------------------------------
        # Map the final hidden state to one regression score per node.
        # ---------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Predict log composite score for each node."""
        # ---------------------------------------------------------
        # Encode node features before graph message passing.
        # ---------------------------------------------------------
        hidden = self.encoder(x)

        # ---------------------------------------------------------
        # Update node states with edge attributes for each layer.
        # ---------------------------------------------------------
        for conv, residual_projection, norm in zip(self.layers, self.residual_projections, self.norms):
            residual = residual_projection(hidden)
            updated = self.activation(conv(hidden, edge_index, edge_attr=edge_attr))
            hidden = norm(self.dropout(updated) + residual)

        # ---------------------------------------------------------
        # Predict the final node-level regression target.
        # ---------------------------------------------------------
        return self.head(hidden).squeeze(-1)

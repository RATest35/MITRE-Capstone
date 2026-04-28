from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import PNAConv


class PNAGNN(nn.Module):
    # ---------------------------------------------------------
    # Build the PNA encoder, message-passing stack, and head.
    # ---------------------------------------------------------
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        degree_histogram: torch.Tensor,
    ) -> None:
        super().__init__()

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList(
            PNAConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=degree_histogram,
                edge_dim=1,
                towers=1,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    # ---------------------------------------------------------
    # Predict the regression target from node and edge tensors.
    # ---------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(x)
        edge_attr = edge_weight.unsqueeze(-1)

        for conv, norm in zip(self.layers, self.norms):
            updated = self.activation(conv(hidden, edge_index, edge_attr=edge_attr))
            hidden = norm(self.dropout(updated) + hidden)

        return self.head(hidden).squeeze(-1)

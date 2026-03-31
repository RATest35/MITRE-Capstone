from __future__ import annotations

import torch
from torch import nn


EPSILON: float = 1e-6


def _weighted_mean_aggregate(
    source_features: torch.Tensor,
    index: torch.Tensor,
    weights: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Aggregate neighbor features with weighted mean."""
    hidden_dim = source_features.size(1)
    weighted_messages = source_features * weights.unsqueeze(1)
    aggregated = torch.zeros(num_nodes, hidden_dim, device=source_features.device, dtype=source_features.dtype)
    denominator = torch.zeros(num_nodes, 1, device=source_features.device, dtype=source_features.dtype)
    aggregated.index_add_(0, index, weighted_messages)
    denominator.index_add_(0, index, weights.unsqueeze(1))
    return aggregated / denominator.clamp_min(EPSILON)


class DirectedSAGELayer(nn.Module):
    """GraphSAGE layer with separate incoming and outgoing aggregation."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.in_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim * 3, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one directed weighted message-passing step."""
        if edge_index.numel() == 0:
            base = torch.cat(
                [self.self_linear(x), torch.zeros_like(x), torch.zeros_like(x)],
                dim=1,
            )
            updated = self.update(base)
            updated = self.activation(updated)
            updated = self.dropout(updated)
            return self.norm(updated + x)

        source = edge_index[0]
        target = edge_index[1]
        weights = edge_weight + 1.0

        incoming = _weighted_mean_aggregate(x[source], target, weights, x.size(0))
        outgoing = _weighted_mean_aggregate(x[target], source, weights, x.size(0))
        base = torch.cat(
            [self.self_linear(x), self.in_linear(incoming), self.out_linear(outgoing)],
            dim=1,
        )
        updated = self.update(base)
        updated = self.activation(updated)
        updated = self.dropout(updated)
        return self.norm(updated + x)


class CompositeScoreGNN(nn.Module):
    """Directed GraphSAGE regressor for composite score prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            DirectedSAGELayer(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        )
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
        """Predict log composite score for each node."""
        hidden = self.encoder(x)
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_weight)
        return self.head(hidden).squeeze(-1)

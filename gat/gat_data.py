"""GraphML loading, feature engineering, and CV-mask construction.

This module is the bridge between the raw GraphML file and the PyTorch Geometric
``Data`` object consumed by training. It computes node features, edge features,
log-transformed labels, and rank-based sample weights, then splits the node set
into K folds for cross-validation.
"""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data

from gat_config import (
    DEVICE,
    EDGE_FEATURE_KEYS,
    IMPORTANCE_COMPONENT_KEYS,
    K_FOLDS,
    NODE_FEATURE_KEYS,
    WEIGHT_MODE,
    WEIGHT_SCALE,
)


def build_data(graphml_path: Path) -> tuple[Data, list[str]]:
    """Load a GraphML file and build the PyG ``Data`` object used for training.

    Pipeline:
        1. Read GraphML into a directed NetworkX graph.
        2. Build node feature vectors from ``NODE_FEATURE_KEYS`` plus in/out degree.
        3. Compute the importance label as ``log1p(sum(IMPORTANCE_COMPONENT_KEYS))``.
        4. Build ``[E, EDGE_DIM]`` edge feature tensor from ``EDGE_FEATURE_KEYS``.
        5. Compute rank-based sample weights via :func:`_build_sample_weights`.
        6. Move all tensors to ``DEVICE``.

    Args:
        graphml_path: Path to the input ``.graphml`` file.

    Returns:
        Tuple ``(data, node_ids)`` where ``data`` is the PyG ``Data`` (with
        ``x``, ``edge_index``, ``edge_attr``, ``y``, ``sample_weights``)
        and ``node_ids`` is the list of node IDs in the same order as ``data.x``.
    """
    graph: nx.DiGraph = nx.read_graphml(graphml_path)
    node_ids: list[str] = list(graph.nodes())
    node_index: dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}

    node_features = [_node_feature_vector(graph, nid) for nid in node_ids]
    raw_targets = [_raw_importance_score(graph, nid) for nid in node_ids]
    labels = [math.log1p(score) for score in raw_targets]

    edge_pairs = [[node_index[u], node_index[v]] for u, v in graph.edges()]
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    # Build multi-dimensional edge features from EDGE_FEATURE_KEYS
    edge_feature_rows: list[list[float]] = []
    for _, _, edata in graph.edges(data=True):
        edge_feature_rows.append(
            [float(edata.get(key, 0.0)) for key in EDGE_FEATURE_KEYS]
        )
    edge_attr = torch.tensor(edge_feature_rows, dtype=torch.float32)

    sample_weights = _build_sample_weights(
        torch.tensor(raw_targets, dtype=torch.float32),
        WEIGHT_MODE,
        WEIGHT_SCALE,
    )

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(labels, dtype=torch.float32),
        sample_weights=sample_weights,
    )
    return data.to(DEVICE), node_ids


def build_kfold_masks(num_nodes: int) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Build K-fold cross-validation masks over a node set.

    Nodes are randomly shuffled then split into ``K_FOLDS`` partitions.
    For fold *k*: partition *k* is the **test** set, partition *(k+1) % K*
    is the **validation** set, and the remaining partitions form the
    **training** set.

    Args:
        num_nodes: Total number of nodes in the graph.

    Returns:
        List of length ``K_FOLDS``, each entry a tuple of three boolean
        masks ``(train_mask, val_mask, test_mask)`` of shape ``[num_nodes]``,
        all placed on ``DEVICE``.
    """
    perm = torch.randperm(num_nodes)
    fold_size = num_nodes // K_FOLDS
    fold_indices: list[Tensor] = []

    for k in range(K_FOLDS):
        start = k * fold_size
        end = start + fold_size if k < K_FOLDS - 1 else num_nodes
        fold_indices.append(perm[start:end])

    masks: list[tuple[Tensor, Tensor, Tensor]] = []
    for k in range(K_FOLDS):
        test_idx = fold_indices[k]
        val_idx = fold_indices[(k + 1) % K_FOLDS]

        train_parts = [fold_indices[j] for j in range(K_FOLDS) if j != k and j != (k + 1) % K_FOLDS]
        train_idx = torch.cat(train_parts)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        masks.append((
            train_mask.to(DEVICE),
            val_mask.to(DEVICE),
            test_mask.to(DEVICE),
        ))

    return masks


def standardize_features(features: Tensor, train_mask: Tensor) -> Tensor:
    """Z-score normalise features using train-split-only mean and std.

    Computing statistics from ``features[train_mask]`` only (not the full set)
    prevents validation/test information from leaking into the normalisation.

    Args:
        features: Float tensor of shape ``[num_nodes, num_features]``.
        train_mask: Boolean mask of length ``num_nodes`` selecting training nodes.

    Returns:
        Normalised tensor of the same shape and device as ``features``.
        Std values smaller than ``1e-6`` are clamped to avoid division blow-ups.
    """
    train_features = features[train_mask]
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (features - mean) / std



# Private helpers
# ---------------------------------------------------------------------------

def _node_feature_vector(graph: nx.DiGraph, node_id: str) -> list[float]:
    """Build the numeric feature list for a single node from precomputed attributes."""
    attrs = graph.nodes[node_id]
    base = [float(attrs.get(key, 0.0)) for key in NODE_FEATURE_KEYS]
    structural = [float(graph.in_degree(node_id)), float(graph.out_degree(node_id))]
    return base + structural


def _raw_importance_score(graph: nx.DiGraph, node_id: str) -> float:
    """Return the raw importance score for a single node."""
    attrs = graph.nodes[node_id]
    return sum(float(attrs.get(key, 0.0)) for key in IMPORTANCE_COMPONENT_KEYS)


def _build_sample_weights(raw_targets: Tensor, mode: str, scale: float) -> Tensor:
    """Build sample weights from target rank (matches composite pipeline).

    Higher-ranked nodes (higher importance) get larger weights so the model
    focuses more on predicting critical nodes accurately.
    """
    normalised_rank = torch.argsort(torch.argsort(raw_targets)).float() / max(len(raw_targets) - 1, 1)
    if mode == "linear":
        return 1.0 + scale * normalised_rank
    if mode == "sqrt":
        return 1.0 + scale * normalised_rank.sqrt()
    if mode == "quadratic":
        return 1.0 + scale * normalised_rank.square()
    raise ValueError(f"Unsupported weight mode: {mode}")

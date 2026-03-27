"""Data loading and preprocessing for the GNN example."""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data

from gnn_config import TRAIN_RATIO, VAL_RATIO

COMPOSITE_SCORE_ATTR = "composite_score"
FLOW_ATTR = "flow"


def build_data(graphml_path: Path) -> tuple[Data, list[str]]:
    """Build a PyG graph from GraphML."""
    graph = nx.read_graphml(graphml_path)
    node_ids = list(graph.nodes())
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}

    records = [_build_node_record(graph, node_id) for node_id in node_ids]
    node_features = [record[0] for record in records]
    local_flow_losses = [record[1] for record in records]
    composite_scores = [record[2] for record in records]
    labels = [math.log1p(composite_score - local_flow_loss) for local_flow_loss, composite_score in zip(local_flow_losses, composite_scores)]
    edge_pairs = [[node_index[source], node_index[target]] for source, target in graph.edges()]
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.float32),
        local_flow_loss=torch.tensor(local_flow_losses, dtype=torch.float32),
        composite_score=torch.tensor(composite_scores, dtype=torch.float32),
    )
    return data, node_ids


def build_masks(num_nodes: int) -> tuple[Tensor, Tensor, Tensor]:
    """Build train, validation, and test masks."""
    train_end = int(num_nodes * TRAIN_RATIO)
    val_end = train_end + int(num_nodes * VAL_RATIO)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True
    return train_mask, val_mask, test_mask


def standardize_features(features: Tensor, train_mask: Tensor) -> Tensor:
    """Standardize features with train-only statistics."""
    train_features = features[train_mask]
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (features - mean) / std


def _build_node_record(graph: nx.DiGraph, node_id: str) -> tuple[list[float], float, float]:
    """Build features and targets for one node."""
    incoming_flows = [float(edge_data[FLOW_ATTR]) for _, _, edge_data in graph.in_edges(node_id, data=True)]
    outgoing_flows = [float(edge_data[FLOW_ATTR]) for _, _, edge_data in graph.out_edges(node_id, data=True)]

    incoming_flow_sum = sum(incoming_flows)
    outgoing_flow_sum = sum(outgoing_flows)
    total_flow = incoming_flow_sum + outgoing_flow_sum
    in_degree = float(graph.in_degree(node_id))
    out_degree = float(graph.out_degree(node_id))
    total_degree = float(graph.degree(node_id))
    incoming_flow_mean = incoming_flow_sum / max(len(incoming_flows), 1)
    outgoing_flow_mean = outgoing_flow_sum / max(len(outgoing_flows), 1)
    incoming_flow_max = max(incoming_flows, default=0.0)
    outgoing_flow_max = max(outgoing_flows, default=0.0)
    incoming_flow_std = _build_flow_std(incoming_flows, incoming_flow_mean)
    outgoing_flow_std = _build_flow_std(outgoing_flows, outgoing_flow_mean)
    incoming_flow_nonzero_count = float(sum(flow > 0.0 for flow in incoming_flows))
    outgoing_flow_nonzero_count = float(sum(flow > 0.0 for flow in outgoing_flows))
    composite_score = float(graph.nodes[node_id][COMPOSITE_SCORE_ATTR])

    features = [
        in_degree,
        out_degree,
        total_degree,
        incoming_flow_sum,
        outgoing_flow_sum,
        total_flow,
        incoming_flow_mean,
        outgoing_flow_mean,
        incoming_flow_max,
        outgoing_flow_max,
        incoming_flow_std,
        outgoing_flow_std,
        incoming_flow_nonzero_count,
        outgoing_flow_nonzero_count,
    ]
    return features, total_flow, composite_score


def _build_flow_std(flows: list[float], mean_flow: float) -> float:
    """Build one flow standard deviation."""
    if not flows:
        return 0.0

    variance = sum((flow - mean_flow) ** 2 for flow in flows) / len(flows)
    return math.sqrt(variance)

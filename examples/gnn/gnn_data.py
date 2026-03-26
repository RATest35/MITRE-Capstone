"""Data loading and preprocessing for the GNN example."""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data

from gnn_config import TRAIN_RATIO, VAL_RATIO

IN_FLOW_ATTR = "in_flow"
OUT_FLOW_ATTR = "out_flow"
FLOW_LOSS_ATTR = "flow_loss"
WEIGHTED_BETWEENNESS_ATTR = "weighted_betweenness"
PAGERANK_ATTR = "pagerank"
IMPORTANCE_ATTR = "importance"
COMPOSITE_SCORE_ATTR = "composite_score"

FLOW_ATTR = "flow"
DISTANCE_ATTR = "distance"


def build_data(graphml_path: Path) -> tuple[Data, list[str]]:
    """Build a PyG graph from GraphML."""
    graph = nx.read_graphml(graphml_path)
    node_ids = list(graph.nodes())
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}

    node_features = [_build_node_features(graph, node_id) for node_id in node_ids]
    labels = [_build_label(graph, node_id) for node_id in node_ids]
    edge_pairs = [[node_index[source], node_index[target]] for source, target in graph.edges()]
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.float32),
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


def _build_node_features(graph: nx.DiGraph, node_id: str) -> list[float]:
    """Build one node feature vector."""
    octets = [int(part) for part in node_id.split(".")]
    first_octet = float(octets[0])
    second_octet = float(octets[1])
    third_octet = float(octets[2])
    fourth_octet = float(octets[3])

    incoming_flows = [float(edge_data[FLOW_ATTR]) for _, _, edge_data in graph.in_edges(node_id, data=True)]
    outgoing_flows = [float(edge_data[FLOW_ATTR]) for _, _, edge_data in graph.out_edges(node_id, data=True)]
    incoming_sum = sum(incoming_flows)
    outgoing_sum = sum(outgoing_flows)
    total_flow = incoming_sum + outgoing_sum
    in_degree = float(graph.in_degree(node_id))
    out_degree = float(graph.out_degree(node_id))
    total_degree = float(graph.degree(node_id))
    mean_incoming_flow = incoming_sum / max(len(incoming_flows), 1)
    mean_outgoing_flow = outgoing_sum / max(len(outgoing_flows), 1)

    return [
        first_octet,
        second_octet,
        third_octet,
        fourth_octet,
        in_degree,
        out_degree,
        total_degree,
        incoming_sum,
        outgoing_sum,
        total_flow,
        mean_incoming_flow,
        mean_outgoing_flow,
    ]


def _build_label(graph: nx.DiGraph, node_id: str) -> float:
    """Build one node label."""
    return math.log1p(float(graph.nodes[node_id][COMPOSITE_SCORE_ATTR]))

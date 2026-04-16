from __future__ import annotations

import math
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


EPSILON: float = 1e-6

NODE_FEATURES = [
    "importance",
    "pagerank_norm",
    "weighted_betweenness_norm",
    "flow_loss_norm",
    "out_flow_norm",
    "in_flow_norm",
    "random_probability",
]

TARGET_NAME = "composite_score"


@dataclass(frozen=True)
class GraphDataset:
    node_ids: list[str]
    node_id_to_index: dict[str, int]
    group_keys: list[str]
    features: torch.Tensor
    targets: torch.Tensor
    raw_targets: torch.Tensor
    sample_weights: torch.Tensor
    in_neighbors: list[list[tuple[int, float]]]
    out_neighbors: list[list[tuple[int, float]]]


@dataclass(frozen=True)
class SamplerConfig:
    num_hops: int
    max_in_neighbors: int
    max_out_neighbors: int


# ---------------------------------------------------------
# Convert GraphML values to float without extra fallback logic.
# ---------------------------------------------------------
def _safe_float(value: object) -> float:
    return 0.0 if value is None else float(value)


# ---------------------------------------------------------
# Build a stable split key from the IPv4 prefix.
# ---------------------------------------------------------
def _group_key(node_id: str, prefix_len: int) -> str:
    parsed = ip_address(node_id)
    return ".".join(str(parsed).split(".")[:prefix_len]) if parsed.version == 4 else node_id


# ---------------------------------------------------------
# Build the minimal node feature vector selected for training.
# # deg_in, deg_out, in_flow, out_flow
# ---------------------------------------------------------
def _feature_row(node_data: dict[str, object]) -> list[float]:
    return [_safe_float(node_data.get(feature_name)) for feature_name in NODE_FEATURES]


# ---------------------------------------------------------
# Load the directed graph and convert it to training tensors.
# ---------------------------------------------------------
def load_graph_dataset(graphml_path: Path, split_prefix_len: int = 2) -> GraphDataset:
    directed_graph = nx.DiGraph(nx.read_graphml(graphml_path))
    node_ids = list(directed_graph.nodes())
    node_id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    group_keys = [_group_key(node_id, split_prefix_len) for node_id in node_ids]
    in_neighbors: list[list[tuple[int, float]]] = [[] for _ in node_ids]
    out_neighbors: list[list[tuple[int, float]]] = [[] for _ in node_ids]

    for source, target, data in directed_graph.edges(data=True):
        source_index = node_id_to_index[source]
        target_index = node_id_to_index[target]
        flow = _safe_float(data.get("flow"))
        out_neighbors[source_index].append((target_index, flow))
        in_neighbors[target_index].append((source_index, flow))

    # ---------------------------------------------------------
    # Sort neighbors by neighbor total flow for deterministic sampling.
    # ---------------------------------------------------------
    total_flows = [
        sum(flow for _, flow in in_neighbors[index]) + sum(flow for _, flow in out_neighbors[index])
        for index in range(len(node_ids))
    ]
    for neighbors in in_neighbors + out_neighbors:
        neighbors.sort(key=lambda item: total_flows[item[0]], reverse=True)

    features = torch.tensor(
        [_feature_row(directed_graph.nodes[node_id]) for node_id in node_ids],
        dtype=torch.float32,
    )

    raw_targets = torch.tensor(
        [_safe_float(directed_graph.nodes[node_id].get(TARGET_NAME)) for node_id in node_ids],
        dtype=torch.float32,
    )

    return GraphDataset(
        node_ids=node_ids,
        node_id_to_index=node_id_to_index,
        group_keys=group_keys,
        features=features,
        targets=torch.log1p(raw_targets),
        raw_targets=raw_targets,
        sample_weights=torch.ones_like(raw_targets),
        in_neighbors=in_neighbors,
        out_neighbors=out_neighbors,
    )


# ---------------------------------------------------------
# Standardize node features with train split statistics only.
# ---------------------------------------------------------
def standardize_features(features: torch.Tensor, train_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_tensor = features[torch.as_tensor(train_indices, dtype=torch.long)]
    mean = train_tensor.mean(dim=0)
    std = train_tensor.std(dim=0).clamp_min(EPSILON)
    return (features - mean) / std, mean, std


# ---------------------------------------------------------
# Split nodes by subnet groups to reduce leakage across splits.
# ---------------------------------------------------------
def subnet_group_split(
    group_keys: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped_indices: dict[str, list[int]] = {}
    for node_index, group_key in enumerate(group_keys):
        grouped_indices.setdefault(group_key, []).append(node_index)

    groups = list(grouped_indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    train_end = int(len(groups) * train_ratio)
    val_end = train_end + int(len(groups) * val_ratio)
    split_groups = {
        "train": set(groups[:train_end]),
        "val": set(groups[train_end:val_end]),
        "test": set(groups[val_end:]),
    }

    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for group_key, indices in grouped_indices.items():
        split_name = "train" if group_key in split_groups["train"] else "val" if group_key in split_groups["val"] else "test"
        split_indices[split_name].extend(indices)

    return tuple(np.asarray(split_indices[name], dtype=np.int64) for name in ("train", "val", "test"))


def subnet_group_k_fold_split(
    group_keys: list[str],
    num_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped_indices: dict[str, list[int]] = {}
    for node_index, group_key in enumerate(group_keys):
        grouped_indices.setdefault(group_key, []).append(node_index)

    groups = list(grouped_indices.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    fold_groups = np.array_split(groups, num_folds)
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_idx in range(num_folds):
        val_group_set = set(fold_groups[fold_idx])
        train_group_set = set(
            group
            for idx, fold in enumerate(fold_groups)
            if idx != fold_idx
            for group in fold
        )

        train_indices: list[int] = []
        val_indices: list[int] = []

        for group_key, indices in grouped_indices.items():
            if group_key in train_group_set:
                train_indices.extend(indices)
            elif group_key in val_group_set:
                val_indices.extend(indices)

        folds.append((
            np.asarray(train_indices, dtype=np.int64),
            np.asarray(val_indices, dtype=np.int64),
        ))

    return folds


class RootedSubgraphDataset(Dataset[Data]):
    # ---------------------------------------------------------
    # Store graph tensors and rooted subgraph sampling settings.
    # ---------------------------------------------------------
    def __init__(
        self,
        graph_dataset: GraphDataset,
        node_indices: np.ndarray,
        allowed_node_indices: np.ndarray,
        sampler_config: SamplerConfig,
    ) -> None:
        self.graph_dataset = graph_dataset
        self.node_indices = node_indices
        self.allowed_mask = np.zeros(len(graph_dataset.node_ids), dtype=np.bool_)
        self.allowed_mask[allowed_node_indices] = True
        self.sampler_config = sampler_config

    # ---------------------------------------------------------
    # Return the number of root nodes in this split.
    # ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.node_indices)

    # ---------------------------------------------------------
    # Build one rooted subgraph sample for PyG training.
    # ---------------------------------------------------------
    def __getitem__(self, item: int) -> Data:
        root_index = int(self.node_indices[item])
        sampled_nodes = self._sample_nodes(root_index)
        local_index = {node_index: offset for offset, node_index in enumerate(sampled_nodes)}
        edge_pairs = [
            [local_index[source], local_index[target]]
            for source in sampled_nodes
            for target, _ in self.graph_dataset.out_neighbors[source]
            if target in local_index
        ]
        edge_weights = [
            math.log1p(flow)
            for source in sampled_nodes
            for target, flow in self.graph_dataset.out_neighbors[source]
            if target in local_index
        ]

        node_tensor = torch.as_tensor(sampled_nodes, dtype=torch.long)
        edge_index = torch.empty((2, 0), dtype=torch.long) if not edge_pairs else torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_weight = torch.empty((0,), dtype=torch.float32) if not edge_weights else torch.tensor(edge_weights, dtype=torch.float32)
        y = torch.zeros(len(sampled_nodes), dtype=torch.float32)
        y_weight = torch.zeros(len(sampled_nodes), dtype=torch.float32)
        seed_mask = torch.zeros(len(sampled_nodes), dtype=torch.bool)
        y[0] = self.graph_dataset.targets[root_index]
        y_weight[0] = self.graph_dataset.sample_weights[root_index]
        seed_mask[0] = True

        return Data(
            x=self.graph_dataset.features[node_tensor],
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            y_weight=y_weight,
            seed_mask=seed_mask,
            root_node=torch.tensor([root_index], dtype=torch.long),
        )

    # ---------------------------------------------------------
    # Collect the root node and its local in/out neighbors.
    # ---------------------------------------------------------
    def _sample_nodes(self, root_index: int) -> list[int]:
        sampled_nodes = [root_index]
        seen = {root_index}
        frontier = [root_index]

        for _ in range(self.sampler_config.num_hops):
            next_frontier: list[int] = []
            for node_index in frontier:
                for neighbors, limit in (
                    (self.graph_dataset.in_neighbors[node_index], self.sampler_config.max_in_neighbors),
                    (self.graph_dataset.out_neighbors[node_index], self.sampler_config.max_out_neighbors),
                ):
                    picked_neighbors = neighbors if limit <= 0 else neighbors[:limit]
                    for neighbor_index, _ in picked_neighbors:
                        if not self.allowed_mask[neighbor_index] or neighbor_index in seen:
                            continue
                        seen.add(neighbor_index)
                        sampled_nodes.append(neighbor_index)
                        next_frontier.append(neighbor_index)
            if not next_frontier:
                break
            frontier = next_frontier

        return sampled_nodes

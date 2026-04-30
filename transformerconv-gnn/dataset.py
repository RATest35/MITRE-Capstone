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


@dataclass(frozen=True)
class GraphDataset:
    """Store graph tensors and edge-attributed adjacency lists.

    :ivar node_ids: Ordered node identifiers from the graph.
    :ivar node_id_to_index: Mapping from node identifier to tensor index.
    :ivar group_keys: Split group keys for each node.
    :ivar features: Node feature matrix.
    :ivar targets: Log-scaled target values.
    :ivar raw_targets: Original target values from GraphML.
    :ivar sample_weights: Per-node sample weights.
    :ivar in_neighbors: Incoming neighbors, flow, and bytes-per-second values.
    :ivar out_neighbors: Outgoing neighbors, flow, and bytes-per-second values.
    """

    node_ids: list[str]
    node_id_to_index: dict[str, int]
    group_keys: list[str]
    features: torch.Tensor
    targets: torch.Tensor
    raw_targets: torch.Tensor
    sample_weights: torch.Tensor
    in_neighbors: list[list[tuple[int, float, float]]]
    out_neighbors: list[list[tuple[int, float, float]]]


@dataclass(frozen=True)
class SamplerConfig:
    """Configure rooted subgraph neighbor sampling.

    :ivar num_hops: Number of sampling hops from the root node.
    :ivar max_in_neighbors: Maximum incoming neighbors per node.
    :ivar max_out_neighbors: Maximum outgoing neighbors per node.
    """

    num_hops: int
    max_in_neighbors: int
    max_out_neighbors: int


# ---------------------------------------------------------
# Convert GraphML values to float without extra fallback logic.
# ---------------------------------------------------------
def _safe_float(value: object) -> float:
    """Convert a GraphML value to a float.

    :param value: Raw GraphML attribute value.
    :return: Converted floating-point value.
    """
    return 0.0 if value is None else float(value)


# ---------------------------------------------------------
# Build a stable split key from the IPv4 prefix.
# ---------------------------------------------------------
def _group_key(node_id: str, prefix_len: int) -> str:
    """Create a stable subnet key for splitting nodes.

    :param node_id: Node identifier, usually an IP address.
    :param prefix_len: Number of IPv4 octets to keep.
    :return: Split group key for the node.
    """
    parsed = ip_address(node_id)
    return ".".join(str(parsed).split(".")[:prefix_len]) if parsed.version == 4 else node_id


# ---------------------------------------------------------
# Build the minimal node feature vector selected for training.
# # deg_in, deg_out, in_flow, out_flow
# ---------------------------------------------------------
def _feature_row(
    in_neighbors: list[tuple[int, float, float]],
    out_neighbors: list[tuple[int, float, float]],
) -> list[float]:
    """Build log-scaled structural and flow features.

    :param in_neighbors: Incoming neighbors, flow, and bytes-per-second values.
    :param out_neighbors: Outgoing neighbors, flow, and bytes-per-second values.
    :return: Feature row for one node.
    """
    in_degree = float(len(in_neighbors))
    out_degree = float(len(out_neighbors))
    in_flow = float(sum(flow for _, flow, _ in in_neighbors))
    out_flow = float(sum(flow for _, flow, _ in out_neighbors))
    return [
        math.log1p(in_degree),
        math.log1p(out_degree),
        math.log1p(in_flow),
        math.log1p(out_flow),
    ]


# ---------------------------------------------------------
# Build target-aware sample weights for ranking-focused training.
# ---------------------------------------------------------
def _build_sample_weights(
    raw_targets: torch.Tensor,
    sample_weight_max: float,
) -> torch.Tensor:
    """Create target-rank sample weights.

    :param raw_targets: Original target values.
    :param sample_weight_max: Maximum allowed sample weight.
    :return: Per-node sample weights.
    """
    sorted_indices = torch.argsort(raw_targets)
    rank_positions = torch.empty_like(sorted_indices, dtype=torch.float32)
    rank_positions[sorted_indices] = torch.arange(len(raw_targets), dtype=torch.float32)
    rank_percentiles = rank_positions / max(len(raw_targets) - 1, 1)
    scaled_weights = 1.0 + rank_percentiles * (sample_weight_max - 1.0)
    return scaled_weights.clamp(max=sample_weight_max)


# ---------------------------------------------------------
# Load the directed graph and convert it to training tensors.
# ---------------------------------------------------------
def load_graph_dataset(
    graphml_path: Path,
    split_prefix_len: int = 2,
    sample_weight_max: float = 5.0,
) -> GraphDataset:
    """Load GraphML data into tensors and edge-attributed neighbors.

    :param graphml_path: Path to the GraphML file.
    :param split_prefix_len: Number of IPv4 octets used for split groups.
    :param sample_weight_max: Maximum target-rank sample weight.
    :return: Loaded graph dataset.
    """
    directed_graph = nx.DiGraph(nx.read_graphml(graphml_path))
    node_ids = list(directed_graph.nodes())
    node_id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    group_keys = [_group_key(node_id, split_prefix_len) for node_id in node_ids]
    in_neighbors: list[list[tuple[int, float, float]]] = [[] for _ in node_ids]
    out_neighbors: list[list[tuple[int, float, float]]] = [[] for _ in node_ids]

    for source, target, data in directed_graph.edges(data=True):
        source_index = node_id_to_index[source]
        target_index = node_id_to_index[target]
        flow = _safe_float(data.get("flow"))
        bytes_per_sec = _safe_float(data.get("bytes_per_sec"))
        out_neighbors[source_index].append((target_index, flow, bytes_per_sec))
        in_neighbors[target_index].append((source_index, flow, bytes_per_sec))

    # ---------------------------------------------------------
    # Sort neighbors by neighbor total flow for deterministic sampling.
    # ---------------------------------------------------------
    total_flows = [
        sum(flow for _, flow, _ in in_neighbors[index]) + sum(flow for _, flow, _ in out_neighbors[index])
        for index in range(len(node_ids))
    ]
    for neighbors in in_neighbors + out_neighbors:
        neighbors.sort(key=lambda item: total_flows[item[0]], reverse=True)

    features = torch.tensor(
        [_feature_row(in_neighbors[index], out_neighbors[index]) for index in range(len(node_ids))],
        dtype=torch.float32,
    )
    raw_targets = torch.tensor(
        [_safe_float(directed_graph.nodes[node_id].get("composite_score")) for node_id in node_ids],
        dtype=torch.float32,
    )
    sample_weights = _build_sample_weights(raw_targets, sample_weight_max)

    return GraphDataset(
        node_ids=node_ids,
        node_id_to_index=node_id_to_index,
        group_keys=group_keys,
        features=features,
        targets=torch.log1p(raw_targets),
        raw_targets=raw_targets,
        sample_weights=sample_weights,
        in_neighbors=in_neighbors,
        out_neighbors=out_neighbors,
    )


# ---------------------------------------------------------
# Standardize node features with train split statistics only.
# ---------------------------------------------------------
def standardize_features(features: torch.Tensor, train_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize features using train split statistics.

    :param features: Full node feature matrix.
    :param train_indices: Node indices used to compute statistics.
    :return: Standardized features, feature mean, and feature standard deviation.
    """
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
    """Split node indices by subnet group.

    :param group_keys: Split group key for each node.
    :param train_ratio: Ratio of groups assigned to training.
    :param val_ratio: Ratio of groups assigned to validation.
    :param seed: Random seed for group shuffling.
    :return: Train, validation, and test node indices.
    """
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


class RootedSubgraphDataset(Dataset[Data]):
    """Create PyG rooted subgraph samples with edge attributes.

    :param graph_dataset: Full graph dataset.
    :param node_indices: Root node indices exposed by this dataset.
    :param allowed_node_indices: Node indices allowed during sampling.
    :param sampler_config: Rooted subgraph sampling configuration.
    """

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
        """Store graph data and sampling constraints.

        :param graph_dataset: Full graph dataset.
        :param node_indices: Root node indices exposed by this dataset.
        :param allowed_node_indices: Node indices allowed during sampling.
        :param sampler_config: Rooted subgraph sampling configuration.
        :return: None.
        """
        self.graph_dataset = graph_dataset
        self.node_indices = node_indices
        self.allowed_mask = np.zeros(len(graph_dataset.node_ids), dtype=np.bool_)
        self.allowed_mask[allowed_node_indices] = True
        self.sampler_config = sampler_config

    # ---------------------------------------------------------
    # Return the number of root nodes in this split.
    # ---------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of root nodes.

        :return: Number of root nodes.
        """
        return len(self.node_indices)

    # ---------------------------------------------------------
    # Build one rooted subgraph sample for PyG training.
    # ---------------------------------------------------------
    def __getitem__(self, item: int) -> Data:
        """Build one rooted subgraph sample.

        :param item: Position of the root node in this dataset.
        :return: PyG data object for the sampled rooted subgraph.
        """
        root_index = int(self.node_indices[item])
        sampled_nodes = self._sample_nodes(root_index)
        local_index = {node_index: offset for offset, node_index in enumerate(sampled_nodes)}
        edge_pairs = [
            [local_index[source], local_index[target]]
            for source in sampled_nodes
            for target, _, _ in self.graph_dataset.out_neighbors[source]
            if target in local_index
        ]
        edge_features = [
            [math.log1p(flow), math.log1p(bytes_per_sec)]
            for source in sampled_nodes
            for target, flow, bytes_per_sec in self.graph_dataset.out_neighbors[source]
            if target in local_index
        ]

        node_tensor = torch.as_tensor(sampled_nodes, dtype=torch.long)
        edge_index = torch.empty((2, 0), dtype=torch.long) if not edge_pairs else torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = (
            torch.empty((0, 2), dtype=torch.float32)
            if not edge_features
            else torch.tensor(edge_features, dtype=torch.float32)
        )
        y = torch.zeros(len(sampled_nodes), dtype=torch.float32)
        y_weight = torch.zeros(len(sampled_nodes), dtype=torch.float32)
        seed_mask = torch.zeros(len(sampled_nodes), dtype=torch.bool)
        y[0] = self.graph_dataset.targets[root_index]
        y_weight[0] = self.graph_dataset.sample_weights[root_index]
        seed_mask[0] = True

        return Data(
            x=self.graph_dataset.features[node_tensor],
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            y_weight=y_weight,
            seed_mask=seed_mask,
            root_node=torch.tensor([root_index], dtype=torch.long),
        )

    # ---------------------------------------------------------
    # Collect the root node and its local in/out neighbors.
    # ---------------------------------------------------------
    def _sample_nodes(self, root_index: int) -> list[int]:
        """Sample local in and out neighbors around a root node.

        :param root_index: Root node index in the full graph.
        :return: Sampled node indices with the root node first.
        """
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
                    for neighbor_index, _, _ in picked_neighbors:
                        if not self.allowed_mask[neighbor_index] or neighbor_index in seen:
                            continue
                        seen.add(neighbor_index)
                        sampled_nodes.append(neighbor_index)
                        next_frontier.append(neighbor_index)
            if not next_frontier:
                break
            frontier = next_frontier

        return sampled_nodes

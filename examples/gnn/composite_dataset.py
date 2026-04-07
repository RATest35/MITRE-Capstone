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
DEFAULT_BASIC_GROUPS: tuple[str, ...] = ("node_flow", "node_degree", "flow_balance", "ip")
DEFAULT_EXTENDED_GROUPS: tuple[str, ...] = DEFAULT_BASIC_GROUPS + (
    "neighbor_flow",
    "neighbor_degree",
    "neighbor_total_flow",
    "two_hop",
    "hub_ratio",
)


@dataclass(frozen=True)
class GraphDataset:
    """Container for graph tensors and adjacency lists."""

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
    """Settings for rooted subgraph sampling."""

    num_hops: int
    max_in_neighbors: int
    max_out_neighbors: int


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convert a value to float."""
    return default if value is None else float(value)


def _mean(values: list[float]) -> float:
    """Return list mean."""
    return float(sum(values)) / max(float(len(values)), 1.0)


def _quantile(values: list[float], q: float) -> float:
    """Return one quantile."""
    return 0.0 if not values else float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _summary(values: list[float], include_total: bool) -> list[float]:
    """Summarize a float list."""
    if not values:
        size = 4 if include_total else 3
        return [0.0] * size

    summary = [_mean(values), float(max(values)), _quantile(values, 0.75)]
    if include_total:
        summary.insert(0, float(sum(values)))
    return [math.log1p(value) for value in summary]


def _parse_ip_features(node_id: str) -> list[float]:
    """Build numeric IP features."""
    try:
        parsed = ip_address(node_id)
    except ValueError:
        return [0.0] * 9

    if parsed.version != 4:
        return [0.0] * 9

    octets = node_id.split(".")
    first_octet = int(octets[0])
    second_octet = int(octets[1])
    return [
        *(int(part) / 255.0 for part in octets),
        float(parsed.is_private),
        float(not parsed.is_private),
        float(node_id.startswith("10.")),
        float(first_octet == 172 and 16 <= second_octet <= 31),
        float(node_id.startswith("192.168.")),
    ]


def _group_key(node_id: str, prefix_len: int) -> str:
    """Return an IPv4 prefix grouping key."""
    try:
        parsed = ip_address(node_id)
    except ValueError:
        return node_id

    if parsed.version != 4:
        return node_id
    return ".".join(node_id.split(".")[:prefix_len])


def _two_hop_counts(neighbor_indices: list[int], adjacency: list[list[tuple[int, float]]]) -> set[int]:
    """Collect two-hop neighbor indices."""
    return {index for neighbor_index in neighbor_indices for index, _ in adjacency[neighbor_index]}


def _extended_feature_row(
    index: int,
    in_neighbors: list[list[tuple[int, float]]],
    out_neighbors: list[list[tuple[int, float]]],
) -> list[float]:
    """Build extended structural features."""
    incoming = in_neighbors[index]
    outgoing = out_neighbors[index]
    incoming_flows = [flow for _, flow in incoming]
    outgoing_flows = [flow for _, flow in outgoing]
    in_neighbor_indices = [neighbor_index for neighbor_index, _ in incoming]
    out_neighbor_indices = [neighbor_index for neighbor_index, _ in outgoing]

    neighbor_degree = lambda node_index: float(len(in_neighbors[node_index]) + len(out_neighbors[node_index]))
    neighbor_flow = lambda node_index: float(
        sum(flow for _, flow in in_neighbors[node_index]) + sum(flow for _, flow in out_neighbors[node_index])
    )

    in_neighbor_degrees = [neighbor_degree(node_index) for node_index in in_neighbor_indices]
    out_neighbor_degrees = [neighbor_degree(node_index) for node_index in out_neighbor_indices]
    in_neighbor_flows = [neighbor_flow(node_index) for node_index in in_neighbor_indices]
    out_neighbor_flows = [neighbor_flow(node_index) for node_index in out_neighbor_indices]

    in_two_hop = _two_hop_counts(in_neighbor_indices, in_neighbors) - {index}
    out_two_hop = _two_hop_counts(out_neighbor_indices, out_neighbors) - {index}
    union_two_hop = in_two_hop | out_two_hop
    overlap_two_hop = in_two_hop & out_two_hop

    in_degree = float(len(incoming))
    out_degree = float(len(outgoing))
    in_flow = float(sum(incoming_flows))
    out_flow = float(sum(outgoing_flows))

    return [
        *_summary(incoming_flows, include_total=True),
        *_summary(outgoing_flows, include_total=True),
        *_summary(in_neighbor_degrees, include_total=False),
        *_summary(out_neighbor_degrees, include_total=False),
        *_summary(in_neighbor_flows, include_total=True)[1:],
        *_summary(out_neighbor_flows, include_total=True)[1:],
        math.log1p(float(len(in_two_hop))),
        math.log1p(float(len(out_two_hop))),
        math.log1p(float(len(union_two_hop))),
        math.log1p(float(len(overlap_two_hop))),
        math.log1p(in_degree / max(_mean(in_neighbor_degrees), EPSILON)),
        math.log1p(out_degree / max(_mean(out_neighbor_degrees), EPSILON)),
        math.log1p(in_flow / max(_mean(in_neighbor_flows), EPSILON)),
        math.log1p(out_flow / max(_mean(out_neighbor_flows), EPSILON)),
    ]


def resolve_feature_groups(feature_set: str, feature_groups: tuple[str, ...] | None) -> tuple[str, ...]:
    """Resolve active feature groups."""
    default_groups = DEFAULT_EXTENDED_GROUPS if feature_set == "extended" else DEFAULT_BASIC_GROUPS
    if feature_groups is None:
        return default_groups

    invalid_groups = [group_name for group_name in feature_groups if group_name not in set(default_groups)]
    if invalid_groups:
        raise ValueError(f"Invalid feature groups for {feature_set}: {invalid_groups}")
    return feature_groups


def _feature_row(
    index: int,
    node_id: str,
    in_neighbors: list[list[tuple[int, float]]],
    out_neighbors: list[list[tuple[int, float]]],
    feature_set: str,
    feature_groups: tuple[str, ...],
) -> list[float]:
    """Build features for one node."""
    incoming = in_neighbors[index]
    outgoing = out_neighbors[index]
    incoming_flows = [flow for _, flow in incoming]
    outgoing_flows = [flow for _, flow in outgoing]

    in_degree = float(len(incoming))
    out_degree = float(len(outgoing))
    total_degree = in_degree + out_degree
    in_flow = float(sum(incoming_flows))
    out_flow = float(sum(outgoing_flows))
    total_flow = in_flow + out_flow

    feature_map = {
        "node_flow": [
            math.log1p(in_flow),
            math.log1p(out_flow),
            math.log1p(total_flow),
            math.log1p(abs(in_flow - out_flow)),
            math.log1p(in_flow / max(out_flow, EPSILON)),
            math.log1p(in_flow / max(in_degree, 1.0)),
            math.log1p(out_flow / max(out_degree, 1.0)),
            math.log1p(max(incoming_flows, default=0.0)),
            math.log1p(max(outgoing_flows, default=0.0)),
        ],
        "node_degree": [math.log1p(in_degree), math.log1p(out_degree), math.log1p(total_degree)],
        "flow_balance": [in_flow / max(total_flow, EPSILON), out_flow / max(total_flow, EPSILON)],
        "ip": _parse_ip_features(node_id),
    }
    if feature_set == "extended":
        extended_values = _extended_feature_row(index, in_neighbors, out_neighbors)
        feature_map.update(
            {
                "neighbor_flow": extended_values[:8],
                "neighbor_degree": extended_values[8:14],
                "neighbor_total_flow": extended_values[14:20],
                "two_hop": extended_values[20:24],
                "hub_ratio": extended_values[24:28],
            }
        )
    return [value for group_name in feature_groups for value in feature_map[group_name]]


def _build_feature_matrix(
    node_ids: list[str],
    in_neighbors: list[list[tuple[int, float]]],
    out_neighbors: list[list[tuple[int, float]]],
    feature_set: str,
    feature_groups: tuple[str, ...],
) -> torch.Tensor:
    """Build node-level features."""
    rows = [
        _feature_row(index, node_id, in_neighbors, out_neighbors, feature_set, feature_groups)
        for index, node_id in enumerate(node_ids)
    ]
    return torch.tensor(rows, dtype=torch.float32)


def _build_sample_weights(raw_targets: torch.Tensor, mode: str, scale: float) -> torch.Tensor:
    """Build sample weights from target rank."""
    normalized_rank = torch.argsort(torch.argsort(raw_targets)).float() / max(len(raw_targets) - 1, 1)
    if mode == "linear":
        return 1.0 + scale * normalized_rank
    if mode == "sqrt":
        return 1.0 + scale * normalized_rank.sqrt()
    if mode == "quadratic":
        return 1.0 + scale * normalized_rank.square()
    raise ValueError(f"Unsupported weight mode: {mode}")


def load_graph_dataset(
    graphml_path: Path,
    feature_set: str,
    weight_mode: str,
    weight_scale: float,
    split_prefix_len: int,
    feature_groups: tuple[str, ...] | None = None,
) -> GraphDataset:
    """Load a GraphML file and build graph tensors."""
    directed_graph = nx.DiGraph(nx.read_graphml(graphml_path))
    node_ids = list(directed_graph.nodes())
    node_id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    group_keys = [_group_key(node_id, split_prefix_len) for node_id in node_ids]
    in_neighbors: list[list[tuple[int, float]]] = [[] for _ in node_ids]
    out_neighbors: list[list[tuple[int, float]]] = [[] for _ in node_ids]

    raw_target_tensor = torch.tensor(
        [_safe_float(directed_graph.nodes[node_id].get("composite_score")) for node_id in node_ids],
        dtype=torch.float32,
    )

    for source, target, data in directed_graph.edges(data=True):
        source_index = node_id_to_index[source]
        target_index = node_id_to_index[target]
        flow = _safe_float(data.get("flow"))
        out_neighbors[source_index].append((target_index, flow))
        in_neighbors[target_index].append((source_index, flow))

    for neighbors in in_neighbors + out_neighbors:
        neighbors.sort(key=lambda item: item[1], reverse=True)

    active_feature_groups = resolve_feature_groups(feature_set, feature_groups)
    return GraphDataset(
        node_ids=node_ids,
        node_id_to_index=node_id_to_index,
        group_keys=group_keys,
        features=_build_feature_matrix(node_ids, in_neighbors, out_neighbors, feature_set, active_feature_groups),
        targets=torch.log1p(raw_target_tensor),
        raw_targets=raw_target_tensor,
        sample_weights=_build_sample_weights(raw_target_tensor, weight_mode, weight_scale),
        in_neighbors=in_neighbors,
        out_neighbors=out_neighbors,
    )


def subnet_group_split(
    group_keys: list[str],
    targets: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    bucket_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split nodes by group while preserving score distribution."""
    grouped_indices: dict[str, list[int]] = {}
    for node_index, group_key in enumerate(group_keys):
        grouped_indices.setdefault(group_key, []).append(node_index)

    groups = list(grouped_indices)
    group_scores = np.array(
        [float(targets[torch.as_tensor(grouped_indices[group_key], dtype=torch.long)].mean().item()) for group_key in groups],
        dtype=np.float64,
    )
    ordered_groups = [groups[position] for position in np.argsort(group_scores)]
    rng = np.random.default_rng(seed)
    group_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for start in range(0, len(ordered_groups), bucket_size):
        bucket = ordered_groups[start:start + bucket_size].copy()
        rng.shuffle(bucket)
        train_end = int(len(bucket) * train_ratio)
        val_end = train_end + int(len(bucket) * val_ratio)
        group_split["train"].extend(bucket[:train_end])
        group_split["val"].extend(bucket[train_end:val_end])
        group_split["test"].extend(bucket[val_end:])

    group_sets = {name: set(values) for name, values in group_split.items()}
    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for group_key, indices in grouped_indices.items():
        split_name = "train" if group_key in group_sets["train"] else "val" if group_key in group_sets["val"] else "test"
        split_indices[split_name].extend(indices)

    return tuple(np.asarray(split_indices[name], dtype=np.int64) for name in ("train", "val", "test"))


def standardize_features(features: torch.Tensor, train_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize features with train-only statistics."""
    train_tensor = features[torch.as_tensor(train_indices, dtype=torch.long)]
    mean = train_tensor.mean(dim=0)
    std = train_tensor.std(dim=0).clamp_min(EPSILON)
    return (features - mean) / std, mean, std


class RootedSubgraphDataset(Dataset[Data]):
    """Dataset that samples a rooted directed subgraph per node."""

    def __init__(
        self,
        graph_dataset: GraphDataset,
        node_indices: np.ndarray,
        allowed_node_indices: np.ndarray,
        sampler_config: SamplerConfig,
        training: bool,
        seed: int,
    ) -> None:
        self.graph_dataset = graph_dataset
        self.node_indices = node_indices
        self.allowed_mask = np.zeros(len(graph_dataset.node_ids), dtype=np.bool_)
        self.allowed_mask[allowed_node_indices] = True
        self.sampler_config = sampler_config
        self.training = training
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.node_indices)

    def __getitem__(self, item: int) -> Data:
        """Sample a rooted subgraph."""
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
        seed_mask[0] = True
        y[0] = self.graph_dataset.targets[root_index]
        y_weight[0] = self.graph_dataset.sample_weights[root_index]

        return Data(
            x=self.graph_dataset.features[node_tensor],
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            y_weight=y_weight,
            seed_mask=seed_mask,
            root_node=torch.tensor([root_index], dtype=torch.long),
        )

    def _sample_nodes(self, root_index: int) -> list[int]:
        """Collect nodes around a root node."""
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
                    for neighbor_index in self._pick_neighbors(neighbors, limit):
                        if neighbor_index in seen:
                            continue
                        seen.add(neighbor_index)
                        sampled_nodes.append(neighbor_index)
                        next_frontier.append(neighbor_index)
            if not next_frontier:
                break
            frontier = next_frontier

        return sampled_nodes

    def _pick_neighbors(self, neighbors: list[tuple[int, float]], limit: int) -> list[int]:
        """Select neighbor indices for one hop."""
        allowed_neighbors = [(neighbor_index, flow) for neighbor_index, flow in neighbors if self.allowed_mask[neighbor_index]]
        if limit <= 0 or not allowed_neighbors:
            return []
        if len(allowed_neighbors) <= limit or not self.training:
            return [neighbor_index for neighbor_index, _ in allowed_neighbors[:limit]]

        weights = np.array([max(flow, 0.0) + 1.0 for _, flow in allowed_neighbors], dtype=np.float64)
        positions = self.rng.choice(len(allowed_neighbors), size=limit, replace=False, p=weights / weights.sum())
        positions.sort()
        return [allowed_neighbors[position][0] for position in positions]

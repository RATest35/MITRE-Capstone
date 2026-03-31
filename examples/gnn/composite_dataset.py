from __future__ import annotations

import math
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Iterator

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
    if value is None:
        return default
    return float(value)


def _parse_ip_features(node_id: str) -> list[float]:
    """Build numeric IP features."""
    try:
        parsed = ip_address(node_id)
    except ValueError:
        return [0.0] * 9

    if parsed.version != 4:
        return [0.0] * 9

    octets = [int(part) / 255.0 for part in node_id.split(".")]
    is_private = float(parsed.is_private)
    is_public = 1.0 - is_private
    is_ten = float(node_id.startswith("10."))
    is_172 = float(172 == int(node_id.split(".")[0]) and 16 <= int(node_id.split(".")[1]) <= 31)
    is_192 = float(node_id.startswith("192.168."))
    return octets + [is_private, is_public, is_ten, is_172, is_192]


def _group_key(node_id: str, prefix_len: int) -> str:
    """Return an IPv4 prefix grouping key."""
    try:
        parsed = ip_address(node_id)
    except ValueError:
        return node_id

    if parsed.version != 4:
        return node_id

    octets = node_id.split(".")
    return ".".join(octets[:prefix_len])


def build_group_keys(node_ids: list[str], prefix_len: int) -> list[str]:
    """Build group keys for all nodes."""
    return [_group_key(node_id, prefix_len) for node_id in node_ids]


def _sort_neighbors(neighbors: list[list[tuple[int, float]]]) -> None:
    """Sort neighbors by flow descending."""
    for values in neighbors:
        values.sort(key=lambda item: item[1], reverse=True)


def _quantile(values: list[float], q: float) -> float:
    """Return one quantile from a float list."""
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _log_summary(values: list[float]) -> list[float]:
    """Summarize a float list with log-scaled stats."""
    if not values:
        return [0.0, 0.0, 0.0, 0.0]

    total = float(sum(values))
    mean = total / float(len(values))
    maximum = float(max(values))
    upper_quartile = _quantile(values, 0.75)
    return [
        math.log1p(total),
        math.log1p(mean),
        math.log1p(maximum),
        math.log1p(upper_quartile),
    ]


def _raw_summary(values: list[float]) -> list[float]:
    """Summarize a float list with raw stats."""
    if not values:
        return [0.0, 0.0, 0.0]

    mean = float(sum(values)) / float(len(values))
    maximum = float(max(values))
    upper_quartile = _quantile(values, 0.75)
    return [
        math.log1p(mean),
        math.log1p(maximum),
        math.log1p(upper_quartile),
    ]


def _two_hop_counts(
    neighbor_indices: list[int],
    adjacency: list[list[tuple[int, float]]],
) -> set[int]:
    """Collect two-hop neighbor indices."""
    results: set[int] = set()
    for neighbor_index in neighbor_indices:
        results.update(index for index, _ in adjacency[neighbor_index])
    return results


def _extended_feature_row(
    index: int,
    in_neighbors: list[list[tuple[int, float]]],
    out_neighbors: list[list[tuple[int, float]]],
) -> list[float]:
    """Build extended structural features for one node."""
    incoming = in_neighbors[index]
    outgoing = out_neighbors[index]
    in_neighbor_indices = [neighbor_index for neighbor_index, _ in incoming]
    out_neighbor_indices = [neighbor_index for neighbor_index, _ in outgoing]
    incoming_flows = [flow for _, flow in incoming]
    outgoing_flows = [flow for _, flow in outgoing]

    in_neighbor_total_degree = [
        float(len(in_neighbors[neighbor_index]) + len(out_neighbors[neighbor_index]))
        for neighbor_index in in_neighbor_indices
    ]
    out_neighbor_total_degree = [
        float(len(in_neighbors[neighbor_index]) + len(out_neighbors[neighbor_index]))
        for neighbor_index in out_neighbor_indices
    ]
    in_neighbor_total_flow = [
        float(sum(flow for _, flow in in_neighbors[neighbor_index]) + sum(flow for _, flow in out_neighbors[neighbor_index]))
        for neighbor_index in in_neighbor_indices
    ]
    out_neighbor_total_flow = [
        float(sum(flow for _, flow in in_neighbors[neighbor_index]) + sum(flow for _, flow in out_neighbors[neighbor_index]))
        for neighbor_index in out_neighbor_indices
    ]

    in_two_hop = _two_hop_counts(in_neighbor_indices, in_neighbors)
    out_two_hop = _two_hop_counts(out_neighbor_indices, out_neighbors)
    in_two_hop.discard(index)
    out_two_hop.discard(index)
    union_two_hop = in_two_hop | out_two_hop
    overlap_two_hop = in_two_hop & out_two_hop

    in_degree = float(len(incoming))
    out_degree = float(len(outgoing))
    in_flow = float(sum(incoming_flows))
    out_flow = float(sum(outgoing_flows))
    in_neighbor_degree_mean = float(sum(in_neighbor_total_degree)) / max(float(len(in_neighbor_total_degree)), 1.0)
    out_neighbor_degree_mean = float(sum(out_neighbor_total_degree)) / max(float(len(out_neighbor_total_degree)), 1.0)
    in_neighbor_flow_mean = float(sum(in_neighbor_total_flow)) / max(float(len(in_neighbor_total_flow)), 1.0)
    out_neighbor_flow_mean = float(sum(out_neighbor_total_flow)) / max(float(len(out_neighbor_total_flow)), 1.0)

    return [
        *_log_summary(incoming_flows),
        *_log_summary(outgoing_flows),
        *_raw_summary(in_neighbor_total_degree),
        *_raw_summary(out_neighbor_total_degree),
        *_log_summary(in_neighbor_total_flow)[1:],
        *_log_summary(out_neighbor_total_flow)[1:],
        math.log1p(float(len(in_two_hop))),
        math.log1p(float(len(out_two_hop))),
        math.log1p(float(len(union_two_hop))),
        math.log1p(float(len(overlap_two_hop))),
        math.log1p(in_degree / max(in_neighbor_degree_mean, EPSILON)),
        math.log1p(out_degree / max(out_neighbor_degree_mean, EPSILON)),
        math.log1p(in_flow / max(in_neighbor_flow_mean, EPSILON)),
        math.log1p(out_flow / max(out_neighbor_flow_mean, EPSILON)),
    ]


def _build_feature_matrix(
    node_ids: list[str],
    in_neighbors: list[list[tuple[int, float]]],
    out_neighbors: list[list[tuple[int, float]]],
    feature_set: str,
    feature_groups: tuple[str, ...],
) -> torch.Tensor:
    """Build node-level features from graph structure."""
    rows: list[list[float]] = []
    for index, node_id in enumerate(node_ids):
        incoming = in_neighbors[index]
        outgoing = out_neighbors[index]

        in_degree = float(len(incoming))
        out_degree = float(len(outgoing))
        total_degree = in_degree + out_degree

        in_flow = sum(flow for _, flow in incoming)
        out_flow = sum(flow for _, flow in outgoing)
        total_flow = in_flow + out_flow

        avg_in_flow = in_flow / max(in_degree, 1.0)
        avg_out_flow = out_flow / max(out_degree, 1.0)
        max_in_flow = max([flow for _, flow in incoming], default=0.0)
        max_out_flow = max([flow for _, flow in outgoing], default=0.0)
        abs_gap = abs(in_flow - out_flow)
        flow_ratio = in_flow / max(out_flow, EPSILON)
        inbound_share = in_flow / max(total_flow, EPSILON)
        outbound_share = out_flow / max(total_flow, EPSILON)

        feature_map = {
            "node_flow": [
                math.log1p(in_flow),
                math.log1p(out_flow),
                math.log1p(total_flow),
                math.log1p(abs_gap),
                math.log1p(flow_ratio),
                math.log1p(avg_in_flow),
                math.log1p(avg_out_flow),
                math.log1p(max_in_flow),
                math.log1p(max_out_flow),
            ],
            "node_degree": [
                math.log1p(in_degree),
                math.log1p(out_degree),
                math.log1p(total_degree),
            ],
            "flow_balance": [inbound_share, outbound_share],
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
        row = [value for group_name in feature_groups for value in feature_map[group_name]]
        rows.append(row)

    return torch.tensor(rows, dtype=torch.float32)


def resolve_feature_groups(feature_set: str, feature_groups: tuple[str, ...] | None) -> tuple[str, ...]:
    """Resolve active feature groups."""
    default_groups = DEFAULT_EXTENDED_GROUPS if feature_set == "extended" else DEFAULT_BASIC_GROUPS
    if feature_groups is None:
        return default_groups

    allowed_groups = set(default_groups)
    invalid_groups = [group_name for group_name in feature_groups if group_name not in allowed_groups]
    if invalid_groups:
        raise ValueError(f"Invalid feature groups for {feature_set}: {invalid_groups}")
    return feature_groups


def _build_sample_weights(raw_targets: torch.Tensor, mode: str, scale: float) -> torch.Tensor:
    """Build sample weights from target rank."""
    normalized_rank = torch.argsort(torch.argsort(raw_targets)).float() / max(len(raw_targets) - 1, 1)
    if mode == "linear":
        factor = normalized_rank
    elif mode == "sqrt":
        factor = normalized_rank.sqrt()
    elif mode == "quadratic":
        factor = normalized_rank.square()
    else:
        raise ValueError(f"Unsupported weight mode: {mode}")
    return 1.0 + scale * factor


def load_graph_dataset(
    graphml_path: Path,
    feature_set: str,
    weight_mode: str,
    weight_scale: float,
    split_prefix_len: int,
    feature_groups: tuple[str, ...] | None = None,
) -> GraphDataset:
    """Load a GraphML file and build graph tensors."""
    graph = nx.read_graphml(graphml_path)
    directed_graph = nx.DiGraph(graph)

    node_ids = list(directed_graph.nodes())
    node_id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    group_keys = build_group_keys(node_ids, split_prefix_len)

    in_neighbors: list[list[tuple[int, float]]] = [[] for _ in node_ids]
    out_neighbors: list[list[tuple[int, float]]] = [[] for _ in node_ids]
    raw_targets: list[float] = []

    for node_id in node_ids:
        raw_targets.append(_safe_float(directed_graph.nodes[node_id].get("composite_score"), 0.0))

    for source, target, data in directed_graph.edges(data=True):
        source_index = node_id_to_index[source]
        target_index = node_id_to_index[target]
        flow = _safe_float(data.get("flow"), 0.0)
        out_neighbors[source_index].append((target_index, flow))
        in_neighbors[target_index].append((source_index, flow))

    _sort_neighbors(in_neighbors)
    _sort_neighbors(out_neighbors)

    active_feature_groups = resolve_feature_groups(feature_set, feature_groups)
    features = _build_feature_matrix(node_ids, in_neighbors, out_neighbors, feature_set, active_feature_groups)
    raw_target_tensor = torch.tensor(raw_targets, dtype=torch.float32)
    targets = torch.log1p(raw_target_tensor)
    sample_weights = _build_sample_weights(raw_target_tensor, weight_mode, weight_scale)

    return GraphDataset(
        node_ids=node_ids,
        node_id_to_index=node_id_to_index,
        group_keys=group_keys,
        features=features,
        targets=targets,
        raw_targets=raw_target_tensor,
        sample_weights=sample_weights,
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
        [
            float(targets[torch.as_tensor(grouped_indices[group_key], dtype=torch.long)].mean().item())
            for group_key in groups
        ],
        dtype=np.float64,
    )
    order = np.argsort(group_scores)
    rng = np.random.default_rng(seed)

    train_groups: set[str] = set()
    val_groups: set[str] = set()
    test_groups: set[str] = set()

    ordered_groups = [groups[position] for position in order]
    for start in range(0, len(ordered_groups), bucket_size):
        bucket = ordered_groups[start:start + bucket_size].copy()
        rng.shuffle(bucket)
        train_end = int(len(bucket) * train_ratio)
        val_end = train_end + int(len(bucket) * val_ratio)
        train_groups.update(bucket[:train_end])
        val_groups.update(bucket[train_end:val_end])
        test_groups.update(bucket[val_end:])

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for group_key, group_node_indices in grouped_indices.items():
        if group_key in train_groups:
            train_indices.extend(group_node_indices)
            continue
        if group_key in val_groups:
            val_indices.extend(group_node_indices)
            continue
        test_indices.extend(group_node_indices)

    return (
        np.array(train_indices, dtype=np.int64),
        np.array(val_indices, dtype=np.int64),
        np.array(test_indices, dtype=np.int64),
    )


def standardize_features(features: torch.Tensor, train_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize features with train-only statistics."""
    train_tensor = features[torch.as_tensor(train_indices, dtype=torch.long)]
    mean = train_tensor.mean(dim=0)
    std = train_tensor.std(dim=0).clamp_min(EPSILON)
    standardized = (features - mean) / std
    return standardized, mean, std


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

        edge_pairs: list[list[int]] = []
        edge_weights: list[float] = []

        for source in sampled_nodes:
            for target, flow in self.graph_dataset.out_neighbors[source]:
                if target not in local_index:
                    continue
                edge_pairs.append([local_index[source], local_index[target]])
                edge_weights.append(math.log1p(flow))

        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)

        x = self.graph_dataset.features[torch.as_tensor(sampled_nodes, dtype=torch.long)]
        y = torch.zeros(len(sampled_nodes), dtype=torch.float32)
        raw_y = torch.zeros(len(sampled_nodes), dtype=torch.float32)
        y_weight = torch.zeros(len(sampled_nodes), dtype=torch.float32)
        seed_mask = torch.zeros(len(sampled_nodes), dtype=torch.bool)
        seed_mask[0] = True
        y[0] = self.graph_dataset.targets[root_index]
        raw_y[0] = self.graph_dataset.raw_targets[root_index]
        y_weight[0] = self.graph_dataset.sample_weights[root_index]

        return Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            raw_y=raw_y,
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
                for neighbor_index in self._pick_neighbors(
                    self.graph_dataset.in_neighbors[node_index],
                    self.sampler_config.max_in_neighbors,
                ):
                    if neighbor_index in seen:
                        continue
                    seen.add(neighbor_index)
                    sampled_nodes.append(neighbor_index)
                    next_frontier.append(neighbor_index)

                for neighbor_index in self._pick_neighbors(
                    self.graph_dataset.out_neighbors[node_index],
                    self.sampler_config.max_out_neighbors,
                ):
                    if neighbor_index in seen:
                        continue
                    seen.add(neighbor_index)
                    sampled_nodes.append(neighbor_index)
                    next_frontier.append(neighbor_index)

            if not next_frontier:
                break
            frontier = next_frontier

        return sampled_nodes

    def _pick_neighbors(self, neighbors: list[tuple[int, float]], limit: int) -> Iterator[int]:
        """Select neighbor indices for one hop."""
        if limit <= 0 or not neighbors:
            return iter(())

        allowed_neighbors = [
            (neighbor_index, flow)
            for neighbor_index, flow in neighbors
            if self.allowed_mask[neighbor_index]
        ]

        if len(allowed_neighbors) <= limit:
            return (neighbor_index for neighbor_index, _ in allowed_neighbors)

        if not self.training:
            return (neighbor_index for neighbor_index, _ in allowed_neighbors[:limit])

        weights = np.array([max(flow, 0.0) + 1.0 for _, flow in allowed_neighbors], dtype=np.float64)
        probabilities = weights / weights.sum()
        sampled_positions = self.rng.choice(len(allowed_neighbors), size=limit, replace=False, p=probabilities)
        sampled_positions.sort()
        return (allowed_neighbors[position][0] for position in sampled_positions)

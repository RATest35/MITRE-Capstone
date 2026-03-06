"""Train a GNN to predict node flow loss from the IP graph dataset."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv
import math
import random

import networkx as nx
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


GRAPHML_PATH: Path = Path(__file__).with_name("ip_graph_39_nodes_243_edges_with_flow.graphml")
PREDICTION_CSV_PATH: Path = Path(__file__).with_name("gnn_predictions.csv")
RANDOM_SEEDS: tuple[int, ...] = (7, 21, 84, 168, 336)
TRAIN_RATIO: float = 0.6
VAL_RATIO: float = 0.2
TEST_RATIO: float = 0.2
HIDDEN_CHANNELS: int = 32
NUM_EPOCHS: int = 400
LEARNING_RATE: float = 0.01
WEIGHT_DECAY: float = 1e-4
PATIENCE: int = 40
DROPOUT: float = 0.1


class FlowLossGNN(nn.Module):
    """Predict log flow loss for each node."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(input_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, data: Data) -> Tensor:
        """Run message passing and return node predictions."""
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        return self.head(x).squeeze(-1)


def build_data(graphml_path: Path) -> tuple[Data, list[str]]:
    """Load the GraphML file and convert it into a PyG graph."""
    graph = nx.read_graphml(graphml_path)
    node_ids: list[str] = list(graph.nodes())
    node_index: dict[str, int] = {node_id: index for index, node_id in enumerate(node_ids)}

    node_features: list[list[float]] = []
    labels: list[float] = []

    for node_id in node_ids:
        octets = [int(part) for part in node_id.split(".")]
        incoming_flows = [float(edge_data["flow"]) for _, _, edge_data in graph.in_edges(node_id, data=True)]
        outgoing_flows = [float(edge_data["flow"]) for _, _, edge_data in graph.out_edges(node_id, data=True)]
        total_flow = sum(incoming_flows) + sum(outgoing_flows)

        node_features.append(
            [
                float(octets[0]),
                float(octets[1]),
                float(octets[2]),
                float(octets[3]),
                float(graph.in_degree(node_id)),
                float(graph.out_degree(node_id)),
                float(graph.degree(node_id)),
                sum(incoming_flows),
                sum(outgoing_flows),
                total_flow,
                sum(incoming_flows) / max(len(incoming_flows), 1),
                sum(outgoing_flows) / max(len(outgoing_flows), 1),
            ]
        )
        labels.append(math.log1p(float(graph.nodes[node_id]["flow_loss"])))

    edge_index = torch.tensor(
        [[node_index[source], node_index[target]] for source, target in graph.edges()],
        dtype=torch.long,
    ).t().contiguous()

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.float32),
    )
    return data, node_ids


def build_masks(num_nodes: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    """Create train, validation, and test masks."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_nodes, generator=generator)
    train_end = int(num_nodes * TRAIN_RATIO)
    val_end = train_end + int(num_nodes * VAL_RATIO)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[order[:train_end]] = True
    val_mask[order[train_end:val_end]] = True
    test_mask[order[val_end:]] = True
    return train_mask, val_mask, test_mask


def standardize_features(features: Tensor, train_mask: Tensor) -> Tensor:
    """Standardize node features with train-node statistics."""
    train_features = features[train_mask]
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (features - mean) / std


def train_and_evaluate(base_data: Data, node_ids: list[str], seed: int) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    """Train one seed and return metrics with predictions."""
    random.seed(seed)
    torch.manual_seed(seed)

    data = deepcopy(base_data)
    train_mask, val_mask, test_mask = build_masks(data.num_nodes, seed)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.x = standardize_features(data.x, train_mask)

    model = FlowLossGNN(input_channels=data.num_node_features, hidden_channels=HIDDEN_CHANNELS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_state: dict[str, Tensor] | None = None
    best_val_loss = float("inf")
    stale_epochs = 0

    for _ in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        predictions = model(data)
        train_loss = F.mse_loss(predictions[train_mask], data.y[train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(data)
            val_loss = F.mse_loss(val_predictions[val_mask], data.y[val_mask]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
            continue

        stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        predictions = model(data)

    actual_log = data.y[test_mask]
    predicted_log = predictions[test_mask]
    actual = torch.expm1(actual_log)
    predicted = torch.expm1(predicted_log).clamp_min(0.0)
    mae = torch.mean(torch.abs(predicted - actual)).item()
    rmse = torch.sqrt(torch.mean((predicted - actual) ** 2)).item()
    log_mae = torch.mean(torch.abs(predicted_log - actual_log)).item()
    log_rmse = torch.sqrt(torch.mean((predicted_log - actual_log) ** 2)).item()

    prediction_rows: list[dict[str, float | int | str]] = []
    test_indices = test_mask.nonzero(as_tuple=False).view(-1).tolist()
    for node_position in test_indices:
        prediction_rows.append(
            {
                "seed": seed,
                "node_id": node_ids[node_position],
                "actual_flow_loss": float(torch.expm1(data.y[node_position]).item()),
                "predicted_flow_loss": float(torch.expm1(predictions[node_position]).clamp_min(0.0).item()),
            }
        )

    metrics = {
        "seed": float(seed),
        "mae": mae,
        "rmse": rmse,
        "log_mae": log_mae,
        "log_rmse": log_rmse,
        "best_val_loss": best_val_loss,
    }
    return metrics, prediction_rows


def write_predictions(rows: list[dict[str, float | int | str]], csv_path: Path) -> None:
    """Write prediction rows to CSV."""
    fieldnames = ["seed", "node_id", "actual_flow_loss", "predicted_flow_loss"]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Train the node-regression GNN across multiple random seeds."""
    data, node_ids = build_data(GRAPHML_PATH)
    metric_rows: list[dict[str, float]] = []
    prediction_rows: list[dict[str, float | int | str]] = []

    for seed in RANDOM_SEEDS:
        metrics, seed_predictions = train_and_evaluate(data, node_ids, seed)
        metric_rows.append(metrics)
        prediction_rows.extend(seed_predictions)
        print(
            f"Seed {seed}: "
            f"MAE={metrics['mae']:.2f}, "
            f"RMSE={metrics['rmse']:.2f}, "
            f"log_MAE={metrics['log_mae']:.4f}, "
            f"log_RMSE={metrics['log_rmse']:.4f}, "
            f"best_val_loss={metrics['best_val_loss']:.4f}"
        )

    write_predictions(prediction_rows, PREDICTION_CSV_PATH)

    mean_mae = sum(row["mae"] for row in metric_rows) / len(metric_rows)
    mean_rmse = sum(row["rmse"] for row in metric_rows) / len(metric_rows)
    mean_log_mae = sum(row["log_mae"] for row in metric_rows) / len(metric_rows)
    mean_log_rmse = sum(row["log_rmse"] for row in metric_rows) / len(metric_rows)

    print(
        "Average: "
        f"MAE={mean_mae:.2f}, "
        f"RMSE={mean_rmse:.2f}, "
        f"log_MAE={mean_log_mae:.4f}, "
        f"log_RMSE={mean_log_rmse:.4f}"
    )
    print(f"Predictions written to: {PREDICTION_CSV_PATH}")


if __name__ == "__main__":
    main()

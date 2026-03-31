from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from composite_dataset import GraphDataset, RootedSubgraphDataset, SamplerConfig, load_graph_dataset, standardize_features, stratified_score_split
from composite_metrics import regression_metrics
from composite_model import CompositeScoreGNN


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration."""

    graphml_path: Path
    output_dir: Path
    hidden_dim: int
    num_layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    train_ratio: float
    val_ratio: float
    num_hops: int
    max_in_neighbors: int
    max_out_neighbors: int
    train_seed: int
    patience: int
    device: str


def parse_args() -> TrainConfig:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train a directed GraphSAGE model for composite score prediction.")
    parser.add_argument("--graphml-path", type=Path, default=Path("examples/gnn/composite_risk.graphml"))
    parser.add_argument("--output-dir", type=Path, default=Path("examples/gnn/output"))
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-hops", type=int, default=2)
    parser.add_argument("--max-in-neighbors", type=int, default=20)
    parser.add_argument("--max-out-neighbors", type=int, default=20)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    arguments = parser.parse_args()
    return TrainConfig(
        graphml_path=arguments.graphml_path,
        output_dir=arguments.output_dir,
        hidden_dim=arguments.hidden_dim,
        num_layers=arguments.num_layers,
        dropout=arguments.dropout,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        train_ratio=arguments.train_ratio,
        val_ratio=arguments.val_ratio,
        num_hops=arguments.num_hops,
        max_in_neighbors=arguments.max_in_neighbors,
        max_out_neighbors=arguments.max_out_neighbors,
        train_seed=arguments.train_seed,
        patience=arguments.patience,
        device=arguments.device,
    )


def set_seed(seed: int) -> None:
    """Set random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataloaders(
    dataset: GraphDataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test loaders."""
    sampler_config = SamplerConfig(
        num_hops=config.num_hops,
        max_in_neighbors=config.max_in_neighbors,
        max_out_neighbors=config.max_out_neighbors,
    )
    train_dataset = RootedSubgraphDataset(
        graph_dataset=dataset,
        node_indices=train_indices,
        sampler_config=sampler_config,
        training=True,
        seed=config.train_seed,
    )
    val_dataset = RootedSubgraphDataset(
        graph_dataset=dataset,
        node_indices=val_indices,
        sampler_config=sampler_config,
        training=False,
        seed=config.train_seed,
    )
    test_dataset = RootedSubgraphDataset(
        graph_dataset=dataset,
        node_indices=test_indices,
        sampler_config=sampler_config,
        training=False,
        seed=config.train_seed,
    )
    return (
        DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False),
    )


def train_epoch(
    model: CompositeScoreGNN,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch."""
    model.train()
    losses: list[float] = []

    for batch in tqdm(loader, desc="Train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        predictions = model(batch.x, batch.edge_index, batch.edge_weight)
        loss = criterion(predictions[batch.seed_mask], batch.y[batch.seed_mask])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else 0.0


def evaluate(
    model: CompositeScoreGNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    """Evaluate the model on one split."""
    model.eval()
    losses: list[float] = []
    actual_logs: list[float] = []
    predicted_logs: list[float] = []
    rows: list[dict[str, float]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            batch = batch.to(device)
            predictions = model(batch.x, batch.edge_index, batch.edge_weight)
            seed_predictions = predictions[batch.seed_mask]
            seed_targets = batch.y[batch.seed_mask]
            loss = criterion(seed_predictions, seed_targets)
            losses.append(float(loss.detach().cpu()))

            actual_logs.extend(seed_targets.detach().cpu().numpy().tolist())
            predicted_logs.extend(seed_predictions.detach().cpu().numpy().tolist())

            root_indices = batch.root_node.detach().cpu().numpy().tolist()
            for root_index, actual_log, predicted_log in zip(root_indices, seed_targets.cpu().numpy(), seed_predictions.cpu().numpy()):
                rows.append(
                    {
                        "root_index": int(root_index),
                        "actual_log_composite_score": float(actual_log),
                        "predicted_log_composite_score": float(predicted_log),
                        "actual_composite_score": float(np.expm1(actual_log)),
                        "predicted_composite_score": float(np.expm1(predicted_log)),
                    }
                )

    metrics = regression_metrics(np.array(actual_logs), np.array(predicted_logs))
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return mean_loss, metrics, rows


def write_predictions(
    rows: list[dict[str, float]],
    node_ids: list[str],
    path: Path,
) -> None:
    """Write prediction rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "node_id",
        "actual_log_composite_score",
        "predicted_log_composite_score",
        "actual_composite_score",
        "predicted_composite_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "node_id": node_ids[int(row["root_index"])],
                    "actual_log_composite_score": row["actual_log_composite_score"],
                    "predicted_log_composite_score": row["predicted_log_composite_score"],
                    "actual_composite_score": row["actual_composite_score"],
                    "predicted_composite_score": row["predicted_composite_score"],
                }
            )


def main() -> None:
    """Train and evaluate the composite score GNN."""
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.train_seed)
    device = torch.device(config.device)

    dataset = load_graph_dataset(config.graphml_path)
    train_indices, val_indices, test_indices = stratified_score_split(
        targets=dataset.targets,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.train_seed,
    )

    standardized_features, feature_mean, feature_std = standardize_features(dataset.features, train_indices)
    dataset = GraphDataset(
        node_ids=dataset.node_ids,
        node_id_to_index=dataset.node_id_to_index,
        features=standardized_features,
        targets=dataset.targets,
        raw_targets=dataset.raw_targets,
        in_neighbors=dataset.in_neighbors,
        out_neighbors=dataset.out_neighbors,
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        config=config,
    )

    model = CompositeScoreGNN(
        input_dim=dataset.features.size(1),
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.HuberLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_spearman={val_metrics['spearman']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            continue

        stale_epochs += 1
        if stale_epochs >= config.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    test_loss, test_metrics, test_rows = evaluate(model, test_loader, criterion, device)

    model_path = config.output_dir / "composite_score_gnn.pt"
    predictions_path = config.output_dir / "composite_score_predictions.csv"
    metrics_path = config.output_dir / "metrics.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "config": asdict(config),
        },
        model_path,
    )
    write_predictions(test_rows, dataset.node_ids, predictions_path)

    metrics_payload = {
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "history": history,
        "train_size": int(len(train_indices)),
        "val_size": int(len(val_indices)),
        "test_size": int(len(test_indices)),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")
    print(f"Saved model to {model_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

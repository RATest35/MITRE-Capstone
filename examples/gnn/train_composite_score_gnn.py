from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from composite_dataset import GraphDataset, RootedSubgraphDataset, SamplerConfig, load_graph_dataset, standardize_features, subnet_group_split
from composite_metrics import regression_metrics
from composite_model import CompositeScoreGNN


PROGRESS_ENABLED: bool = sys.stdout.isatty()


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
    eval_batch_size: int
    train_ratio: float
    val_ratio: float
    num_hops: int
    max_in_neighbors: int
    max_out_neighbors: int
    num_workers: int
    prefetch_factor: int
    selection_metric: str
    feature_set: str
    feature_groups: tuple[str, ...] | None
    group_by_prefix: int
    split_bucket_size: int
    weight_mode: str
    weight_scale: float
    ranking_loss_weight: float
    ranking_margin: float
    ranking_pairs: int
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
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-hops", type=int, default=2)
    parser.add_argument("--max-in-neighbors", type=int, default=20)
    parser.add_argument("--max-out-neighbors", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=max((os.cpu_count() or 1) - 2, 1))
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--selection-metric", type=str, default="ndcg_1pct")
    parser.add_argument("--feature-set", type=str, choices=["basic", "extended"], default="basic")
    parser.add_argument("--feature-groups", type=str, default="")
    parser.add_argument("--group-by-prefix", type=int, choices=[16, 24], default=24)
    parser.add_argument("--split-bucket-size", type=int, default=32)
    parser.add_argument("--weight-mode", type=str, choices=["linear", "sqrt", "quadratic"], default="linear")
    parser.add_argument("--weight-scale", type=float, default=4.0)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=0.02)
    parser.add_argument("--ranking-pairs", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    arguments = vars(parser.parse_args())
    feature_groups_text = arguments.pop("feature_groups")
    arguments["feature_groups"] = tuple(part.strip() for part in feature_groups_text.split(",") if part.strip()) or None
    return TrainConfig(**arguments)

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
    loader_kwargs: dict[str, object] = {
        "num_workers": config.num_workers,
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    def create_loader(node_indices: np.ndarray, batch_size: int, shuffle: bool, training: bool) -> DataLoader:
        dataset_split = RootedSubgraphDataset(
            graph_dataset=dataset,
            node_indices=node_indices,
            allowed_node_indices=node_indices,
            sampler_config=sampler_config,
            training=training,
            seed=config.train_seed,
        )
        return DataLoader(dataset_split, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)

    return (
        create_loader(train_indices, config.batch_size, True, True),
        create_loader(val_indices, config.eval_batch_size, False, False),
        create_loader(test_indices, config.eval_batch_size, False, False),
    )


def seed_predictions_and_targets(
    batch: torch.Tensor,
    predictions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract seed predictions, targets, and weights."""
    return (
        predictions[batch.seed_mask],
        batch.y[batch.seed_mask],
        batch.y_weight[batch.seed_mask],
    )


def weighted_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted Huber loss."""
    per_sample_loss = nn.functional.huber_loss(predictions, targets, reduction="none")
    return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)


def train_epoch(
    model: CompositeScoreGNN,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    config: TrainConfig,
) -> float:
    """Run one training epoch."""
    model.train()
    losses: list[float] = []

    for batch in tqdm(loader, desc="Train", leave=False, disable=not PROGRESS_ENABLED):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch.x, batch.edge_index, batch.edge_weight)
        seed_predictions, seed_targets, seed_weights = seed_predictions_and_targets(batch, predictions)
        regression_loss = weighted_huber_loss(seed_predictions, seed_targets, seed_weights)
        ranking_loss = pairwise_ranking_loss(
            seed_predictions,
            seed_targets,
            seed_weights,
            margin=config.ranking_margin,
            max_pairs=config.ranking_pairs,
        )
        loss = regression_loss + config.ranking_loss_weight * ranking_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else 0.0


def evaluate(
    model: CompositeScoreGNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    """Evaluate the model on one split."""
    model.eval()
    losses: list[float] = []
    actual_logs: list[float] = []
    predicted_logs: list[float] = []
    rows: list[dict[str, float]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False, disable=not PROGRESS_ENABLED):
            batch = batch.to(device)
            predictions = model(batch.x, batch.edge_index, batch.edge_weight)
            seed_predictions, seed_targets, seed_weights = seed_predictions_and_targets(batch, predictions)
            loss = weighted_huber_loss(seed_predictions, seed_targets, seed_weights)
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


def pairwise_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    margin: float,
    max_pairs: int,
) -> torch.Tensor:
    """Compute a simple pairwise ranking loss."""
    if predictions.numel() < 2 or max_pairs <= 0:
        return predictions.new_zeros(())

    order = torch.argsort(targets)
    pair_count = min(max_pairs, predictions.numel() // 2)
    if pair_count == 0:
        return predictions.new_zeros(())

    low_indices = order[:pair_count]
    high_indices = order[-pair_count:]
    valid_mask = targets[high_indices] > targets[low_indices]
    if not torch.any(valid_mask):
        return predictions.new_zeros(())

    gaps = predictions[high_indices][valid_mask] - predictions[low_indices][valid_mask]
    pair_weights = 0.5 * (weights[high_indices][valid_mask] + weights[low_indices][valid_mask])
    pair_losses = torch.relu(margin - gaps)
    return (pair_losses * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)


def serializable_config(config: TrainConfig) -> dict[str, object]:
    """Convert config values into JSON-safe objects."""
    values = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def prepare_dataset(
    config: TrainConfig,
) -> tuple[GraphDataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, split, and standardize the dataset."""
    raw_dataset = load_graph_dataset(
        graphml_path=config.graphml_path,
        feature_set=config.feature_set,
        feature_groups=config.feature_groups,
        weight_mode=config.weight_mode,
        weight_scale=config.weight_scale,
        split_prefix_len=config.group_by_prefix // 8,
    )
    train_indices, val_indices, test_indices = subnet_group_split(
        group_keys=raw_dataset.group_keys,
        targets=raw_dataset.targets,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.train_seed,
        bucket_size=config.split_bucket_size,
    )
    standardized_features, feature_mean, feature_std = standardize_features(raw_dataset.features, train_indices)
    dataset = GraphDataset(
        node_ids=raw_dataset.node_ids,
        node_id_to_index=raw_dataset.node_id_to_index,
        group_keys=raw_dataset.group_keys,
        features=standardized_features,
        targets=raw_dataset.targets,
        raw_targets=raw_dataset.raw_targets,
        sample_weights=raw_dataset.sample_weights,
        in_neighbors=raw_dataset.in_neighbors,
        out_neighbors=raw_dataset.out_neighbors,
    )
    return dataset, dataset.group_keys, train_indices, val_indices, test_indices, feature_mean, feature_std


def save_outputs(
    config: TrainConfig,
    model: CompositeScoreGNN,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    test_loss: float,
    test_metrics: dict[str, float],
    history: list[dict[str, float]],
    group_keys: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[Path, Path, Path]:
    """Save model and metrics artifacts."""
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

    metrics_payload = {
        "config": serializable_config(config),
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "history": history,
        "train_size": int(len(train_indices)),
        "val_size": int(len(val_indices)),
        "test_size": int(len(test_indices)),
        "train_group_count": int(len({group_keys[index] for index in train_indices.tolist()})),
        "val_group_count": int(len({group_keys[index] for index in val_indices.tolist()})),
        "test_group_count": int(len({group_keys[index] for index in test_indices.tolist()})),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=str), encoding="utf-8")
    return model_path, predictions_path, metrics_path


def main() -> None:
    """Train and evaluate the composite score GNN."""
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(config.train_seed)
    torch.manual_seed(config.train_seed)
    device = torch.device(config.device)

    dataset, group_keys, train_indices, val_indices, test_indices, feature_mean, feature_std = prepare_dataset(config)

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

    best_state: dict[str, torch.Tensor] | None = None
    best_val_score = float("-inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, config)
        val_loss, val_metrics, _ = evaluate(model, val_loader, device)
        val_score = float(val_metrics[config.selection_metric])
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
            f"{config.selection_metric}={val_score:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
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
    test_loss, test_metrics, _ = evaluate(model, test_loader, device)
    model_path, predictions_path, metrics_path = save_outputs(
        config=config,
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        test_loss=test_loss,
        test_metrics=test_metrics,
        history=history,
        group_keys=group_keys,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )

    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")
    print(f"Saved model to {model_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

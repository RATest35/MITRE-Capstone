from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from composite_dataset import GraphDataset, RootedSubgraphDataset, SamplerConfig, load_graph_dataset, subnet_group_split
from composite_metrics import regression_metrics
from composite_model import CompositeScoreGNN


@dataclass(frozen=True)
class InferenceConfig:
    """Inference configuration."""

    checkpoint_path: Path
    graphml_path: Path | None
    output_csv: Path
    batch_size: int
    num_workers: int
    device: str
    top_k: int
    eval_split: str


@dataclass(frozen=True)
class CheckpointConfig:
    """Minimal training settings needed for inference."""

    graphml_path: Path
    hidden_dim: int
    num_layers: int
    dropout: float
    train_ratio: float
    val_ratio: float
    num_hops: int
    max_in_neighbors: int
    max_out_neighbors: int
    feature_set: str
    feature_groups: tuple[str, ...] | None
    group_by_prefix: int
    split_bucket_size: int
    weight_mode: str
    weight_scale: float
    train_seed: int


def parse_args() -> InferenceConfig:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a trained composite score GNN.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("examples/gnn/output/composite_score_gnn.pt"),
    )
    parser.add_argument("--graphml-path", type=Path, default=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("examples/gnn/output/composite_score_inference.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--eval-split", type=str, choices=["all", "train", "val", "test"], default="all")
    arguments = parser.parse_args()
    return InferenceConfig(
        checkpoint_path=arguments.checkpoint_path,
        graphml_path=arguments.graphml_path,
        output_csv=arguments.output_csv,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        device=arguments.device,
        top_k=arguments.top_k,
        eval_split=arguments.eval_split,
    )


def parse_checkpoint_config(values: dict[str, object]) -> CheckpointConfig:
    """Extract the training settings needed for inference."""
    feature_groups_value = values.get("feature_groups")
    feature_groups = tuple(feature_groups_value) if feature_groups_value is not None else None
    return CheckpointConfig(
        graphml_path=Path(str(values["graphml_path"])),
        hidden_dim=int(values["hidden_dim"]),
        num_layers=int(values["num_layers"]),
        dropout=float(values["dropout"]),
        train_ratio=float(values["train_ratio"]),
        val_ratio=float(values["val_ratio"]),
        num_hops=int(values["num_hops"]),
        max_in_neighbors=int(values["max_in_neighbors"]),
        max_out_neighbors=int(values["max_out_neighbors"]),
        feature_set=str(values["feature_set"]),
        feature_groups=feature_groups,
        group_by_prefix=int(values["group_by_prefix"]),
        split_bucket_size=int(values["split_bucket_size"]),
        weight_mode=str(values["weight_mode"]),
        weight_scale=float(values["weight_scale"]),
        train_seed=int(values["train_seed"]),
    )


def load_inference_artifacts(
    config: InferenceConfig,
) -> tuple[CheckpointConfig, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Load checkpoint settings and tensors."""
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = parse_checkpoint_config(dict(checkpoint["config"]))
    model_state_dict = dict(checkpoint["model_state_dict"])
    feature_mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32)
    feature_std = torch.as_tensor(checkpoint["feature_std"], dtype=torch.float32)
    return checkpoint_config, model_state_dict, feature_mean, feature_std


def build_dataset(
    inference_config: InferenceConfig,
    checkpoint_config: CheckpointConfig,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> GraphDataset:
    """Build the standardized graph dataset for inference."""
    graphml_path = inference_config.graphml_path or checkpoint_config.graphml_path
    dataset = load_graph_dataset(
        graphml_path=graphml_path,
        feature_set=checkpoint_config.feature_set,
        feature_groups=checkpoint_config.feature_groups,
        weight_mode=checkpoint_config.weight_mode,
        weight_scale=checkpoint_config.weight_scale,
        split_prefix_len=checkpoint_config.group_by_prefix // 8,
    )
    standardized_features = (dataset.features - feature_mean) / feature_std
    return GraphDataset(
        node_ids=dataset.node_ids,
        node_id_to_index=dataset.node_id_to_index,
        group_keys=dataset.group_keys,
        features=standardized_features,
        targets=dataset.targets,
        raw_targets=dataset.raw_targets,
        sample_weights=dataset.sample_weights,
        in_neighbors=dataset.in_neighbors,
        out_neighbors=dataset.out_neighbors,
    )


def resolve_eval_indices(dataset: GraphDataset, checkpoint_config: CheckpointConfig, eval_split: str) -> np.ndarray:
    """Resolve node indices for one evaluation split."""
    if eval_split == "all":
        return np.arange(len(dataset.node_ids), dtype=np.int64)

    train_indices, val_indices, test_indices = subnet_group_split(
        group_keys=dataset.group_keys,
        targets=dataset.targets,
        train_ratio=checkpoint_config.train_ratio,
        val_ratio=checkpoint_config.val_ratio,
        seed=checkpoint_config.train_seed,
        bucket_size=checkpoint_config.split_bucket_size,
    )
    split_indices = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    return split_indices[eval_split]


def build_loader(dataset: GraphDataset, checkpoint_config: CheckpointConfig, config: InferenceConfig) -> DataLoader:
    """Build the inference dataloader."""
    sampler_config = SamplerConfig(
        num_hops=checkpoint_config.num_hops,
        max_in_neighbors=checkpoint_config.max_in_neighbors,
        max_out_neighbors=checkpoint_config.max_out_neighbors,
    )
    node_indices = resolve_eval_indices(dataset, checkpoint_config, config.eval_split)
    rooted_dataset = RootedSubgraphDataset(
        graph_dataset=dataset,
        node_indices=node_indices,
        allowed_node_indices=node_indices,
        sampler_config=sampler_config,
        training=False,
        seed=0,
    )
    return DataLoader(
        rooted_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )


def build_model(dataset: GraphDataset, checkpoint_config: CheckpointConfig, state_dict: dict[str, torch.Tensor], device: torch.device) -> CompositeScoreGNN:
    """Build and restore the trained model."""
    model = CompositeScoreGNN(
        input_dim=dataset.features.size(1),
        hidden_dim=checkpoint_config.hidden_dim,
        num_layers=checkpoint_config.num_layers,
        dropout=checkpoint_config.dropout,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference(
    model: CompositeScoreGNN,
    loader: DataLoader,
    dataset: GraphDataset,
    device: torch.device,
) -> list[dict[str, float | str]]:
    """Run inference and collect prediction rows."""
    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            batch = batch.to(device)
            predictions = model(batch.x, batch.edge_index, batch.edge_weight)
            seed_predictions = predictions[batch.seed_mask]
            root_indices = batch.root_node.detach().cpu().numpy().tolist()
            predicted_logs = seed_predictions.detach().cpu().numpy().tolist()

            for root_index, predicted_log in zip(root_indices, predicted_logs):
                actual_log = float(dataset.targets[root_index].item())
                actual_score = float(dataset.raw_targets[root_index].item())
                rows.append(
                    {
                        "node_id": dataset.node_ids[root_index],
                        "actual_log_composite_score": actual_log,
                        "predicted_log_composite_score": float(predicted_log),
                        "actual_composite_score": actual_score,
                        "predicted_composite_score": float(np.expm1(predicted_log)),
                    }
                )
    return rows


def write_predictions(rows: list[dict[str, float | str]], path: Path) -> None:
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
        writer.writerows(rows)


def compute_metrics(rows: list[dict[str, float | str]]) -> dict[str, float]:
    """Compute inference metrics from prediction rows."""
    actual_logs = np.asarray([float(row["actual_log_composite_score"]) for row in rows], dtype=np.float64)
    predicted_logs = np.asarray([float(row["predicted_log_composite_score"]) for row in rows], dtype=np.float64)
    metrics = regression_metrics(actual_logs, predicted_logs)
    metric_names = (
        "top_1pct_recall",
        "top_5pct_recall",
        "ndcg_1pct",
        "ndcg_5pct",
        "spearman",
        "log_mae",
    )
    return {name: float(metrics[name]) for name in metric_names}


def print_metrics(metrics: dict[str, float]) -> None:
    """Print the requested inference metrics."""
    print("Inference metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")


def print_top_predictions(rows: list[dict[str, float | str]], top_k: int) -> None:
    """Print the top predicted nodes."""
    top_rows = sorted(rows, key=lambda row: float(row["predicted_composite_score"]), reverse=True)[:top_k]
    print(f"Top {len(top_rows)} predicted nodes:")
    for rank, row in enumerate(top_rows, start=1):
        print(
            f"{rank:02d}. "
            f"node_id={row['node_id']} | "
            f"predicted={float(row['predicted_composite_score']):.6f} | "
            f"actual={float(row['actual_composite_score']):.6f}"
        )


def main() -> None:
    """Run composite score GNN inference."""
    config = parse_args()
    checkpoint_config, model_state_dict, feature_mean, feature_std = load_inference_artifacts(config)
    dataset = build_dataset(config, checkpoint_config, feature_mean, feature_std)
    loader = build_loader(dataset, checkpoint_config, config)
    device = torch.device(config.device)
    model = build_model(dataset, checkpoint_config, model_state_dict, device)
    rows = run_inference(model, loader, dataset, device)
    metrics = compute_metrics(rows)
    write_predictions(rows, config.output_csv)
    print_metrics(metrics)
    print(f"Evaluated split: {config.eval_split} ({len(rows)} nodes)")
    print_top_predictions(rows, config.top_k)
    print(f"Saved predictions to {config.output_csv}")


if __name__ == "__main__":
    main()

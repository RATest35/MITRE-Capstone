from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from dataset import GraphDataset, RootedSubgraphDataset, SamplerConfig, load_graph_dataset, standardize_features, subnet_group_split
from model import TransformerConvGNN

DEFAULT_LAYER_DIMS = [128, 96, 64, 32]

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# Parse the minimum set of CLI arguments for training.
# ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line options for TransformerConv training.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphml-path", type=Path, default=BASE_DIR.parent / "examples" / "ip-address" / "composite_score_with_bytes_per_sec.graphml")
    parser.add_argument("--output-path", type=Path, default=Path("transformerconv-gnn/composite_score_gnn.pt"))
    parser.add_argument("--layer-dims", type=int, nargs="+", default=DEFAULT_LAYER_DIMS)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--max-in-neighbors", type=int, default=32)
    parser.add_argument("--max-out-neighbors", type=int, default=32)
    parser.add_argument("--ranking-alpha", type=float, default=1.0)
    parser.add_argument("--sample-weight-max", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------
# Build the standardized graph dataset and split indices.
# ---------------------------------------------------------
def prepare_dataset(
    args: argparse.Namespace,
) -> tuple[GraphDataset, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Load, split, and standardize the graph dataset.

    :param args: Parsed training arguments.
    :return: Dataset, feature mean, feature standard deviation, and split indices.
    """
    raw_dataset = load_graph_dataset(
        args.graphml_path,
        sample_weight_max=args.sample_weight_max,
    )
    train_indices, val_indices, test_indices = subnet_group_split(
        group_keys=raw_dataset.group_keys,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=args.seed,
    )
    features, feature_mean, feature_std = standardize_features(raw_dataset.features, train_indices)
    dataset = GraphDataset(
        node_ids=raw_dataset.node_ids,
        node_id_to_index=raw_dataset.node_id_to_index,
        group_keys=raw_dataset.group_keys,
        features=features,
        targets=raw_dataset.targets,
        raw_targets=raw_dataset.raw_targets,
        sample_weights=raw_dataset.sample_weights,
        in_neighbors=raw_dataset.in_neighbors,
        out_neighbors=raw_dataset.out_neighbors,
    )
    return dataset, feature_mean, feature_std, train_indices, val_indices, test_indices


# ---------------------------------------------------------
# Build one rooted-subgraph loader for a split.
# ---------------------------------------------------------
def build_loader(
    dataset: GraphDataset,
    node_indices: np.ndarray,
    allowed_node_indices: np.ndarray,
    sampler_config: SamplerConfig,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build a rooted-subgraph data loader.

    :param dataset: Full graph dataset.
    :param node_indices: Root node indices exposed by the loader.
    :param allowed_node_indices: Node indices allowed during sampling.
    :param sampler_config: Rooted subgraph sampling configuration.
    :param batch_size: Number of samples per batch.
    :param shuffle: Whether to shuffle the dataset.
    :return: PyG data loader for rooted subgraphs.
    """
    rooted_dataset = RootedSubgraphDataset(
        graph_dataset=dataset,
        node_indices=node_indices,
        allowed_node_indices=allowed_node_indices,
        sampler_config=sampler_config,
    )
    return DataLoader(rooted_dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------
# Compute the weighted regression loss for root predictions.
# ---------------------------------------------------------
def compute_weighted_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute sample-weighted mean squared error.

    :param predictions: Predicted target values.
    :param targets: Expected target values.
    :param weights: Sample weights for each prediction.
    :return: Weighted MSE loss.
    """
    squared_errors = (predictions - targets).pow(2)
    weighted_errors = squared_errors * weights
    return weighted_errors.sum() / weights.sum().clamp_min(1e-12)


# ---------------------------------------------------------
# Compute the pairwise ranking loss across root predictions.
# ---------------------------------------------------------
def compute_pairwise_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted pairwise ranking loss.

    :param predictions: Predicted target values.
    :param targets: Expected target values.
    :param weights: Sample weights for each prediction.
    :return: Weighted pairwise ranking loss.
    """
    target_differences = targets.unsqueeze(1) - targets.unsqueeze(0)
    pair_mask = target_differences > 0

    if not pair_mask.any():
        return predictions.new_zeros(())

    prediction_differences = predictions.unsqueeze(1) - predictions.unsqueeze(0)
    pair_weights = 0.5 * (weights.unsqueeze(1) + weights.unsqueeze(0))
    ranking_terms = -torch.nn.functional.logsigmoid(prediction_differences)
    weighted_terms = ranking_terms[pair_mask] * pair_weights[pair_mask]
    return weighted_terms.sum() / pair_weights[pair_mask].sum().clamp_min(1e-12)


# ---------------------------------------------------------
# Compute the weighted regression and ranking loss terms.
# ---------------------------------------------------------
def compute_batch_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    ranking_alpha: float,
) -> torch.Tensor:
    """Combine regression and ranking losses.

    :param predictions: Predicted target values.
    :param targets: Expected target values.
    :param weights: Sample weights for each prediction.
    :param ranking_alpha: Multiplier applied to ranking loss.
    :return: Combined batch loss.
    """
    regression_loss = compute_weighted_mse_loss(predictions, targets, weights)
    ranking_loss = compute_pairwise_ranking_loss(predictions, targets, weights)
    return regression_loss + ranking_alpha * ranking_loss


# ---------------------------------------------------------
# Run one epoch and compute the masked training objective.
# ---------------------------------------------------------
def run_epoch(
    model: TransformerConvGNN,
    loader: DataLoader,
    optimizer: Adam | None,
    device: torch.device,
    ranking_alpha: float,
) -> float:
    """Run one training or evaluation epoch.

    :param model: TransformerConv model.
    :param loader: Data loader for a split.
    :param optimizer: Optimizer for training, or ``None`` for evaluation.
    :param device: Device used for tensor computation.
    :param ranking_alpha: Multiplier applied to ranking loss.
    :return: Average masked loss for the epoch.
    """
    total_loss = 0.0
    total_count = 0
    model.train() if optimizer is not None else model.eval()

    for batch in loader:
        batch = batch.to(device)
        predictions = model(batch.x, batch.edge_index, batch.edge_attr)
        mask = batch.seed_mask
        root_predictions = predictions[mask]
        root_targets = batch.y[mask]
        root_weights = batch.y_weight[mask]
        loss = compute_batch_loss(
            root_predictions,
            root_targets,
            root_weights,
            ranking_alpha,
        )

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_count = int(mask.sum().item())
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count

    return total_loss / max(total_count, 1)


# ---------------------------------------------------------
# Compute ranking metrics for top-risk node retrieval.
# ---------------------------------------------------------
def evaluate_ranking(
    model: TransformerConvGNN,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute top-risk retrieval metrics.

    :param model: TransformerConv model.
    :param loader: Data loader for a split.
    :param device: Device used for tensor computation.
    :return: Precision and NDCG metrics at the top five percent.
    """
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch.x, batch.edge_index, batch.edge_attr)
        mask = batch.seed_mask
        predictions.append(outputs[mask].detach().cpu())
        targets.append(batch.y[mask].detach().cpu())

    prediction_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    node_count = int(target_tensor.numel())
    top_5 = max(1, math.ceil(node_count * 0.05))
    predicted_order = torch.argsort(prediction_tensor, descending=True)
    target_order = torch.argsort(target_tensor, descending=True)
    predicted_top_5 = set(predicted_order[:top_5].tolist())
    target_top_5 = set(target_order[:top_5].tolist())
    dcg = sum(
        float(target_tensor[int(predicted_order[i])].item()) / math.log2(i + 2)
        for i in range(top_5)
    )
    ideal_dcg = sum(
        float(target_tensor[int(target_order[i])].item()) / math.log2(i + 2)
        for i in range(top_5)
    )

    return {
        "precision_at_5": len(predicted_top_5 & target_top_5) / top_5,
        "ndcg_at_5": dcg / max(ideal_dcg, 1e-12),
    }


# ---------------------------------------------------------
# Train the model, track the best validation loss, and save it.
# ---------------------------------------------------------
def main() -> None:
    """Train and save the TransformerConv model.

    :return: None.
    """
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dataset, feature_mean, feature_std, train_indices, val_indices, test_indices = prepare_dataset(args)
    sampler_config = SamplerConfig(
        num_hops=args.num_hops,
        max_in_neighbors=args.max_in_neighbors,
        max_out_neighbors=args.max_out_neighbors,
    )
    train_loader = build_loader(dataset, train_indices, train_indices, sampler_config, args.batch_size, True)
    val_loader = build_loader(dataset, val_indices, train_indices, sampler_config, args.batch_size, False)
    test_loader = build_loader(dataset, test_indices, train_indices, sampler_config, args.batch_size, False)
    model = TransformerConvGNN(
        input_dim=dataset.features.size(1),
        layer_dims=args.layer_dims,
        dropout=args.dropout,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val_precision = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.ranking_alpha,
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model,
                val_loader,
                None,
                device,
                args.ranking_alpha,
            )
            val_metrics = evaluate_ranking(model, val_loader, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"p@5={val_metrics['precision_at_5']:.4f} ndcg@5={val_metrics['ndcg_at_5']:.4f}"
        )
        if val_metrics["precision_at_5"] > best_val_precision:
            best_val_precision = val_metrics["precision_at_5"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience_count = 0
            continue

        patience_count += 1
        if patience_count >= args.patience:
            print(f"early_stop_epoch={epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        test_loss = run_epoch(
            model,
            test_loader,
            None,
            device,
            args.ranking_alpha,
        )
        test_metrics = evaluate_ranking(model, test_loader, device)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "config": vars(args),
            "test_loss": test_loss,
            "test_metrics": test_metrics,
        },
        args.output_path,
    )
    print(f"test_loss={test_loss:.6f}")
    print(
        f"test_p@5={test_metrics['precision_at_5']:.4f} "
        f"test_ndcg@5={test_metrics['ndcg_at_5']:.4f}"
    )
    print(f"saved={args.output_path}")


if __name__ == "__main__":
    main()

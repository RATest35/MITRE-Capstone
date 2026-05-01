from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from dataset import GraphDataset, RootedSubgraphDataset, SamplerConfig, load_graph_dataset, standardize_features, subnet_group_split
from model import GraphSAGEGNN

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# Parse the minimum set of CLI arguments for training.
# ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line options for GraphSAGE training.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphml-path", type=Path, default=BASE_DIR.parent / "examples" / "ip-address" / "composite_score_with_bytes_per_sec.graphml")
    parser.add_argument("--output-path", type=Path, default=Path("gnn/composite_score_gnn.pt"))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--max-in-neighbors", type=int, default=32)
    parser.add_argument("--max-out-neighbors", type=int, default=32)
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
    raw_dataset = load_graph_dataset(args.graphml_path)
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
# Run one epoch and compute the masked regression loss.
# ---------------------------------------------------------
def run_epoch(
    model: GraphSAGEGNN,
    loader: DataLoader,
    optimizer: Adam | None,
    device: torch.device,
) -> float:
    """Run one training or evaluation epoch.

    :param model: GraphSAGE model.
    :param loader: Data loader for a split.
    :param optimizer: Optimizer for training, or ``None`` for evaluation.
    :param device: Device used for tensor computation.
    :return: Average masked loss for the epoch.
    """
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_count = 0
    model.train() if optimizer is not None else model.eval()

    for batch in loader:
        batch = batch.to(device)
        predictions = model(batch.x, batch.edge_index, batch.edge_weight)
        mask = batch.seed_mask
        loss = criterion(predictions[mask], batch.y[mask])

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
    model: GraphSAGEGNN,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute top-risk retrieval metrics.

    :param model: GraphSAGE model.
    :param loader: Data loader for a split.
    :param device: Device used for tensor computation.
    :return: Precision and NDCG metrics at the top five percent.
    """
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch.x, batch.edge_index, batch.edge_weight)
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
    """Train and save the GraphSAGE model.

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
    model = GraphSAGEGNN(
        input_dim=dataset.features.size(1),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val_precision = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, device)
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
        test_loss = run_epoch(model, test_loader, None, device)
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

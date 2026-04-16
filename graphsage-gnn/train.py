from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from dataset import GraphDataset, RootedSubgraphDataset, SamplerConfig, load_graph_dataset, standardize_features, subnet_group_k_fold_split
from model import GraphSAGEGNN


# ---------------------------------------------------------
# Parse the minimum set of CLI arguments for training.
# ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphml-path", type=Path, default=Path("gnn/dataset/composite_risk.graphml"))
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
    parser.add_argument("--num-folds", type=int, default=5)
    return parser.parse_args()


# ---------------------------------------------------------
# Build the standardized graph dataset and split indices.
# ---------------------------------------------------------
def prepare_dataset(args: argparse.Namespace):
    raw_dataset = load_graph_dataset(args.graphml_path)
    folds = subnet_group_k_fold_split(
        group_keys=raw_dataset.group_keys,
        num_folds=args.num_folds,
        seed=args.seed,
    )
    return raw_dataset, folds

def build_fold_dataset(
    raw_dataset,
    train_indices: np.ndarray,
):
    features, feature_mean, feature_std = standardize_features(raw_dataset.features, train_indices)
    log_targets = torch.log1p(raw_dataset.raw_targets)

    dataset = GraphDataset(
        node_ids=raw_dataset.node_ids,
        node_id_to_index=raw_dataset.node_id_to_index,
        group_keys=raw_dataset.group_keys,
        features=features,
        targets=log_targets,
        raw_targets=raw_dataset.raw_targets,
        sample_weights=raw_dataset.sample_weights,
        in_neighbors=raw_dataset.in_neighbors,
        out_neighbors=raw_dataset.out_neighbors,
    )
    return dataset, feature_mean, feature_std


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

def evaluate_regression(model, loader, device):
    predictions_log = []
    targets_log = []
    targets_raw = []

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch.x, batch.edge_index, batch.edge_weight)
        mask = batch.seed_mask

        predictions_log.append(outputs[mask].detach().cpu())
        targets_log.append(batch.y[mask].detach().cpu())

        targets_raw.append(torch.expm1(batch.y[mask].detach().cpu()))

    pred_log = torch.cat(predictions_log)
    true_log = torch.cat(targets_log)

    pred_raw = torch.expm1(pred_log)
    true_raw = torch.cat(targets_raw)

    log_mse = torch.mean((pred_log - true_log) ** 2).item()

    mse = torch.mean((pred_raw - true_raw) ** 2).item()
    rmse = torch.sqrt(torch.mean((pred_raw - true_raw) ** 2)).item()
    mae = torch.mean(torch.abs(pred_raw - true_raw)).item()

    # R^2 score
    ss_res = torch.sum((true_raw - pred_raw) ** 2)
    ss_tot = torch.sum((true_raw - torch.mean(true_raw)) ** 2)
    r2 = 1 - (ss_res / ss_tot).item()

    return {
        "log_mse": log_mse,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


# ---------------------------------------------------------
# Train the model, track the best validation loss, and save it.
# ---------------------------------------------------------
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("device:", device)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    raw_dataset, folds = prepare_dataset(args)

    sampler_config = SamplerConfig(
        num_hops=args.num_hops,
        max_in_neighbors=args.max_in_neighbors,
        max_out_neighbors=args.max_out_neighbors,
    )

    fold_results = []

    for fold_num, (train_indices, val_indices) in enumerate(folds, start=1):
        print(f"\n===== fold {fold_num}/{args.num_folds} =====")

        dataset, feature_mean, feature_std = build_fold_dataset(raw_dataset, train_indices)

        train_loader = build_loader(
            dataset, train_indices, train_indices,
            sampler_config, args.batch_size, True
        )
        val_loader = build_loader(
            dataset, val_indices, train_indices,
            sampler_config, args.batch_size, False
        )

        model = GraphSAGEGNN(
            input_dim=dataset.features.size(1),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=args.lr)
        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        patience_count = 0
        best_val_reg = None

        for epoch in range(1, args.epochs + 1):

            train_loss = run_epoch(model, train_loader, optimizer, device)

            with torch.no_grad():
                val_loss = run_epoch(model, val_loader, None, device)
                val_reg = evaluate_regression(model, val_loader, device)


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_reg = val_reg
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                patience_count = 0
                continue

            patience_count += 1
            if patience_count >= args.patience:
                print(f"early_stop_fold={fold_num} epoch={epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        fold_results.append({
            "fold": fold_num,
            "best_val_loss": best_val_loss,
            "rmse": best_val_reg["rmse"],
            "mae": best_val_reg["mae"],
            "r2": best_val_reg["r2"],
            "log_mse": best_val_reg["log_mse"],
        })



    mean_val_loss = float(np.mean([result["best_val_loss"] for result in fold_results]))
    mean_rmse = float(np.mean([result["rmse"] for result in fold_results]))
    mean_mae = float(np.mean([result["mae"] for result in fold_results]))
    mean_r2 = float(np.mean([result["r2"] for result in fold_results]))
    mean_log_mse = float(np.mean([result["log_mse"] for result in fold_results]))

    std_rmse = float(np.std([result["rmse"] for result in fold_results]))
    std_mae = float(np.std([result["mae"] for result in fold_results]))
    std_r2 = float(np.std([result["r2"] for result in fold_results]))

    print("\n===== cross-validation summary =====")
    for result in fold_results:
        print(
            f"fold={result['fold']} "
            f"val_loss={result['best_val_loss']:.6f} "
            f"rmse={result['rmse']:.6f} "
            f"mae={result['mae']:.6f} "
            f"r2={result['r2']:.4f}"
        )

    print(
        f"cv_mean_val_loss={mean_val_loss:.6f} "
        f"cv_mean_log_mse={mean_log_mse:.6f} "
        f"cv_mean_rmse={mean_rmse:.6f} "
        f"cv_std_rmse={std_rmse:.6f} "
        f"cv_mean_mae={mean_mae:.6f} "
        f"cv_std_mae={std_mae:.6f} "
        f"cv_mean_r2={mean_r2:.4f} "
        f"cv_std_r2={std_r2:.4f}"
    )


if __name__ == "__main__":
    main()

"""Hyperparameter search for the GATv2 pipeline using Optuna.

Run from this directory:
    python tune.py                     # 50 trials (default)
    python tune.py --n-trials 100      # more trials
    python tune.py --n-trials 20       # quick smoke test

Each trial trains one GATv2 configuration on a fixed 60/20 train/val split
and returns val top_5pct_recall as the objective.  After the search finishes,
best parameters are written back into gat_config.py automatically and saved
to experiments/best_params.json.
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

import optuna
import torch
import torch.nn.functional as F
from torch import Tensor

from gat_config import DEVICE, GRAPHML_PATH
from gat_data import build_data, standardize_features
from gat_metrics import regression_metrics
from gat_model import GATv2Regressor


# ---------------------------------------------------------------------------
# Search-phase constants (not shared with production config)
# ---------------------------------------------------------------------------
_SEARCH_EPOCHS: int = 150
_SEARCH_PATIENCE: int = 30
_TRAIN_RATIO: float = 0.6
_VAL_RATIO: float = 0.2
_SEED: int = 42

# Maps trial param names → gat_config.py constant names
_PARAM_TO_CONST: dict[str, str] = {
    "hidden_channels":    "HIDDEN_CHANNELS",
    "num_heads":          "NUM_HEADS",
    "dropout":            "DROPOUT",
    "drop_edge_prob":     "DROP_EDGE_PROB",
    "learning_rate":      "LEARNING_RATE",
    "weight_decay":       "WEIGHT_DECAY",
    "ranking_loss_weight":"RANKING_LOSS_WEIGHT",
    "ranking_margin":     "RANKING_MARGIN",
    "weight_mode":        "WEIGHT_MODE",
    "weight_scale":       "WEIGHT_SCALE",
}


def _fixed_split(num_nodes: int) -> tuple[Tensor, Tensor]:
    """Return a fixed train/val boolean mask pair (identical across all trials)."""
    gen = torch.Generator()
    gen.manual_seed(_SEED)
    perm = torch.randperm(num_nodes, generator=gen)
    n_train = int(num_nodes * _TRAIN_RATIO)
    n_val = int(num_nodes * _VAL_RATIO)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train : n_train + n_val]] = True

    return train_mask.to(DEVICE), val_mask.to(DEVICE)


def _sample_weights(raw_targets: Tensor, mode: str, scale: float) -> Tensor:
    """Rank-based sample weights matching the production pipeline."""
    norm_rank = torch.argsort(torch.argsort(raw_targets)).float() / max(len(raw_targets) - 1, 1)
    if mode == "sqrt":
        return 1.0 + scale * norm_rank.sqrt()
    if mode == "quadratic":
        return 1.0 + scale * norm_rank.square()
    return 1.0 + scale * norm_rank


def _ranking_loss(preds: Tensor, targets: Tensor, margin: float, max_pairs: int = 128) -> Tensor:
    """Pairwise ranking hinge loss."""
    if preds.numel() < 2:
        return preds.new_zeros(())
    order = torch.argsort(targets)
    pair_count = min(max_pairs, preds.numel() // 2)
    low_idx, high_idx = order[:pair_count], order[-pair_count:]
    valid = targets[high_idx] > targets[low_idx]
    if not torch.any(valid):
        return preds.new_zeros(())
    gaps = preds[high_idx][valid] - preds[low_idx][valid]
    return torch.relu(margin - gaps).mean()


def _apply_best_params(config_path: Path, params: dict[str, object]) -> None:
    """Patch best-found hyperparameters back into gat_config.py in-place."""
    content = config_path.read_text(encoding="utf-8")

    for param, const in _PARAM_TO_CONST.items():
        if param not in params:
            continue
        value = params[param]
        formatted = f'"{value}"' if isinstance(value, str) else repr(value)
        content = re.sub(
            rf"^({re.escape(const)}\s*:\s*\w+\s*=\s*).*$",
            rf"\g<1>{formatted}",
            content,
            flags=re.MULTILINE,
        )

    # Keep SELECTION_METRIC aligned with what we optimised for
    content = re.sub(
        r'^(SELECTION_METRIC\s*:\s*str\s*=\s*).*$',
        r'\g<1>"top_5pct_recall"',
        content,
        flags=re.MULTILINE,
    )

    config_path.write_text(content, encoding="utf-8")


def objective(trial: optuna.Trial, base_data: object, train_mask: Tensor, val_mask: Tensor) -> float:
    """Train one GATv2 configuration and return val top_5pct_recall."""
    hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128, 256])
    num_heads       = trial.suggest_categorical("num_heads", [2, 4, 8])
    dropout         = trial.suggest_float("dropout", 0.1, 0.5)
    drop_edge_prob  = trial.suggest_float("drop_edge_prob", 0.0, 0.4)
    lr              = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay    = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    rank_weight     = trial.suggest_categorical("ranking_loss_weight", [0.0, 0.05, 0.1, 0.2])
    rank_margin     = trial.suggest_float("ranking_margin", 0.01, 0.1)
    weight_mode     = trial.suggest_categorical("weight_mode", ["linear", "sqrt", "quadratic"])
    weight_scale    = trial.suggest_float("weight_scale", 1.0, 8.0)

    data = deepcopy(base_data)
    data.x = standardize_features(data.x, train_mask)

    raw_targets = torch.expm1(data.y)
    weights = _sample_weights(raw_targets, weight_mode, weight_scale).to(DEVICE)

    model = GATv2Regressor(
        input_channels=data.num_node_features,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        dropout=dropout,
        drop_edge_prob=drop_edge_prob,
        edge_dim=data.edge_attr.shape[1],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_score = float("-inf")
    stale_epochs = 0

    for epoch in range(_SEARCH_EPOCHS):
        model.train()
        optimizer.zero_grad()
        preds = model(data)

        w = weights[train_mask]
        per_sample = F.huber_loss(preds[train_mask], data.y[train_mask], reduction="none")
        loss = (per_sample * w).sum() / w.sum().clamp_min(1e-6)

        if rank_weight > 0.0:
            loss = loss + rank_weight * _ranking_loss(preds[train_mask], data.y[train_mask], rank_margin)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(data)

        val_score = regression_metrics(
            data.y[val_mask].cpu().numpy(),
            val_preds[val_mask].cpu().numpy(),
        )["top_5pct_recall"]

        trial.report(val_score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_score > best_val_score:
            best_val_score = val_score
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= _SEARCH_PATIENCE:
                break

    return best_val_score


def main() -> None:
    """Run hyperparameter search, then apply best params to gat_config.py."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("experiments/best_params.json"))
    args = parser.parse_args()

    torch.manual_seed(_SEED)
    print(f"Loading graph from {GRAPHML_PATH} ...")
    data, _ = build_data(GRAPHML_PATH)
    train_mask, val_mask = _fixed_split(data.num_nodes)
    print(f"Nodes: {data.num_nodes}  |  Train: {train_mask.sum().item()}  |  Val: {val_mask.sum().item()}")
    print(f"Running {args.n_trials} trials (objective: top_5pct_recall) ...\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )
    study.optimize(
        lambda trial: objective(trial, data, train_mask, val_mask),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_value = study.best_value

    print(f"\nBest trial       : #{study.best_trial.number}")
    print(f"Best top_5% rec  : {best_value:.4f}")
    print("Best params      :")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Save JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"best_top_5pct_recall": best_value, "best_params": best_params}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved to {args.output}")

    # Patch gat_config.py in-place
    config_path = Path(__file__).parent / "gat_config.py"
    _apply_best_params(config_path, best_params)
    print(f"Updated {config_path.name} with best parameters.")
    print("Run `python train.py` to evaluate with full K-fold CV.")


if __name__ == "__main__":
    main()

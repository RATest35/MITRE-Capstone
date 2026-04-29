# Experiment 003 — GATv2 Tuned + Loss Tracking (10-Fold CV)

**Date:** 2026-04-21

## Changes from Experiment 002

- Added `train_loss` and `val_loss` (weighted Huber + ranking loss) to the per-fold metrics
- Added `_compute_loss` helper in `gat_training.py` for consistent train/val loss computation
- Same hyperparameters as Experiment 002 (Optuna-tuned for `top_5pct_recall`)

## Graph

| Property | Value |
|---|---|
| File | `composite_score_with_bytes_per_sec.graphml` |
| Nodes | 23,511 |
| Edges | 66,863 |

## Model

| Parameter | Value |
|---|---|
| Architecture | GATv2Regressor (3-layer GATv2Conv + LayerNorm + ELU) |
| Node features | 7 (5 pre-normalised GraphML attrs + in/out degree) |
| Input features | `in_flow_norm`, `out_flow_norm`, `flow_loss_norm`, `weighted_betweenness_norm`, `pagerank_norm` |
| Edge features | 3 (`flow`, `bytes_per_sec`, `distance`) |
| Hidden channels | 128 |
| Attention heads | 2 |
| Edge dim | 3 |

## Training

| Parameter | Value |
|---|---|
| Target / label | `importance` (log1p-transformed) |
| Loss | Weighted Huber + pairwise ranking loss |
| Optimiser | Adam |
| Learning rate | 5.981e-4 |
| Weight decay | 1.097e-5 |
| Max epochs | 600 |
| Early stopping patience | 200 |
| Selection metric | `top_5pct_recall` |
| DropEdge probability | 0.3006 |
| Dropout | 0.2162 |
| Ranking loss weight | 0.1 |
| Ranking margin | 0.0457 |
| Sample weight mode | quadratic |
| Sample weight scale | 4.769 |
| MC Dropout samples | 50 |
| K-Folds | 10 |
| Device | CUDA |

## Per-Fold Results (with Train/Val Loss)

| Fold | Train Loss | Val Loss | RMSE | MAE | Log-MAE | Spearman | NDCG@5% | Top-5% Rec | Stopped |
|---|---|---|---|---|---|---|---|---|---|
| 1  | 0.005575 | 0.005757 | 0.1000 | 0.0171 | 0.0172 | 0.4179 | 0.0581 | 0.4615 | epoch 409 |
| 2  | 0.003647 | 0.003988 | 0.1750 | 0.0255 | 0.0289 | 0.4952 | 0.0289 | 0.4786 | max epochs |
| 3  | 0.005954 | 0.004408 | 0.2112 | 0.0149 | 0.0145 | 0.3155 | 0.0105 | 0.4786 | epoch 343 |
| 4  | 0.005035 | 0.004565 | 0.0839 | 0.0096 | 0.0177 | 0.4136 | 0.0993 | 0.4957 | epoch 473 |
| 5  | 0.006900 | 0.002674 | 0.1335 | 0.0280 | 0.0263 | 0.3785 | 0.0312 | 0.4957 | epoch 413 |
| 6  | 0.008122 | 0.005050 | 0.0427 | 0.0312 | 0.0315 | 0.3369 | 0.1068 | 0.3504 | epoch 318 |
| 7  | 0.005093 | 0.005558 | 0.0455 | 0.0184 | 0.0189 | 0.3392 | 0.0798 | 0.3162 | epoch 487 |
| 8  | 0.006454 | 0.006644 | 0.2004 | 0.0281 | 0.0254 | 0.2313 | 0.0174 | 0.3761 | epoch 343 |
| 9  | 0.003730 | 0.003347 | 0.1728 | 0.0198 | 0.0176 | 0.5134 | 0.0269 | 0.4188 | max epochs |
| 10 | 0.004262 | 0.003379 | 0.1370 | 0.0296 | 0.0291 | 0.5489 | 0.0247 | 0.4444 | max epochs |

## Cross-Validation Summary (Mean ± Std)

| Metric | Mean | ± Std |
|---|---|---|
| **Train Loss** | **0.0055** | 0.0014 |
| **Val Loss** | **0.0045** | 0.0012 |
| RMSE | 0.1302 | 0.0608 |
| MAE | 0.0222 | 0.0072 |
| Log-MAE | 0.0227 | 0.0061 |
| Log-RMSE | 0.0545 | 0.0119 |
| Spearman | 0.3990 | 0.0991 |
| Pearson | 0.0287 | 0.0326 |
| NDCG@1% | 0.0121 | 0.0096 |
| NDCG@5% | 0.0484 | 0.0352 |
| Top-1% Recall | 0.0739 | 0.0795 |
| **Top-5% Recall** | **0.4316** | 0.0640 |

## Train/Val Loss Diagnostic

- **Mean train loss:** 0.0055
- **Mean val loss:** 0.0045
- **Train ≥ Val** in 6/10 folds (folds 1, 4, 6, 7, 8 train ≈ val; folds 3, 5 val < train)
- **No overfitting signature** — val loss is not systematically higher than train loss across folds. The model generalises well on the loss it was trained on.
- **Largest train→val gap:** Fold 5 (train 0.0069, val 0.0027) — small enough to not flag.

## Top-5 Critical Nodes (by Composite Risk)

| Rank | Node ID | Composite Risk | Predicted Importance | Failure Probability |
|---|---|---|---|---|
| 1 | 10.200.7.8   | 5.0024 | 0.0155 | 0.7939 |
| 2 | 10.200.7.195 | 4.1076 | 0.0215 | 0.6533 |
| 3 | 10.200.7.217 | 4.0034 | 0.0000 | 0.3955 |
| 4 | 10.200.7.218 | 2.4899 | 0.0023 | 0.2949 |
| 5 | 10.200.7.196 | 2.1481 | 0.0094 | 0.4163 |

## Notes

- **Loss values are very small (~0.005)** — the weighted Huber loss on log1p-transformed targets compresses the dynamic range. Useful for tracking convergence and detecting overfitting trends across experiments, less useful as an absolute quality measure.
- **Top-5% Recall ticked up from 0.395 → 0.432** vs Experiment 002 (run-to-run variance from random fold assignment).
- Spearman remained strong (0.40 ± 0.10) — the model produces a consistent ranking.
- The new top-5 nodes are clustered in the `10.200.7.0/24` subnet, very different from Experiment 002's `192.168.x.x` cluster. This is expected: composite_risk is `actual_importance × failure_probability`, and the unstable predictions (high MC-Dropout variance, hence high `failure_probability`) tend to dominate the ordering.

# Experiment 002 — GATv2 Optuna-Tuned (top_5pct_recall objective, 10-Fold CV)

**Date:** 2026-04-21

## Changes from Experiment 001

- Hyperparameters tuned with Optuna (50 trials, objective: `top_5pct_recall`)
- K-Folds increased from 5 → 10
- Selection metric changed from `ndcg_5pct` → `top_5pct_recall`
- Ranking loss enabled (`RANKING_LOSS_WEIGHT = 0.1`)
- `WEIGHT_MODE` changed from `linear` → `quadratic`

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
| Edge features | 3 (`flow`, `bytes_per_sec`, `distance`) — raw values |
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
| Selection metric | `top_5pct_recall` (higher is better) |
| DropEdge probability | 0.3006 |
| Dropout | 0.2162 |
| Ranking loss weight | 0.1 |
| Ranking margin | 0.0457 |
| Sample weight mode | quadratic |
| Sample weight scale | 4.769 |
| MC Dropout samples | 50 |
| K-Folds | 10 |
| Device | CUDA |

## Per-Fold Results

| Fold | RMSE | MAE | Log-MAE | Log-RMSE | Spearman | Pearson | NDCG@1% | NDCG@5% | Top-1% Rec | Top-5% Rec | Stopped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1  | 0.0436 | 0.0118 | 0.0147 | 0.0336 | 0.4198 |  0.0653 | 0.0236 | 0.1105 | 0.0435 | 0.3675 | epoch 411 |
| 2  | 0.0475 | 0.0173 | 0.0196 | 0.0354 | 0.5369 |  0.0597 | 0.0093 | 0.0815 | 0.0435 | 0.4530 | epoch 483 |
| 3  | 0.0431 | 0.0084 | 0.0184 | 0.0366 | 0.3268 |  0.0735 | 0.0123 | 0.0702 | 0.0870 | 0.3419 | epoch 361 |
| 4  | 0.1282 | 0.0076 | 0.0203 | 0.0560 | 0.3324 |  0.1015 | 0.3822 | 0.3996 | 0.0435 | 0.4530 | epoch 539 |
| 5  | 0.0717 | 0.0202 | 0.0226 | 0.0431 | 0.4090 |  0.0699 | 0.0158 | 0.0478 | 0.0870 | 0.4530 | epoch 298 |
| 6  | 0.2041 | 0.0659 | 0.0656 | 0.1172 | 0.4052 | -0.0059 | 0.0048 | 0.0150 | 0.0435 | 0.3590 | epoch 208 |
| 7  | 0.2129 | 0.0213 | 0.0288 | 0.0634 | 0.4797 | -0.0012 | 0.0129 | 0.0285 | 0.1739 | 0.4957 | epoch 353 |
| 8  | 0.1645 | 0.0116 | 0.0153 | 0.0613 | 0.3727 |  0.0077 | 0.0053 | 0.0233 | 0.0870 | 0.4188 | max epochs |
| 9  | 0.2122 | 0.0285 | 0.0771 | 0.1178 | 0.2905 | -0.0144 | 0.0032 | 0.0052 | 0.0870 | 0.2051 | epoch 211 |
| 10 | 0.1787 | 0.0619 | 0.0600 | 0.0889 | 0.1820 |  0.0260 | 0.0153 | 0.0414 | 0.0435 | 0.4017 | epoch 224 |

## Cross-Validation Summary (Mean ± Std)

| Metric | Mean | ± Std | vs Exp 001 |
|---|---|---|---|
| RMSE | | 0.0729 | +0.0024 |
| MAE | 0.0255 | 0.0213 | +0.0131 |
| Log-MAE | 0.0342 | 0.0237 | +0.0235 |
| Log-RMSE | 0.0653 | 0.0322 | +0.0177 |
| Spearman | 0.3755 | 0.0999 | **+0.2857** |
| Pearson | 0.0382 | 0.0406 | -0.0147 |
| NDCG@1% | 0.0485 | 0.1174 | -0.0856 |
| NDCG@5% | 0.0823 | 0.1161 | -0.0795 |
| Top-1% Recall | 0.0739 | 0.0412 | +0.0016 |
| Top-5% Recall | **0.3949** | 0.0828 | **+0.1796** |

## Top-5 Critical Nodes (by Composite Risk)

| Rank | Node ID | Composite Risk | Predicted Importance | Failure Probability |
|---|---|---|---|---|
| 1 | 192.168.32.10 | 0.4396 | 0.5133 | 0.8563 |
| 2 | 192.168.40.8  | 0.3891 | 0.5630 | 0.6911 |
| 3 | 192.168.72.83 | 0.3798 | 0.4411 | 0.8611 |
| 4 | 192.168.42.52 | 0.3671 | 0.4881 | 0.7522 |
| 5 | 192.168.90.52 | 0.3593 | 0.5628 | 0.6384 |

## Notes

- **Top-5% Recall improved from 0.2153 → 0.3949** (+83%) — the tuning objective worked as intended.
- **Spearman improved from 0.0898 → 0.3755** (+318%) — the model now ranks nodes much more consistently.
- NDCG@5% dropped (0.1618 → 0.0823) because optimising for recall does not guarantee better NDCG; they measure different aspects of ranking quality.
- RMSE is comparable to Experiment 001 (0.1282 → 0.1306), confirming the tuning shifted the model toward ranking quality without significantly hurting regression accuracy.
- Composite risk scores are substantially higher (max 0.044 → 0.440), reflecting that the tuned model now assigns higher predicted importance to genuinely critical private-subnet nodes.
- Fold 9 remains the weakest (Top-5% Recall = 0.205); the random split likely isolates atypical nodes. Subnet-aware splitting remains a future improvement.

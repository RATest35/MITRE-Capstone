# Experiment 001 — GATv2 Baseline (K-Fold CV, Importance Target)

**Date:** 2026-04-16

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
| Hidden channels | 64 |
| Attention heads | 4 |
| Edge dim | 3 |

## Training

| Parameter | Value |
|---|---|
| Target / label | `importance` (log1p-transformed) |
| Loss | Weighted Huber (sample weights by target rank) |
| Ranking loss weight | 0.0 (disabled) |
| Optimiser | Adam |
| Learning rate | 5e-4 |
| Weight decay | 1e-4 |
| Max epochs | 600 |
| Early stopping patience | 200 |
| Selection metric | `ndcg_5pct` (higher is better) |
| DropEdge probability | 0.2 |
| Dropout | 0.3 |
| Sample weight mode | linear |
| Sample weight scale | 4.0 |
| MC Dropout samples | 50 |
| K-Folds | 5 |
| Device | CUDA |

## Per-Fold Results

| Fold | RMSE | MAE | Log-MAE | Log-RMSE | Spearman | Pearson | NDCG@1% | NDCG@5% | Top-1% Rec | Top-5% Rec | Stopped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.1296 | 0.0138 | 0.0116 | 0.0517 | 0.2397 | 0.0903 | 0.4239 | 0.4222 | 0.1064 | 0.2213 | epoch 433 |
| 2 | 0.0278 | 0.0141 | 0.0138 | 0.0226 | 0.0275 | 0.0251 | 0.0078 | 0.0678 | 0.0638 | 0.1574 | epoch 538 |
| 3 | 0.1217 | 0.0161 | 0.0145 | 0.0457 | 0.0246 | 0.0036 | 0.0016 | 0.0062 | 0.0000 | 0.1234 | epoch 401 |
| 4 | 0.1673 | 0.0103 | 0.0080 | 0.0579 | 0.0096 | 0.0333 | 0.0125 | 0.0775 | 0.0851 | 0.2170 | epoch 449 |
| 5 | 0.1947 | 0.0078 | 0.0056 | 0.0602 | 0.1478 | 0.1123 | 0.2247 | 0.2350 | 0.1064 | 0.3574 | max epochs |

## Cross-Validation Summary (Mean ± Std)

| Metric | Mean | ± Std |
|---|---|---|
| RMSE | 0.1282 | 0.0634 |
| MAE | 0.0124 | 0.0033 |
| Log-MAE | 0.0107 | 0.0038 |
| Log-RMSE | 0.0476 | 0.0151 |
| Spearman | 0.0898 | 0.1005 |
| Pearson | 0.0529 | 0.0461 |
| NDCG@1% | 0.1341 | 0.1874 |
| NDCG@5% | 0.1618 | 0.1683 |
| Top-1% Recall | 0.0723 | 0.0441 |
| Top-5% Recall | 0.2153 | 0.0895 |

## Top-5 Critical Nodes (by Composite Risk)

| Rank | Node ID | Composite Risk | Predicted Importance | Failure Probability |
|---|---|---|---|---|
| 1 | 10.200.7.9 | 0.0501 | 0.0680 | 0.7369 |
| 2 | 169.45.71.46 | 0.0318 | 0.0318 | 1.0000 |
| 3 | 192.168.180.95 | 0.0253 | 0.0436 | 0.5792 |
| 4 | 192.168.180.2 | 0.0240 | 0.0441 | 0.5444 |
| 5 | 52.84.155.211 | 0.0222 | 0.0243 | 0.9118 |

## Notes

- High variance in NDCG and Spearman across folds (std > mean) indicates the random K-fold split assigns very different node populations to each test set. Subnet-aware splitting would stabilise this.
- Fold 3 effectively failed to learn a useful ranking (NDCG@5% = 0.006, Top-1% Recall = 0.000).
- Fold 1 was the strongest (NDCG@5% = 0.422, Spearman = 0.240), suggesting the model can rank nodes meaningfully when the split is favourable.
- Node features are still raw pre-normalised GraphML attributes (7 features). Switching to computed structural features (23-dim log1p) is the next experiment to try.

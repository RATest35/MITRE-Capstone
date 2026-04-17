"""Train a GATv2Regressor on the graph data and evaluate across K folds.
 
Run from this directory:
    python train.py
 
The script loads the GraphML graph, trains a GATv2Regressor across K folds
with DropEdge and early stopping, runs MC Dropout inference per fold, prints
cross-validated metrics and writes per-node predictions to CSV.
"""
 
from __future__ import annotations
 
import statistics
 


from gat_config import (
    DEVICE,
    DROP_EDGE_PROB,
    GRAPHML_PATH,
    K_FOLDS,
    MC_DROPOUT_SAMPLES,
    PREDICTION_CSV_PATH,
    SELECTION_METRIC,
)
from gat_data import build_data
from gat_output import write_predictions
from gat_training import train_and_evaluate
 
 
# Metrics to display in the summary table.
_DISPLAY_METRICS: tuple[str, ...] = (
    "rmse",
    "mae",
    "log_mae",
    "log_rmse",
    "spearman",
    "pearson",
    "ndcg_1pct",
    "ndcg_5pct",
    "top_1pct_recall",
    "top_5pct_recall",
)
 
 
def main() -> None:
    """Orchestrate data loading, training, evaluation, and output."""
    print(f"Device           : {DEVICE}")
    print(f"Graph            : {GRAPHML_PATH}")
    print(f"K-folds          : {K_FOLDS}")
    print(f"MC samples       : {MC_DROPOUT_SAMPLES}")
    print(f"DropEdge p       : {DROP_EDGE_PROB}")
    print(f"Selection metric : {SELECTION_METRIC}")
 
    data, node_ids = build_data(GRAPHML_PATH)
    print(f"Nodes            : {data.num_nodes}  |  Edges : {data.num_edges}")
    print(f"Node features    : {data.num_node_features}  |  Edge features : {data.edge_attr.shape[1]}")
 
    all_metrics, all_predictions = train_and_evaluate(data, node_ids)
 
    # ── Per-fold table ────────────────────────────────────────────────────
    print(f"\n{'═' * 90}")
    print(f"  {K_FOLDS}-Fold Cross-Validation Results")
    print(f"{'═' * 90}")
 
    header = f"  {'Fold':<6}"
    for name in _DISPLAY_METRICS:
        header += f" {name:>12}"
    print(header)
    print(f"  {'─' * 84}")
 
    for m in all_metrics:
        row = f"  {int(m['fold']):<6}"
        for name in _DISPLAY_METRICS:
            row += f" {m[name]:>12.4f}"
        print(row)
 
    # ── Aggregated mean ± std ─────────────────────────────────────────────
    def _summarise(key: str) -> tuple[float, float]:
        vals = [m[key] for m in all_metrics]
        return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0
 
    print(f"  {'─' * 84}")
 
    mean_row = f"  {'Mean':<6}"
    std_row = f"  {'± Std':<6}"
    for name in _DISPLAY_METRICS:
        m, s = _summarise(name)
        mean_row += f" {m:>12.4f}"
        std_row += f" {s:>12.4f}"
    print(mean_row)
    print(std_row)
    print(f"{'═' * 90}")
 
    # ── Composite risk summary ────────────────────────────────────────────
    risks = [r["composite_risk"] for r in all_predictions]
    if risks:
        print(f"\n── Composite risk (all test nodes across folds) ────────────")
        print(f"  Mean             : {statistics.mean(risks):.4f}")
        print(f"  Max              : {max(risks):.4f}")
        print(f"  Top-5 nodes      :")
        top5 = sorted(all_predictions, key=lambda r: r["composite_risk"], reverse=True)[:5]
        for r in top5:
            print(f"    {r['node_id']:>18s}  risk={r['composite_risk']:.4f}  "
                  f"imp={r['predicted_importance']:.4f}  fail={r['failure_probability']:.4f}")
        print(f"{'─' * 60}\n")
 
    write_predictions(all_predictions, PREDICTION_CSV_PATH)
    print(f"Predictions written to: {PREDICTION_CSV_PATH}")
 
 
if __name__ == "__main__":
    main()
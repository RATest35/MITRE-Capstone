"""Configuration constants for the GATv2 training pipeline."""

from __future__ import annotations

from pathlib import Path

import torch


# Device

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Paths
BASE_DIR: Path = Path(__file__).resolve().parent
 
# GraphML input: ip-address graph with flow node/edge attributes.
GRAPHML_PATH: Path = BASE_DIR.parent / "examples" / "ip-address" / "composite_score_with_bytes_per_sec.graphml"

# CSV file where per-node predictions are written after evaluation.
PREDICTION_CSV_PATH: Path = BASE_DIR / "gat_predictions.csv"


# Data
# Node attribute names used as input features (must not include the label).
# Using pre-normalised attributes from the GraphML.
# in/out degree are appended programmatically in gat_data.py.
NODE_FEATURE_KEYS: list[str] = [
    "in_flow_norm",
    "out_flow_norm",
    "flow_loss_norm",
    "weighted_betweenness_norm",
    "pagerank_norm",
]

# Importance score components used to build the training target.
# The target is sum(component values), then log1p-transformed during loading.
IMPORTANCE_COMPONENT_KEYS: list[str] = [
    "weighted_betweenness_norm",
    "pagerank_norm",
    "flow_loss_norm",
]

# Edge attribute names used as edge features fed into GATv2Conv.
EDGE_FEATURE_KEYS: list[str] = ["flow", "bytes_per_sec", "distance"]

# Dimensionality of the edge feature vector.
EDGE_DIM: int = 3


# Cross-validation

# Number of folds for K-fold cross-validation.
K_FOLDS: int = 10


# Model architecture
# ---------------------------------------------------------------------------
# Number of attention heads in each GATv2Conv layer.
NUM_HEADS: int = 2

# Hidden feature dimension per attention head.
HIDDEN_CHANNELS: int = 128


# Training
# ---------------------------------------------------------------------------
# Maximum number of gradient-update epochs.
NUM_EPOCHS: int = 600

# Adam learning rate (lower than GraphSAGE — GAT benefits from gentler updates).
LEARNING_RATE: float = 0.0005981364320010053

# L2 regularisation strength (Adam weight_decay).
WEIGHT_DECAY: float = 1.0969671773921393e-05

# Consecutive epochs without validation improvement before early stopping.
PATIENCE: int = 200

# Dropout probability applied after each GATv2Conv layer.
DROPOUT: float = 0.21621159087579556


# DropEdge & MC Dropout

# Probability of dropping each edge per forward pass (DropEdge).
DROP_EDGE_PROB: float = 0.30061413828907585

# Number of stochastic forward passes at inference for MC Dropout.
MC_DROPOUT_SAMPLES: int = 50


# Sample weighting (matches composite pipeline)

# Weight samples by target rank: "linear", "sqrt", or "quadratic".
WEIGHT_MODE: str = "quadratic"

# Scale factor for sample weighting.  weight = 1 + scale * normalised_rank.
WEIGHT_SCALE: float = 4.769499605360235


# Ranking loss (optional, matches composite pipeline)
# Weight of the pairwise ranking loss added to the regression loss.
# Set to 0.0 to disable ranking loss entirely.
RANKING_LOSS_WEIGHT: float = 0.1

# Margin for the pairwise ranking hinge loss.
RANKING_MARGIN: float = 0.04568004906727538

# Number of ranking pairs sampled per training step.
RANKING_PAIRS: int = 128

# ---------------------------------------------------------------------------
# Model selection metric
# ---------------------------------------------------------------------------
# Metric used for early stopping / best-model selection during validation.
# Options: any key from regression_metrics() e.g. "log_mae", "rmse",
# "spearman", "ndcg_5pct".  Metrics where higher is better are detected
# automatically.
SELECTION_METRIC: str = "top_5pct_recall"

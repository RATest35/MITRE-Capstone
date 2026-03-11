"""Configuration for GNN training."""

from __future__ import annotations

from pathlib import Path


BASE_DIR: Path = Path(__file__).resolve().parent

# Path to the input GraphML file used for training and evaluation.
GRAPHML_PATH: Path = BASE_DIR / "ip_graph_23511_nodes_edges_with_flow-2.graphml"

# Path to the CSV file where per-node predictions are written.
PREDICTION_CSV_PATH: Path = BASE_DIR / "gnn_predictions.csv"

# Fraction of nodes assigned to the training split.
TRAIN_RATIO: float = 0.6

# Fraction of nodes assigned to the validation split.
VAL_RATIO: float = 0.2

# Fraction of nodes assigned to the test split.
TEST_RATIO: float = 0.2

# Hidden feature size used inside the GNN layers.
HIDDEN_CHANNELS: int = 32

# Maximum number of training epochs for one run.
NUM_EPOCHS: int = 400

# Step size used by the optimizer.
LEARNING_RATE: float = 0.01

# L2 regularization strength applied by the optimizer.
WEIGHT_DECAY: float = 1e-4

# Number of epochs to wait for validation improvement before early stopping.
PATIENCE: int = 40

# Dropout probability applied during training.
DROPOUT: float = 0.1

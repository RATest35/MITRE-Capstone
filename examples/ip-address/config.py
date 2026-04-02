"""
config.py
---------
Centralized configuration for the IP graph pipeline.

All file paths, attribute names, and tunable constants live here.
Nothing in the rest of the codebase should have a hardcoded path or magic string.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PipelineConfig:
    """
    Immutable configuration object.

    Freeze the dataclass so that no module can accidentally mutate
    a shared config instance at runtime.
    """

    # ------------------------------------------------------------------ I/O
    input_graphml: str = "ip_graph_39_nodes_243_edges_with_flow.graphml"
    output_graphml: str = "composite_risk.graphml"

    # ------------------------------------------------------------------ CSV / dataset
    csv_path: str = ""                  # resolved at runtime if empty
    kaggle_dataset: str = "jsrojas/ip-network-traffic-flows-labeled-with-87-apps"

    # ------------------------------------------------------------------ Column names
    col_source_ip: str = "Source.IP"
    col_dest_ip: str = "Destination.IP"
    col_fwd_bytes: str = "Total.Length.of.Fwd.Packets"

    # ------------------------------------------------------------------ Graph attributes
    flow_attr: str = "flow"
    strength_attr: str = "bytes_per_sec"
    distance_attr: str = "distance"
    pagerank_weight: str = "flow"

    # ------------------------------------------------------------------ Numeric constants
    epsilon: float = 1e-9               # guard against division by zero
    pagerank_normalized: bool = True
    betweenness_normalized: bool = True

    # ------------------------------------------------------------------ Output
    top_n_nodes: int = 10               # how many critical nodes to print


# Singleton default config — import and use this directly, or override in tests.
DEFAULT_CONFIG = PipelineConfig()

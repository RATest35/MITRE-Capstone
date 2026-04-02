"""
graph_builder.py
----------------
SRP: converts a raw pandas DataFrame into a NetworkX DiGraph.

``GraphBuilder`` does *not* enrich the graph (no flow sums, no PageRank).
Enrichment is the responsibility of ``graph_enricher.py``.  This separation
makes each class easy to test independently.
"""

from __future__ import annotations

import logging

import networkx as nx
import pandas as pd

from config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds a directed graph from a ``DataFrame`` of IP flow records.

    The builder:
      1. Aggregates (source, destination) pairs by summing forward-packet bytes.
      2. Creates a ``DiGraph`` with a ``flow`` edge attribute.
      3. Extracts the largest weakly-connected component.

    Parameters
    ----------
    config:
        Pipeline configuration.  Defaults to ``DEFAULT_CONFIG``.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config

    # ------------------------------------------------------------------ public

    def build(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Return a ``DiGraph`` built from *df*.

        Parameters
        ----------
        df:
            DataFrame with at least three columns:
            ``Source.IP``, ``Destination.IP``, and
            ``Total.Length.of.Fwd.Packets`` (names resolved from config).

        Returns
        -------
        nx.DiGraph
            Largest weakly-connected component of the flow graph.
        """
        cfg = self._cfg
        logger.info("Aggregating flows …")
        df_agg = df.groupby(
            [cfg.col_source_ip, cfg.col_dest_ip], as_index=False
        ).agg(flow=(cfg.col_fwd_bytes, "sum"))

        logger.info("Building directed graph …")
        G = nx.DiGraph()
        G.add_weighted_edges_from(
            zip(df_agg[cfg.col_source_ip], df_agg[cfg.col_dest_ip], df_agg["flow"]),
            weight=cfg.flow_attr,
        )

        logger.info(
            "Full graph — nodes: %d  edges: %d",
            G.number_of_nodes(),
            G.number_of_edges(),
        )

        subgraph = self._largest_component(G)
        logger.info(
            "Largest WCC — nodes: %d  edges: %d",
            subgraph.number_of_nodes(),
            subgraph.number_of_edges(),
        )
        return subgraph

    # ------------------------------------------------------------------ private

    @staticmethod
    def _largest_component(G: nx.DiGraph) -> nx.DiGraph:
        """Return the largest weakly-connected component as an independent copy."""
        cc = max(nx.weakly_connected_components(G), key=len)
        return G.subgraph(cc).copy()

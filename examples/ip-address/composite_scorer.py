"""
composite_scorer.py
-------------------
Facade pattern: ``CompositeScorer`` orchestrates the enrichment pipeline
without knowing *which* enrichers are used or *how* they work.

Design principles applied
-------------------------
* **SRP** — this class is responsible only for running the pipeline and
  persisting the result; metric logic lives in ``graph_enricher.py``.
* **DIP** — enrichers and IO are injected; ``CompositeScorer`` depends only
  on the ``MetricEnricher`` abstraction and the ``GraphIO`` interface.
* **OCP** — swapping or extending the enricher list requires zero changes here.
"""

from __future__ import annotations

import logging
import time

import networkx as nx

from config import PipelineConfig, DEFAULT_CONFIG
from graph_enricher import MetricEnricher, default_enricher_pipeline
from graph_io import GraphIO

logger = logging.getLogger(__name__)


class CompositeScorer:
    """
    Orchestrates graph enrichment and persists the result.

    Parameters
    ----------
    enrichers:
        Ordered list of ``MetricEnricher`` instances to apply to the graph.
        Defaults to the standard pipeline returned by
        ``default_enricher_pipeline()``.
    io:
        ``GraphIO`` instance used for loading and saving the graph.
        Defaults to a new ``GraphIO()`` instance.
    config:
        Pipeline configuration.  Defaults to ``DEFAULT_CONFIG``.

    Examples
    --------
    Default usage::

        scorer = CompositeScorer()
        G = scorer.run("input.graphml", "output.graphml")

    Custom enricher pipeline::

        scorer = CompositeScorer(
            enrichers=[FlowEnricher(), PageRankEnricher(), CompositeScoreEnricher()]
        )
        G = scorer.run("input.graphml", "output.graphml")
    """

    def __init__(
        self,
        enrichers: list[MetricEnricher] | None = None,
        io: GraphIO | None = None,
        config: PipelineConfig = DEFAULT_CONFIG,
    ) -> None:
        self._config = config
        self._io = io or GraphIO()
        self._enrichers: list[MetricEnricher] = (
            enrichers if enrichers is not None else default_enricher_pipeline(config)
        )

    # ------------------------------------------------------------------ public

    def run(self, input_path: str, output_path: str) -> nx.DiGraph:
        """
        Load a graph, apply all enrichers in order, save, and return the result.

        Parameters
        ----------
        input_path:
            Path to the input ``.graphml`` file.
        output_path:
            Path where the enriched graph will be saved.

        Returns
        -------
        nx.DiGraph
            The fully enriched graph.
        """
        start = time.perf_counter()

        G = self._io.load(input_path)

        for enricher in self._enrichers:
            logger.info("Applying enricher: %s", type(enricher).__name__)
            G = enricher.enrich(G)

        self._io.save(G, output_path)

        elapsed = time.perf_counter() - start
        logger.info("Pipeline completed in %.4f seconds.", elapsed)

        return G

    # ------------------------------------------------------------------ reporting

    def top_nodes(
        self,
        G: nx.DiGraph,
        n: int | None = None,
        score_attr: str = "composite_score",
    ) -> list[tuple[str, dict]]:
        """
        Return the top-*n* nodes sorted by *score_attr* descending.

        Parameters
        ----------
        G:
            Enriched graph.
        n:
            Number of top nodes to return.  Defaults to ``config.top_n_nodes``.
        score_attr:
            Node attribute to rank by.

        Returns
        -------
        list of (node_id, attribute_dict) tuples
        """
        k = n if n is not None else self._config.top_n_nodes
        return sorted(
            G.nodes(data=True),
            key=lambda x: x[1].get(score_attr, 0),
            reverse=True,
        )[:k]

"""
graph_enricher.py
-----------------
Strategy pattern: every graph metric is encapsulated as a self-contained
``MetricEnricher`` subclass.

Design principles applied
-------------------------
* **SRP** — each enricher does exactly one thing (add one attribute / metric).
* **OCP** — to add a new metric, create a new subclass; no existing code changes.
* **LSP** — every enricher is substitutable; ``CompositeScorer`` only knows the
  ``MetricEnricher`` interface.
* **ISP** — the interface is a single method ``enrich(G) -> DiGraph``; no
  enricher is forced to implement unused methods.
* **DIP** — ``CompositeScorer`` accepts a ``list[MetricEnricher]`` (the
  abstraction) and is injected with concrete instances at construction time.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod

import networkx as nx

from config import PipelineConfig, DEFAULT_CONFIG
from utils import safe_float

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Abstract base — the Strategy interface
# ══════════════════════════════════════════════════════════════════════════════

class MetricEnricher(ABC):
    """
    Abstract enricher that adds one metric to every node (or edge) of a graph.

    Subclasses implement ``enrich`` and return the *same* graph object
    (mutated in-place and returned for fluent chaining).
    """

    @abstractmethod
    def enrich(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Compute the metric and store it as node/edge attributes on *G*.

        Parameters
        ----------
        G:
            The directed graph to enrich.

        Returns
        -------
        nx.DiGraph
            The same graph with new attributes added.
        """


# ══════════════════════════════════════════════════════════════════════════════
# Concrete enrichers
# ══════════════════════════════════════════════════════════════════════════════

class FlowEnricher(MetricEnricher):
    """
    Computes ``in_flow``, ``out_flow``, and ``flow_loss`` for every node.

    This replaces the duplicated flow-loop that existed in both
    ``generate_graph_model.py`` and ``composite_score.py``.

    Parameters
    ----------
    config:
        Pipeline configuration (used for the flow attribute name).
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config

    # Attributes this enricher exclusively owns.
    _OWNED_ATTRS = ("in_flow", "out_flow", "flow_loss")

    def enrich(self, G: nx.DiGraph) -> nx.DiGraph:
        logger.info("FlowEnricher: computing in/out/flow_loss for %d nodes …", G.number_of_nodes())
        flow_attr = self._cfg.flow_attr

        for n in G.nodes():
            # Remove any pre-existing values for these attributes.
            # Graphs loaded from a GraphML that was produced by the old
            # generate_graph_model.py store these as integers ("long").
            # If we don't clear them first, NetworkX will register two
            # separate <key> declarations (one "long", one "double") and
            # write both into the output GraphML, creating duplicates.
            node_data = G.nodes[n]
            for attr in self._OWNED_ATTRS:
                node_data.pop(attr, None)

            # Use 0.0 as the start value so sum() returns a float even when
            # a node has no in-edges or no out-edges (empty generator).
            # Without 0.0, sum() returns int(0) for empty iterables, which
            # causes NetworkX to register a duplicate "long" key alongside
            # the "double" key in the output GraphML.
            in_flow = sum(
                (safe_float(data.get(flow_attr, 0.0))
                 for _, _, data in G.in_edges(n, data=True)),
                0.0,
            )
            out_flow = sum(
                (safe_float(data.get(flow_attr, 0.0))
                 for _, _, data in G.out_edges(n, data=True)),
                0.0,
            )
            node_data["in_flow"] = in_flow
            node_data["out_flow"] = out_flow
            node_data["flow_loss"] = in_flow + out_flow

        return G


class RandomProbabilityEnricher(MetricEnricher):
    """
    Assigns a random probability in ``[0, 1)`` to every node as
    ``random_probability``.

    This value is used downstream as the *risk* factor in the composite score,
    replacing the original hardcoded ``risk = 1`` with a genuine per-node
    stochastic signal.

    The ``random.random()`` call is intentionally kept here so that the risk
    signal is generated once, stored on the graph, and available to any later
    enricher (e.g. ``CompositeScoreEnricher``) without re-rolling.
    """

    def enrich(self, G: nx.DiGraph) -> nx.DiGraph:
        logger.info(
            "RandomProbabilityEnricher: assigning random_probability to %d nodes …",
            G.number_of_nodes(),
        )
        for n in G.nodes():
            G.nodes[n]["random_probability"] = random.random()
        return G


class PageRankEnricher(MetricEnricher):
    """
    Computes PageRank weighted by the ``flow`` edge attribute and stores it
    as ``pagerank`` on every node.

    Parameters
    ----------
    config:
        Pipeline configuration (used for weight attribute name).
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config

    def enrich(self, G: nx.DiGraph) -> nx.DiGraph:
        logger.info("PageRankEnricher: computing PageRank …")
        pr = nx.pagerank(G, weight=self._cfg.pagerank_weight)
        for n, val in pr.items():
            G.nodes[n]["pagerank"] = val
        return G


class DistanceEnricher(MetricEnricher):
    """
    Converts a strength edge attribute into a distance attribute using
    ``distance = 1 / (strength + epsilon)``.

    Higher traffic ↔ stronger connection ↔ shorter distance.
    This is a prerequisite for ``WeightedBetweennessEnricher``.

    Parameters
    ----------
    config:
        Pipeline configuration (strength/distance attribute names + epsilon).
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config

    def enrich(self, G: nx.DiGraph) -> nx.DiGraph:
        logger.info("DistanceEnricher: computing edge distances …")
        cfg = self._cfg
        for _, _, data in G.edges(data=True):
            strength = safe_float(data.get(cfg.strength_attr, 0.0))
            data[cfg.distance_attr] = 1.0 / (strength + cfg.epsilon)
        return G


class WeightedBetweennessEnricher(MetricEnricher):
    """
    Computes betweenness centrality weighted by the *distance* edge attribute
    and stores it as ``weighted_betweenness`` on every node.

    **Important:** ``DistanceEnricher`` must run before this enricher so that
    the distance attribute is already present on each edge.  The enricher
    registration order in ``CompositeScorer`` enforces this.

    Parameters
    ----------
    config:
        Pipeline configuration (distance attribute name + normalization flag).
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config

    def enrich(self, G: nx.DiGraph) -> nx.DiGraph:
        logger.info("WeightedBetweennessEnricher: computing betweenness centrality …")
        bet = nx.betweenness_centrality(
            G,
            weight=self._cfg.distance_attr,
            normalized=self._cfg.betweenness_normalized,
        )
        for n, score in bet.items():
            G.nodes[n]["weighted_betweenness"] = score
        return G


class CompositeScoreEnricher(MetricEnricher):
    """
    Combines ``flow_loss``, ``weighted_betweenness``, and ``pagerank`` into
    an ``importance`` score, then multiplies by the node's
    ``random_probability`` to produce ``composite_score``.

    Formula
    -------
    ::

        importance      = flow_loss + weighted_betweenness + pagerank
        composite_score = random_probability * importance

    **Prerequisites** (must run earlier in the pipeline):
      - ``FlowEnricher``               → ``flow_loss``
      - ``WeightedBetweennessEnricher``→ ``weighted_betweenness``
      - ``PageRankEnricher``           → ``pagerank``
      - ``RandomProbabilityEnricher``  → ``random_probability``

    If a prerequisite attribute is missing the value defaults to ``0``,
    so the pipeline is fault-tolerant but a warning is logged.
    """

    def enrich(self, G: nx.DiGraph) -> nx.DiGraph:
        logger.info("CompositeScoreEnricher: computing composite scores …")
        for n, data in G.nodes(data=True):
            flow = data.get("flow_loss", 0.0)
            bet = data.get("weighted_betweenness", 0.0)
            pr = data.get("pagerank", 0.0)
            risk = data.get("random_probability", 0.0)

            if risk == 0.0:
                logger.warning(
                    "Node %s has no random_probability attribute; "
                    "ensure RandomProbabilityEnricher runs first.",
                    n,
                )

            importance = flow + bet + pr
            data["importance"] = importance
            data["composite_score"] = risk * importance

        return G


# ══════════════════════════════════════════════════════════════════════════════
# Factory helper — builds the default enricher pipeline
# ══════════════════════════════════════════════════════════════════════════════

def default_enricher_pipeline(
    config: PipelineConfig = DEFAULT_CONFIG,
) -> list[MetricEnricher]:
    """
    Return the standard ordered list of enrichers for the IP-graph pipeline.

    Order matters:
      1. ``FlowEnricher``                — node flow attributes
      2. ``DistanceEnricher``            — edge distance (needed by betweenness)
      3. ``WeightedBetweennessEnricher`` — needs distance edges
      4. ``PageRankEnricher``            — independent of above two
      5. ``RandomProbabilityEnricher``   — per-node risk factor
      6. ``CompositeScoreEnricher``      — must run last; reads all above

    Callers may substitute, extend, or reorder this list when constructing
    ``CompositeScorer`` — the scorer itself is oblivious to which enrichers
    are used (OCP).
    """
    return [
        FlowEnricher(config),
        DistanceEnricher(config),
        WeightedBetweennessEnricher(config),
        PageRankEnricher(config),
        RandomProbabilityEnricher(),
        CompositeScoreEnricher(),
    ]

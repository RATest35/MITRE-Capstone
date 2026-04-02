"""
graph_io.py
-----------
Repository pattern: all GraphML file I/O is isolated here.

No other module should call ``nx.read_graphml`` or ``nx.write_graphml``
directly.  Routing all I/O through this class makes it trivial to swap the
file format (e.g., to JSON/GML) without touching any pipeline logic.
"""

import logging

import networkx as nx

logger = logging.getLogger(__name__)


class GraphIO:
    """
    Handles reading and writing of graph files.

    Responsibilities (SRP)
    ----------------------
    One class, one concern: persist and restore a NetworkX graph to/from disk.
    All other pipeline classes depend on this abstraction, not on raw
    ``networkx`` I/O calls (DIP).
    """

    # ------------------------------------------------------------------ read

    def load(self, path: str) -> nx.DiGraph:
        """
        Load a GraphML file from *path* and return a ``DiGraph``.

        Parameters
        ----------
        path:
            Filesystem path to the ``.graphml`` file.

        Returns
        -------
        nx.DiGraph
        """
        logger.info("Loading graph from %s", path)
        graph = nx.read_graphml(path)
        logger.info(
            "Loaded graph — nodes: %d  edges: %d",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    # ------------------------------------------------------------------ write

    def save(self, graph: nx.DiGraph, path: str) -> None:
        """
        Write *graph* to *path* as a GraphML file.

        Parameters
        ----------
        graph:
            The graph to persist.
        path:
            Destination filesystem path.
        """
        logger.info("Saving graph to %s", path)
        nx.write_graphml(graph, path)
        logger.info("Graph saved successfully.")

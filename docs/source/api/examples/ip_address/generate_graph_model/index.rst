examples.ip_address.generate_graph_model
========================================

.. py:module:: examples.ip_address.generate_graph_model


Attributes
----------

.. autoapisummary::

   examples.ip_address.generate_graph_model.DATA_PATH
   examples.ip_address.generate_graph_model.SEED


Functions
---------

.. autoapisummary::

   examples.ip_address.generate_graph_model.bell_curve_probability
   examples.ip_address.generate_graph_model.load_and_aggregate_data
   examples.ip_address.generate_graph_model.build_graph
   examples.ip_address.generate_graph_model.largest_component
   examples.ip_address.generate_graph_model.add_flow_metrics
   examples.ip_address.generate_graph_model.add_random_probability
   examples.ip_address.generate_graph_model.main


Module Contents
---------------

.. py:data:: DATA_PATH

.. py:data:: SEED
   :value: 74


.. py:function:: bell_curve_probability(mean: float = 0.1, std_dev: float = 0.04) -> float

   Generate a bounded random probability from a normal distribution.

   :param mean: Mean of the Gaussian distribution.
   :param std_dev: Standard deviation of the Gaussian distribution.
   :return: Float probability clipped to range [0, 1].

   Notes:
   - Uses random.gauss for sampling
   - Values are clipped to ensure valid probability bounds


.. py:function:: load_and_aggregate_data(path: pathlib.Path) -> pandas.DataFrame

   Load CSV data and aggregate flows between IP pairs.

   :param path: Path to cleaned CSV file.
   :return: Aggregated DataFrame with flow and bytes_per_sec columns.

   Notes:
   - Converts numeric columns safely
   - Drops rows with invalid/missing values
   - Aggregates total flow as the sum of packet lengths.
   - Aggregates bytes_per_sec as the median for robustness.


.. py:function:: build_graph(df: pandas.DataFrame) -> networkx.DiGraph

   Construct a directed graph from aggregated flow data.

   :param df: DataFrame containing edge-level data.
   :return: Directed NetworkX graph with edge attributes.

   Notes:
   - Nodes represent IP addresses.
   - Edges represent communication between IPs.
   - Edge attributes include flow and bytes_per_sec.


.. py:function:: largest_component(G: networkx.DiGraph) -> networkx.DiGraph

   Extract the largest weakly connected component.

   :param G: Input directed graph.
   :return: Subgraph containing the largest component.

   Notes:
   - Uses weak connectivity (ignores edge direction)
   - Helps remove isolated or insignificant subgraphs


.. py:function:: add_flow_metrics(G: networkx.DiGraph) -> networkx.DiGraph

   Compute flow-based node metrics.

   :param G: Input graph.
   :return: Graph with node attributes:
       - in_flow
       - out_flow
       - flow_loss

   Notes:
   - in_flow: sum of incoming edge flow
   - out_flow: sum of outgoing edge flow
   - flow_loss: total traffic through node


.. py:function:: add_random_probability(G: networkx.DiGraph) -> networkx.DiGraph

   Assign random probability scores to nodes.

   :param G: Input graph.
   :return: Graph with 'random_probability' node attribute.

   Notes:
   - Uses bell_curve_probability for realistic distribution
   - Simulates uncertainty or risk likelihood per node


.. py:function:: main()

   Run full pipeline to build and export graph.

   Steps:
   1. Load and aggregate CSV data
   2. Build directed graph
   3. Extract largest connected component
   4. Compute node flow metrics
   5. Assign random probabilities
   6. Save graph as GraphML file

   :return: None



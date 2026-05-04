examples.ip_address.composite_score
===================================

.. py:module:: examples.ip_address.composite_score


Attributes
----------

.. autoapisummary::

   examples.ip_address.composite_score.INPUT_PATH
   examples.ip_address.composite_score.OUTPUT_PATH
   examples.ip_address.composite_score.input_path


Functions
---------

.. autoapisummary::

   examples.ip_address.composite_score.to_float
   examples.ip_address.composite_score.load_graph
   examples.ip_address.composite_score.add_total_flow
   examples.ip_address.composite_score.normalize_multiple_attributes
   examples.ip_address.composite_score.add_pagerank
   examples.ip_address.composite_score.add_distance_from_strength
   examples.ip_address.composite_score.add_weighted_betweenness
   examples.ip_address.composite_score.add_composite_score
   examples.ip_address.composite_score.compute_composite_score


Module Contents
---------------

.. py:data:: INPUT_PATH
   :value: 'ip_graph_23511_nodes_66863_edges_seed74.graphml'


.. py:data:: OUTPUT_PATH
   :value: 'composite_score_with_bytes_per_sec.graphml'


.. py:function:: to_float(value)

   Convert a value to float if possible

   :param value: Input value to convert
   :return: Float value or None if conversion fails


.. py:function:: load_graph(file_name)

   Load a GraphML file into a NetworkX graph

   :param file_name: Path to the GraphML file
   :return: Loaded NetworkX graph


.. py:function:: add_total_flow(G, flow_attr='flow')

   Compute total in/out flow and flow loss for each node

   :param G: Input graph
   :param flow_attr: Edge attribute representing flow
   :return: Graph with added node attributes (in_flow, out_flow, flow_loss)


.. py:function:: normalize_multiple_attributes(G, attr_specs)

   Normalize multiple node attributes to specified ranges

   :param G: Input graph
   :param attr_specs: List of tuples (attr_name, normalized_name, new_min, new_max)
   :return: Graph with normalized attributes added


.. py:function:: add_pagerank(G)

   Compute PageRank scores using edge weights

   :param G: Input graph
   :return: Graph with 'pagerank' node attribute


.. py:function:: add_distance_from_strength(G, strength_attr='bytes_per_sec', distance_attr='distance', epsilon=1e-09)

   Convert edge strength to distance

   :param G: Input graph
   :param strength_attr: Edge attribute representing strength
   :param distance_attr: Name of distance attribute to create
   :param epsilon: Small constant to avoid division by zero
   :return: Graph with distance attribute added to edges


.. py:function:: add_weighted_betweenness(G, strength_attr='bytes_per_sec', distance_attr='distance')

   Compute weighted betweenness centrality

   :param G: Input graph
   :param strength_attr: Edge attribute representing strength
   :param distance_attr: Edge attribute used for distance
   :return: Graph with 'weighted_betweenness' node attribute


.. py:function:: add_composite_score(G)

   Compute composite score based on risk and importance

   :param G: Input graph with normalized attributes
   :return: Graph with 'importance' and 'composite_score' attributes


.. py:function:: compute_composite_score(input_file, output_file)

   Run full pipeline to compute composite scores

   :param input_file: Path to input GraphML file
   :param output_file: Path to save output GraphML file
   :return: Processed graph with computed attributes


.. py:data:: input_path
   :value: 'ip_graph_23511_nodes_66863_edges_seed74.graphml'



# Node Importance Metrics for NetworkX Graph Visualization

To quantify node importance in a graph, these centrality metrics are commonly used:

1. Degree Centrality  
   Measures how many direct connections a node has.  
   Use this when you want a simple and fast "how connected is this node?" score.

2. Betweenness Centrality  
   Measures how often a node lies on shortest paths between other nodes.  
   Use this to find bridge or hub nodes that connect different parts of the network.

3. Closeness Centrality  
   Measures how close a node is, on average, to all other nodes.  
   Use this when you care about global reachability from each node.

4. Eigenvector Centrality  
   Gives higher scores to nodes connected to other high-scoring nodes.  
   Use this to capture influence through important neighbors.

5. PageRank  
   A link-based importance score, especially useful for directed graphs.  
   Use this when direction and recursive influence matter.

## Practical Recommendation

For highlighting the top 20 important nodes in an urban-geography-style graph visualization, start with **Betweenness Centrality**.  
It usually makes key transit/bridge nodes stand out clearly in the plot.

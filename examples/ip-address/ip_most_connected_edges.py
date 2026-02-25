import networkx as nx
import pandas as pd
import heapq


k, x = 40, 250
df = pd.read_csv("Dataset-Unicauca-Version2-87Atts.csv",
                 usecols=["Source.IP","Destination.IP", "Flow.Packets.s"])

# csv with source and destination IPs and the max flow rate of packets, with the number of times a flow occurred
df_max_flow = df.groupby(["Source.IP", "Destination.IP"], as_index=False).agg(flow=("Flow.Packets.s", "max"),
                                                                              count_flows=("Flow.Packets.s", "count"))

G = nx.DiGraph()
# Zip changes the rows from separate columns into a tuple allowing for edges to be added
G.add_weighted_edges_from(zip(df_max_flow["Source.IP"], df_max_flow["Destination.IP"], df_max_flow["flow"]),
                          weight="flow")

# Find the largest connected component and convert to undirected, stored in graph U
cc = max(nx.connected_components(G.to_undirected()), key=len)
U = G.subgraph(cc).to_undirected().copy()

# Store the degree of each node n in U as a dictionary {node: degree of node}
deg = {n: U.degree(n) for n in U}

# Store a tuple of (degree, node) in heap h
h = [(d,n) for n,d in deg.items()]
heapq.heapify(h)

# Every node in U is initially alive
alive = set(U.nodes())

# While the number of nodes in alive is > k continue to remove nodes,
# then decrement the degree of each neighbor of the removed node
# then update the heap with the new neighbor values
while len(alive) > k:
    d,n = heapq.heappop(h)
    if n not in alive or deg[n] != d: continue
    alive.remove(n)
    for nb in list(U.neighbors(n)):
        if nb in alive:
            deg[nb] -= 1
            heapq.heappush(h,(deg[nb],nb))
    U.remove_node(n)

# S is the set of nodes left in U, k nodes in K
S = set(U)

# Copy S into H
H = G.subgraph(S).copy()

# While the edges > x, remove the node with the highest degree
while H.number_of_edges() > x:
    # H.degree returns a list of tuples (node, degree), it finds the degree t[1] then stores the node in rm
    rm = max(H.degree, key=lambda t: t[1])[0]
    S.remove(rm)
    H = G.subgraph(S)

# Write out H as a graphml file
nx.write_graphml(H, f"ip_graph_edges_{x}_with_weights.graphml")
print("Nodes:", H.number_of_nodes(), "Edges:", H.number_of_edges())
import networkx as nx
import pandas as pd
import heapq


k, x = 40, 250
df = pd.read_csv("Dataset-Unicauca-Version2-87Atts.csv",
                 usecols=["Source.IP","Destination.IP", "Total.Length.of.Fwd.Packets"])

# Csv with source and destination IPs and the sum of the bytes from source to destination stored as flow
df_max_flow = df.groupby(["Source.IP", "Destination.IP"], as_index=False).agg(flow=("Total.Length.of.Fwd.Packets", "sum"))

G = nx.DiGraph()
# Zip changes the rows from separate columns into a tuple allowing for edges to be added
G.add_weighted_edges_from(zip(df_max_flow["Source.IP"], df_max_flow["Destination.IP"], df_max_flow["flow"]),
                          weight="flow")

# Find the largest connected component and convert to undirected, stored in graph U
cc = max(nx.weakly_connected_components(G), key=len)
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
    if n not in alive or deg[n] != d:
        continue
    alive.remove(n)
    for nb in list(U.neighbors(n)):
        if nb in alive:
            deg[nb] -= 1
            heapq.heappush(h,(deg[nb],nb))
    U.remove_node(n)

# S is the set of nodes left in U, k nodes in K
S = set(alive)

# Copy S into H
H = G.subgraph(S).copy()

# While the edges > x, remove the node with the highest degree
while H.number_of_edges() > x:
    # H.degree returns a list of tuples (node, degree), it finds the degree t[1] then stores the node in rm
    rm = max(H.degree, key=lambda t: t[1])[0]
    S.remove(rm)
    H = G.subgraph(S).copy()

# Compute flow-loss (importance value) for each node, using in & out edges
for n in H.nodes():

    incoming_flow = 0.0
    # H.in_edges returns a list of tuples (u, v, data), where data is a list of the attributes stored
    for src, des, data in H.in_edges(n, data=True):
        if "flow" in data:
            incoming_flow += data["flow"]
        else:
            incoming_flow += 0.0

    outgoing_flow = 0.0
    for src, des, data in H.out_edges(n, data=True):
        if "flow" in data:
            outgoing_flow += data["flow"]
        else:
            outgoing_flow += 0.0

    H.nodes[n]["flow_loss"] = incoming_flow + outgoing_flow

# Write out H as a graphml file
nx.write_graphml(H, f"ip_graph_{len(H.nodes())}_nodes_{len(H.edges())}_edges_with_flow.graphml")
print("Nodes:", H.number_of_nodes(), "Edges:", H.number_of_edges())
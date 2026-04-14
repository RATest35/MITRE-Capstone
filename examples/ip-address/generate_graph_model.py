import random
import networkx as nx
import pandas as pd

DATA_PATH = r""
SEED = 74
random.seed(SEED)

def bell_curve_probability(mean=0.1, std_dev=0.04):
    return max(0.0, min(1.0, random.gauss(mean, std_dev)))

df = pd.read_csv(DATA_PATH,
                 usecols=["Source.IP","Destination.IP", "Total.Length.of.Fwd.Packets", "Flow.Bytes.s"])

# Csv with source and destination IPs and the sum of the bytes from source to destination stored as flow
df_sum_flow = df.groupby(["Source.IP", "Destination.IP"], as_index=False).agg(flow=("Total.Length.of.Fwd.Packets", "sum"),
                                                                               bytes_per_sec=("Flow.Bytes.s", "median"))

G = nx.DiGraph()
for _, row in df_sum_flow.iterrows():
    G.add_edge(
        row["Source.IP"],
        row["Destination.IP"],
        flow=row["flow"],
        bytes_per_sec=row["bytes_per_sec"]
    )

# Find the largest connected component
cc = max(nx.weakly_connected_components(G), key=len)
U = G.subgraph(cc).copy()


# Compute flow-loss (importance value) for each node, using in & out edges
for n in sorted(U.nodes()):

    incoming_flow = 0.0
    # U.in_edges returns a list of tuples (u, v, data), where data is a list of the attributes stored
    for src, des, data in U.in_edges(n, data=True):
        if "flow" in data:
            incoming_flow += data["flow"]
        else:
            incoming_flow += 0.0

    outgoing_flow = 0.0
    for src, des, data in U.out_edges(n, data=True):
        if "flow" in data:
            outgoing_flow += data["flow"]
        else:
            outgoing_flow += 0.0

    U.nodes[n]["in_flow"] = incoming_flow
    U.nodes[n]["out_flow"] = outgoing_flow
    U.nodes[n]["flow_loss"] = incoming_flow + outgoing_flow

for n in U.nodes():
    U.nodes[n]["random_probability"] = bell_curve_probability()


# Write out H as a graphml file
nx.write_graphml(U, f"ip_graph_{len(U.nodes())}_nodes_{len(U.edges())}_edges_seed{SEED}.graphml")
print("Nodes:", U.number_of_nodes(), "Edges:", U.number_of_edges(), "Seed:", SEED)

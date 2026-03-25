import networkx as nx
import pandas as pd


df = pd.read_csv(r"C:\Users\ryant\Downloads\IP_dataset\cleaned_flows.csv",
                 usecols=["Source.IP","Destination.IP", "Total.Length.of.Fwd.Packets"])

# Csv with source and destination IPs and the sum of the bytes from source to destination stored as flow
df_sum_flow = df.groupby(["Source.IP", "Destination.IP"], as_index=False).agg(flow=("Total.Length.of.Fwd.Packets", "sum"))

G = nx.DiGraph()
# Zip changes the rows from separate columns into a tuple allowing for edges to be added
G.add_weighted_edges_from(zip(df_sum_flow["Source.IP"], df_sum_flow["Destination.IP"], df_sum_flow["flow"]),
                          weight="flow")

# Find the largest connected component
cc = max(nx.weakly_connected_components(G), key=len)
U = G.subgraph(cc).copy()


# Compute flow-loss (importance value) for each node, using in & out edges
for n in U.nodes():

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

# Write out U as a graphml file
nx.write_graphml(U, f"ip_graph_{len(U.nodes())}_nodes_edges_with_flow.graphml")
print("Nodes:", U.number_of_nodes(), "Edges:", U.number_of_edges())
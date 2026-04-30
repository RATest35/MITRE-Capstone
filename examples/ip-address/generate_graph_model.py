import random
import networkx as nx
import pandas as pd
from pathlib import Path


DATA_PATH = Path("cleaned_flows.csv")
SEED = 74
random.seed(SEED)



def bell_curve_probability(mean: float = 0.1, std_dev: float = 0.04) -> float:
    return max(0.0, min(1.0, random.gauss(mean, std_dev)))



def load_and_aggregate_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=[
            "Source.IP",
            "Destination.IP",
            "Total.Length.of.Fwd.Packets",
            "Flow.Bytes.s",
        ],
    )

    # Ensure numeric
    df["Total.Length.of.Fwd.Packets"] = pd.to_numeric(
        df["Total.Length.of.Fwd.Packets"], errors="coerce"
    )
    df["Flow.Bytes.s"] = pd.to_numeric(df["Flow.Bytes.s"], errors="coerce")

    df = df.dropna()

    # Aggregate flows
    df_agg = df.groupby(
        ["Source.IP", "Destination.IP"], as_index=False
    ).agg(
        flow=("Total.Length.of.Fwd.Packets", "sum"),
        bytes_per_sec=("Flow.Bytes.s", "median"),
    )

    return df_agg



def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.from_pandas_edgelist(
        df,
        source="Source.IP",
        target="Destination.IP",
        edge_attr=["flow", "bytes_per_sec"],
        create_using=nx.DiGraph(),
    )
    return G


# ---------------------------------------------------------
# Extract largest component
# ---------------------------------------------------------
def largest_component(G: nx.DiGraph) -> nx.DiGraph:
    cc = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(cc).copy()


# ---------------------------------------------------------
# Compute node flow metrics
# ---------------------------------------------------------
def add_flow_metrics(G: nx.DiGraph) -> nx.DiGraph:
    for n in G.nodes():
        in_flow = sum(data.get("flow", 0.0) for _, _, data in G.in_edges(n, data=True))
        out_flow = sum(data.get("flow", 0.0) for _, _, data in G.out_edges(n, data=True))

        G.nodes[n]["in_flow"] = float(in_flow)
        G.nodes[n]["out_flow"] = float(out_flow)
        G.nodes[n]["flow_loss"] = float(in_flow + out_flow)

    return G


# ---------------------------------------------------------
# Add node risk probability
# ---------------------------------------------------------
def add_random_probability(G: nx.DiGraph) -> nx.DiGraph:
    for n in G.nodes():
        G.nodes[n]["random_probability"] = bell_curve_probability()
    return G


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main():
    df = load_and_aggregate_data(DATA_PATH)

    G = build_graph(df)
    G = largest_component(G)

    G = add_flow_metrics(G)
    G = add_random_probability(G)

    output_path = Path(
        f"ip_graph_{G.number_of_nodes()}_nodes_{G.number_of_edges()}_edges_seed{SEED}.graphml"
    )

    nx.write_graphml(G, output_path)

    print("Graph processed successfully")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Seed:", SEED)
    print("Saved to:", output_path)


# ---------------------------------------------------------
if __name__ == "__main__":
    main()
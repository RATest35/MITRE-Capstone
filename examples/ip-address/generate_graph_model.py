import random
import networkx as nx
import pandas as pd
from pathlib import Path


DATA_PATH = Path("cleaned_flows.csv")
SEED = 74
random.seed(SEED)



def bell_curve_probability(mean: float = 0.1, std_dev: float = 0.04) -> float:
    """Generate a bounded random probability from a normal distribution.

    :param mean: Mean of the Gaussian distribution.
    :param std_dev: Standard deviation of the Gaussian distribution.
    :return: Float probability clipped to range [0, 1].

    Notes:
    - Uses random.gauss for sampling
    - Values are clipped to ensure valid probability bounds
    """
    return max(0.0, min(1.0, random.gauss(mean, std_dev)))


def load_and_aggregate_data(path: Path) -> pd.DataFrame:
    """Load CSV data and aggregate flows between IP pairs.

    :param path: Path to cleaned CSV file.
    :return: Aggregated DataFrame with flow and bytes_per_sec columns.

    Notes:
    - Converts numeric columns safely
    - Drops rows with invalid/missing values
    - Aggregates total flow as the sum of packet lengths.
    - Aggregates bytes_per_sec as the median for robustness.
    """
    df = pd.read_csv(
        path,
        usecols=[
            "Source.IP",
            "Destination.IP",
            "Total.Length.of.Fwd.Packets",
            "Flow.Bytes.s",
        ],
    )

    # Ensure numeric types (invalid values → NaN)
    df["Total.Length.of.Fwd.Packets"] = pd.to_numeric(
        df["Total.Length.of.Fwd.Packets"], errors="coerce"
    )
    df["Flow.Bytes.s"] = pd.to_numeric(df["Flow.Bytes.s"], errors="coerce")

    # Remove invalid rows
    df = df.dropna()

    # Aggregate flows into edge-level representation
    df_agg = df.groupby(
        ["Source.IP", "Destination.IP"], as_index=False
    ).agg(
        flow=("Total.Length.of.Fwd.Packets", "sum"),
        bytes_per_sec=("Flow.Bytes.s", "median"),  # median reduces outlier impact
    )

    return df_agg


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Construct a directed graph from aggregated flow data.

    :param df: DataFrame containing edge-level data.
    :return: Directed NetworkX graph with edge attributes.

    Notes:
    - Nodes represent IP addresses.
    - Edges represent communication between IPs.
    - Edge attributes include flow and bytes_per_sec.
    """
    G = nx.from_pandas_edgelist(
        df,
        source="Source.IP",
        target="Destination.IP",
        edge_attr=["flow", "bytes_per_sec"],
        create_using=nx.DiGraph(),
    )
    return G


def largest_component(G: nx.DiGraph) -> nx.DiGraph:
    """Extract the largest weakly connected component.

    :param G: Input directed graph.
    :return: Subgraph containing the largest component.

    Notes:
    - Uses weak connectivity (ignores edge direction)
    - Helps remove isolated or insignificant subgraphs
    """
    cc = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(cc).copy()


def add_flow_metrics(G: nx.DiGraph) -> nx.DiGraph:
    """Compute flow-based node metrics.

    :param G: Input graph.
    :return: Graph with node attributes:
        - in_flow
        - out_flow
        - flow_loss

    Notes:
    - in_flow: sum of incoming edge flow
    - out_flow: sum of outgoing edge flow
    - flow_loss: total traffic through node
    """
    for n in G.nodes():
        in_flow = sum(data.get("flow", 0.0) for _, _, data in G.in_edges(n, data=True))
        out_flow = sum(data.get("flow", 0.0) for _, _, data in G.out_edges(n, data=True))

        G.nodes[n]["in_flow"] = float(in_flow)
        G.nodes[n]["out_flow"] = float(out_flow)
        G.nodes[n]["flow_loss"] = float(in_flow + out_flow)

    return G


def add_random_probability(G: nx.DiGraph) -> nx.DiGraph:
    """Assign random probability scores to nodes.

    :param G: Input graph.
    :return: Graph with 'random_probability' node attribute.

    Notes:
    - Uses bell_curve_probability for realistic distribution
    - Simulates uncertainty or risk likelihood per node
    """
    for n in G.nodes():
        G.nodes[n]["random_probability"] = bell_curve_probability()
    return G


def main():
    """Run full pipeline to build and export graph.

    Steps:
    1. Load and aggregate CSV data
    2. Build directed graph
    3. Extract largest connected component
    4. Compute node flow metrics
    5. Assign random probabilities
    6. Save graph as GraphML file

    :return: None
    """
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

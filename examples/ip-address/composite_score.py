import networkx as nx
import time

# Change path to the graphml file
INPUT_PATH = 'ip_graph_23511_nodes_edges_with_flow.graphml'
OUTPUT_PATH = 'composite_risk.graphml'

def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return print(f'{value} is not able to be converted to float')

def load_graph(file_name):
    G = nx.read_graphml(file_name)
    return G


def add_total_flow(G, flow_attr='flow'):
    for n in G.nodes():
        in_flow = sum(to_float(data.get(flow_attr, 0.0)) for _, _, data in G.in_edges(n, data=True))
        out_flow = sum(to_float(data.get(flow_attr, 0.0)) for _, _, data in G.out_edges(n, data=True))
        G.nodes[n]['in_flow'] = in_flow
        G.nodes[n]['out_flow'] = out_flow
        G.nodes[n]['flow_loss'] = in_flow + out_flow

    return G


def add_pagerank(G):
    pr = nx.pagerank(G, weight='flow')
    for n, val in pr.items():
        G.nodes[n]['pagerank'] = val
    return G


def add_distance_from_strength(G, strength_attr="bytes_per_sec", distance_attr="distance", epsilon=1e-9):
    """
    Convert an edge strength attribute into a distance attribute:
        distance = 1 / (strength + epsilon)
        epsilon is added to avoid division by zero.
    """
    for u, v, data in G.edges(data=True):
        strength = to_float(data.get(strength_attr, 0.0))
        data[distance_attr] = 1.0 / (strength + epsilon)

    return G


def add_weighted_betweenness(G, strength_attr="bytes_per_sec", distance_attr="distance", normalized=True):
    """
    Uses inverse of strength_attr as distance because higher traffic
    should act like a stronger/shorter connection.
    """
    add_distance_from_strength(G, strength_attr=strength_attr, distance_attr=distance_attr)
    bet = nx.betweenness_centrality(G, weight=distance_attr, normalized=normalized)

    for n, score in bet.items():
        G.nodes[n]["weighted_betweenness"] = score

    return G


def add_composite_score(G):
    for n, data in G.nodes(data=True):

        risk = 1
        flow = data.get("flow_loss", 0)
        bet = data.get("weighted_betweenness", 0)
        pr = data.get("pagerank", 0)

        importance = flow + bet + pr

        data["importance"] = importance
        data["composite_score"] = risk * importance

    return G


def compute_composite_score(input_file, output_file):
    G = load_graph(input_file)
    add_total_flow(G)
    add_weighted_betweenness(G)
    add_pagerank(G)
    add_composite_score(G)

    nx.write_graphml(G, output_file)

    return G




if __name__ == "__main__":

    input_path = INPUT_PATH
    output_path = OUTPUT_PATH

    start = time.perf_counter()

    G = compute_composite_score(input_path, output_path)

    end = time.perf_counter()

    print("\nGraph processed successfully")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print(f"Total runtime: {end - start:.4f} seconds")

    top_nodes = sorted(
        G.nodes(data=True),
        key=lambda x: x[1].get("composite_score", 0),
        reverse=True
    )[:10]

    print("\nTop 10 Critical Nodes:")
    for node, data in top_nodes:
        print(
            node,
            "Composite:", data.get("composite_score", 0),
            "Risk:", data.get("risk_percent", 0),
        )

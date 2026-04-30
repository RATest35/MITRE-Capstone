import networkx as nx
import time

# Change path to the graphml file
INPUT_PATH = 'ip_graph_23511_nodes_66863_edges_seed74.graphml'
OUTPUT_PATH = 'composite_score_with_bytes_per_sec.graphml'

def to_float(value):
    """Convert a value to float if possible

    :param value: Input value to convert
    :return: Float value or None if conversion fails
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return print(f'{value} is not able to be converted to float')


def load_graph(file_name):
    """Load a GraphML file into a NetworkX graph

    :param file_name: Path to the GraphML file
    :return: Loaded NetworkX graph
    """
    G = nx.read_graphml(file_name)
    return G


def add_total_flow(G, flow_attr='flow'):
    """Compute total in/out flow and flow loss for each node

    :param G: Input graph
    :param flow_attr: Edge attribute representing flow
    :return: Graph with added node attributes (in_flow, out_flow, flow_loss)
    """
    for n in G.nodes():
        in_flow = sum(to_float(data.get(flow_attr)) for _, _, data in G.in_edges(n, data=True))
        out_flow = sum(to_float(data.get(flow_attr)) for _, _, data in G.out_edges(n, data=True))
        G.nodes[n]['in_flow'] = in_flow
        G.nodes[n]['out_flow'] = out_flow
        G.nodes[n]['flow_loss'] = in_flow + out_flow

    return G


def normalize_multiple_attributes(G, attr_specs):
    """Normalize multiple node attributes to specified ranges

    :param G: Input graph
    :param attr_specs: List of tuples (attr_name, normalized_name, new_min, new_max)
    :return: Graph with normalized attributes added
    """
    values_dict = {attr: [] for attr, _, _, _ in attr_specs}

    for _, data in G.nodes(data=True):
        for attr, _, _, _ in attr_specs:
            values_dict[attr].append(to_float(data.get(attr, 0.0)))

    min_max = {attr: (min(values), max(values)) for attr, values in values_dict.items()}

    for _, data in G.nodes(data=True):
        for attr, norm_attr, new_min, new_max in attr_specs:
            val = to_float(data.get(attr, 0.0))
            min_val, max_val = min_max[attr]

            if max_val == min_val:
                data[norm_attr] = new_min
            else:
                normalized_01 = (val - min_val) / (max_val - min_val)
                data[norm_attr] = new_min + normalized_01 * (new_max - new_min)

    return G


def add_pagerank(G):
    """Compute PageRank scores using edge weights

    :param G: Input graph
    :return: Graph with 'pagerank' node attribute
    """
    pr = nx.pagerank(G, weight='bytes_per_sec')
    for n, val in pr.items():
        G.nodes[n]['pagerank'] = val
    return G


def add_distance_from_strength(G, strength_attr="bytes_per_sec", distance_attr="distance", epsilon=1e-9):
    """Convert edge strength to distance

    :param G: Input graph
    :param strength_attr: Edge attribute representing strength
    :param distance_attr: Name of distance attribute to create
    :param epsilon: Small constant to avoid division by zero
    :return: Graph with distance attribute added to edges
    """
    for u, v, data in G.edges(data=True):
        strength = to_float(data.get(strength_attr))
        data[distance_attr] = 1.0 / (strength + epsilon)

    return G


def add_weighted_betweenness(G, strength_attr="bytes_per_sec", distance_attr="distance"):
    """Compute weighted betweenness centrality

    :param G: Input graph
    :param strength_attr: Edge attribute representing strength
    :param distance_attr: Edge attribute used for distance
    :return: Graph with 'weighted_betweenness' node attribute
    """
    add_distance_from_strength(G, strength_attr=strength_attr, distance_attr=distance_attr)
    bet = nx.betweenness_centrality(G, weight=distance_attr)

    for n, score in bet.items():
        G.nodes[n]["weighted_betweenness"] = score

    return G


def add_composite_score(G):
    """Compute composite score based on risk and importance

    :param G: Input graph with normalized attributes
    :return: Graph with 'importance' and 'composite_score' attributes
    """
    for _, data in G.nodes(data=True):
        risk = to_float(data.get("random_probability"))
        flow = to_float(data.get("flow_loss_norm"))
        bet = to_float(data.get("weighted_betweenness_norm"))
        pr = to_float(data.get("pagerank_norm"))

        importance = flow + bet + pr

        data["importance"] = importance
        data["composite_score"] = risk * importance

    return G


def compute_composite_score(input_file, output_file):
    """Run full pipeline to compute composite scores

    :param input_file: Path to input GraphML file
    :param output_file: Path to save output GraphML file
    :return: Processed graph with computed attributes
    """
    G = load_graph(input_file)

    add_total_flow(G)
    add_weighted_betweenness(G)
    add_pagerank(G)

    normalize_multiple_attributes(G, [
        ("in_flow", "in_flow_norm", 0, 1),
        ("out_flow", "out_flow_norm", 0, 1),
        ("flow_loss", "flow_loss_norm", 0, 1),
        ("weighted_betweenness", "weighted_betweenness_norm", 0, 5),
        ("pagerank", "pagerank_norm", 0, 5),
    ])

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
        key=lambda x: x[1].get("composite_score"),
        reverse=True
    )[:10]

    print("\nTop 10 Critical Nodes:")
    for node, data in top_nodes:
        print(
            node,
            "Composite:", data.get("composite_score"),
            "Risk:", data.get("random_probability"),
        )

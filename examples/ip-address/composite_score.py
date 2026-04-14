import networkx as nx
import time

# Change path to the graphml file
INPUT_PATH = ''
OUTPUT_PATH = ''

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
        in_flow = sum(to_float(data.get(flow_attr)) for _, _, data in G.in_edges(n, data=True))
        out_flow = sum(to_float(data.get(flow_attr)) for _, _, data in G.out_edges(n, data=True))
        G.nodes[n]['in_flow'] = in_flow
        G.nodes[n]['out_flow'] = out_flow
        G.nodes[n]['flow_loss'] = in_flow + out_flow

    return G

def normalize_multiple_attributes(G, attr_specs):
    """
    attr_specs: list of tuples
        (attr_name, normalized_name, new_min, new_max)
    """

    # Step 1: collect values for each attribute
    values_dict = {attr: [] for attr, _, _, _ in attr_specs}

    for _, data in G.nodes(data=True):
        for attr, _, _, _ in attr_specs:
            values_dict[attr].append(to_float(data.get(attr, 0.0)))

    # Step 2: compute min/max for each attribute
    min_max = {}
    for attr, values in values_dict.items():
        min_max[attr] = (min(values), max(values))

    # Step 3: normalize all attributes in one pass
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
    pr = nx.pagerank(G, weight='bytes_per_sec')
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
        strength = to_float(data.get(strength_attr))
        data[distance_attr] = 1.0 / (strength + epsilon)

    return G


def add_weighted_betweenness(G, strength_attr="bytes_per_sec", distance_attr="distance"):
    """
    Uses inverse of strength_attr as distance because higher traffic
    should act like a stronger/shorter connection.
    """
    add_distance_from_strength(G, strength_attr=strength_attr, distance_attr=distance_attr)
    bet = nx.betweenness_centrality(G, weight=distance_attr)

    for n, score in bet.items():
        G.nodes[n]["weighted_betweenness"] = score

    return G


def add_composite_score(G):
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

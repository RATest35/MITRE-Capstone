import networkx as nx
import matplotlib.pyplot as plt
from typing import Literal


MetricName = Literal["degree", "betweenness", "closeness", "pagerank"]


def compute_node_importance(graph: nx.Graph, metric_name: MetricName) -> dict[str, float]:
    metric_functions = {
        "degree": nx.degree_centrality,
        "betweenness": nx.betweenness_centrality,
        "closeness": nx.closeness_centrality,
        "pagerank": nx.pagerank,
    }
    if metric_name not in metric_functions:
        available = ", ".join(metric_functions.keys())
        raise ValueError(f"Unknown metric '{metric_name}'. Available metrics: {available}")

    return metric_functions[metric_name](graph)


def get_top_n_nodes(scores: dict[str, float], n: int) -> set[str]:
    sorted_nodes = sorted(scores, key=scores.get, reverse=True)
    return set(sorted_nodes[:n])


def draw_graph_with_highlights(graph: nx.Graph, highlighted_nodes: set[str], title: str) -> None:
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)
    node_colors = ["red" if node in highlighted_nodes else "skyblue" for node in graph.nodes]

    nx.draw(
        graph,
        pos,
        node_color=node_colors,
        node_size=30,
        width=0.5,
        with_labels=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main() -> None:
    graph = nx.read_graphml("Urban Geography.graphml")
    scores = compute_node_importance(graph, "pagerank")
    top_nodes = get_top_n_nodes(scores, n=10)

    title = f"Urban Geography Graph (METRIC_NAME top TOP_N in red)"
    draw_graph_with_highlights(graph, top_nodes, title)


if __name__ == "__main__":
    main()

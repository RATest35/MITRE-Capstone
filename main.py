import networkx as nx
import matplotlib.pyplot as plt


def main() -> None:
    graph_path = "Urban Geography.graphml"
    graph = nx.read_graphml(graph_path)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(
        graph,
        pos,
        node_size=30,
        width=0.5,
        with_labels=False,
    )
    plt.title("Urban Geography Graph")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

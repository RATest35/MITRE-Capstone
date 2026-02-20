
# directed, labeled

from pathlib import Path
import csv
import kagglehub


def download_dataset() -> Path:
    """Download the dataset and return its local path."""
    dataset_path: str = kagglehub.dataset_download(
        "jsrojas/ip-network-traffic-flows-labeled-with-87-apps"
    )
    return Path(dataset_path)


def count_nodes(csv_path: Path) -> int:
    """Count unique IP address nodes from source and destination columns."""
    nodes: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            nodes.update([row["Source.IP"], row["Destination.IP"]])
    return len(nodes)


def main() -> None:
    """Run dataset download and print node count."""
    dataset_path: Path = download_dataset()
    csv_path: Path = next(dataset_path.glob("*.csv"))
    node_count: int = count_nodes(csv_path)

    print("Path to dataset files:", dataset_path)
    print("Node count:", node_count)


if __name__ == "__main__":
    main()


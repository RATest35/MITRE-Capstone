"""
data_loader.py
--------------
SRP: responsible only for acquiring the raw CSV dataset.

Two strategies are supported:
  1. ``KaggleDataLoader``  — downloads the dataset via the kagglehub API.
  2. ``LocalDataLoader``   — reads from a path already present on disk.

Both implement the ``AbstractDataLoader`` interface (DIP / LSP) so that
``GraphBuilder`` never cares *where* the data came from.
"""

from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Abstract interface  (Dependency-Inversion anchor)
# ══════════════════════════════════════════════════════════════════════════════

class AbstractDataLoader(ABC):
    """
    Interface that every data-loader must satisfy.

    ``GraphBuilder`` depends on this abstraction, never on a concrete loader.
    """

    @abstractmethod
    def load(self, columns: list[str] | None = None) -> pd.DataFrame:
        """
        Return a ``DataFrame`` containing at minimum the requested *columns*.

        Parameters
        ----------
        columns:
            Optional list of column names to keep.  ``None`` means keep all.
        """


# ══════════════════════════════════════════════════════════════════════════════
# Concrete loaders
# ══════════════════════════════════════════════════════════════════════════════

class KaggleDataLoader(AbstractDataLoader):
    """
    Downloads the dataset from Kaggle via ``kagglehub`` and returns a DataFrame.

    Parameters
    ----------
    dataset_id:
        Kaggle dataset slug, e.g.
        ``"jsrojas/ip-network-traffic-flows-labeled-with-87-apps"``.
    """

    def __init__(self, dataset_id: str) -> None:
        self._dataset_id = dataset_id

    def load(self, columns: list[str] | None = None) -> pd.DataFrame:
        try:
            import kagglehub  # optional dependency — only needed for this loader
        except ImportError as exc:
            raise ImportError(
                "kagglehub is required for KaggleDataLoader. "
                "Install it with: pip install kagglehub"
            ) from exc

        logger.info("Downloading Kaggle dataset: %s", self._dataset_id)
        dataset_path = Path(kagglehub.dataset_download(self._dataset_id))
        csv_path = next(dataset_path.glob("*.csv"))
        logger.info("Dataset downloaded to %s", csv_path)
        return self._read_csv(csv_path, columns)

    # ------------------------------------------------------------------ helper

    @staticmethod
    def _read_csv(path: Path, columns: list[str] | None) -> pd.DataFrame:
        kwargs = {"usecols": columns} if columns else {}
        return pd.read_csv(path, **kwargs)


class LocalDataLoader(AbstractDataLoader):
    """
    Loads a CSV file that already exists on the local filesystem.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    """

    def __init__(self, csv_path: str | Path) -> None:
        self._csv_path = Path(csv_path)

    def load(self, columns: list[str] | None = None) -> pd.DataFrame:
        logger.info("Loading local CSV from %s", self._csv_path)
        kwargs = {"usecols": columns} if columns else {}
        return pd.read_csv(self._csv_path, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Utility: unique-node counter (preserved from original main.py)
# ══════════════════════════════════════════════════════════════════════════════

def count_unique_nodes(
    csv_path: Path,
    source_col: str = "Source.IP",
    dest_col: str = "Destination.IP",
) -> int:
    """
    Count the number of unique IP addresses (nodes) in the CSV.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    source_col / dest_col:
        Column names for source and destination IPs.

    Returns
    -------
    int
        Number of unique IP addresses.
    """
    nodes: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            nodes.update([row[source_col], row[dest_col]])
    return len(nodes)

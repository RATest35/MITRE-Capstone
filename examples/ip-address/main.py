"""
main.py
-------
Composition Root: the single entry point that wires the entire pipeline.

This file is the *only* place where concrete classes are instantiated and
assembled.  All other modules depend exclusively on abstractions, keeping
the codebase fully testable and extensible.

Pipeline modes
--------------
1. **Score-only** (default) — reads an existing ``.graphml`` produced by
   ``GraphBuilder``, runs the enrichment + scoring pipeline, and writes a
   new enriched ``.graphml``.

2. **Build-then-score** — downloads (Kaggle) or reads (local) the raw CSV,
   builds the graph, then immediately scores it.  Activated by passing a
   ``--csv`` argument on the command line.

Usage examples
--------------
Score an existing graphml::

    python main.py

Build from a local CSV and score::

    python main.py --csv path/to/cleaned_flows.csv

Build from Kaggle and score::

    python main.py --kaggle
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import DEFAULT_CONFIG, PipelineConfig
from composite_scorer import CompositeScorer
from data_loader import KaggleDataLoader, LocalDataLoader
from graph_builder import GraphBuilder
from graph_io import GraphIO
from utils import configure_logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IP-graph composite risk scorer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help="Path to a local cleaned_flows CSV file.  "
             "Triggers graph-build mode before scoring.",
    )
    source.add_argument(
        "--kaggle",
        action="store_true",
        default=False,
        help="Download the dataset from Kaggle before building and scoring.",
    )

    parser.add_argument(
        "--input",
        metavar="PATH",
        default=DEFAULT_CONFIG.input_graphml,
        help="Input .graphml file (score-only mode).",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=DEFAULT_CONFIG.output_graphml,
        help="Output .graphml file for the enriched graph.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_CONFIG.top_n_nodes,
        help="Number of top critical nodes to display.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    return parser.parse_args(argv)


# ══════════════════════════════════════════════════════════════════════════════
# Build-mode helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph_from_csv(csv_path: str, config: PipelineConfig, output_graphml: str) -> None:
    """
    Load a CSV, build the DiGraph, and save it as a GraphML file ready for
    the scoring pipeline.
    """
    loader = LocalDataLoader(csv_path)
    df = loader.load(
        columns=[config.col_source_ip, config.col_dest_ip, config.col_fwd_bytes]
    )
    builder = GraphBuilder(config)
    G = builder.build(df)

    io = GraphIO()
    io.save(G, output_graphml)
    logger.info("Graph saved to %s — ready for scoring.", output_graphml)


def _build_graph_from_kaggle(config: PipelineConfig, output_graphml: str) -> None:
    """
    Download the Kaggle dataset, build the DiGraph, and save as GraphML.
    """
    loader = KaggleDataLoader(config.kaggle_dataset)
    df = loader.load(
        columns=[config.col_source_ip, config.col_dest_ip, config.col_fwd_bytes]
    )
    builder = GraphBuilder(config)
    G = builder.build(df)

    io = GraphIO()
    io.save(G, output_graphml)
    logger.info("Graph saved to %s — ready for scoring.", output_graphml)


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(scorer: CompositeScorer, G, top: int) -> None:
    """Print a brief summary and the top-N critical nodes to stdout."""
    print("\n── Graph processed successfully ──────────────────────────────")
    print(f"  Nodes : {G.number_of_nodes()}")
    print(f"  Edges : {G.number_of_edges()}")

    top_nodes = scorer.top_nodes(G, n=top)
    print(f"\n  Top {top} Critical Nodes:")
    print(f"  {'Node':<22} {'Composite Score':>16}  {'Importance':>12}")
    print("  " + "─" * 70)
    for node, data in top_nodes:
        print(
            f"  {str(node):<22}"
            f"  {data.get('composite_score', 0):>16.6f}"
            f"  {data.get('importance', 0):>12.6f}"
        )
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

# Directory that contains main.py — used to resolve default relative paths.
_SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_path(p: str) -> Path:
    """
    Resolve a path so the pipeline works from any working directory.

    Rules (in order):
    1. Absolute path  → used as-is.
    2. Bare filename (no directory component, e.g. ``composite_risk.graphml``)
       → resolved relative to the directory that contains ``main.py``, so
       default filenames always land next to the script regardless of CWD.
    3. Relative path with directory components (e.g. ``data/graph.graphml``)
       → resolved relative to the current working directory, respecting the
       caller's intent.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    # A bare filename has no parent directory (parent == '.')
    if path.parent == Path("."):
        return (_SCRIPT_DIR / path).resolve()
    # Caller supplied an explicit relative path — honour it from CWD
    return path.resolve()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    config = DEFAULT_CONFIG

    # Resolve input/output paths relative to this script's directory so the
    # pipeline works whether the user cds into the folder or runs it from
    # the project root.
    input_path = str(_resolve_path(args.input))
    output_path = str(_resolve_path(args.output))

    # ── Optional: build graph from raw data first ──────────────────────────
    if args.csv:
        logger.info("Build mode: local CSV → %s", args.csv)
        _build_graph_from_csv(args.csv, config, input_path)
    elif args.kaggle:
        logger.info("Build mode: Kaggle dataset → %s", input_path)
        _build_graph_from_kaggle(config, input_path)

    # ── Score the (existing or just-built) graphml ─────────────────────────
    if not Path(input_path).exists():
        logger.error(
            "Input file not found: %s\n"
            "Run with --csv or --kaggle to build the graph first, "
            "or provide --input pointing to an existing .graphml file.",
            input_path,
        )
        sys.exit(1)

    logger.info("Score mode: %s → %s", input_path, output_path)

    # Composition root: inject dependencies into CompositeScorer
    scorer = CompositeScorer(config=config)
    G = scorer.run(input_path, output_path)

    _print_summary(scorer, G, top=args.top)


if __name__ == "__main__":
    main()

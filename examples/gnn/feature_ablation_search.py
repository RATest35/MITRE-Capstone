from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from composite_dataset import DEFAULT_EXTENDED_GROUPS


BASE_COMMAND: tuple[str, ...] = (
    "python",
    "examples/gnn/train_composite_score_gnn.py",
    "--graphml-path",
    "examples/gnn/composite_risk.graphml",
    "--epochs",
    "6",
    "--batch-size",
    "1024",
    "--eval-batch-size",
    "1024",
    "--hidden-dim",
    "256",
    "--num-layers",
    "2",
    "--dropout",
    "0.1",
    "--learning-rate",
    "3e-4",
    "--weight-decay",
    "1e-4",
    "--train-ratio",
    "0.7",
    "--val-ratio",
    "0.15",
    "--num-hops",
    "1",
    "--max-in-neighbors",
    "32",
    "--max-out-neighbors",
    "32",
    "--num-workers",
    "4",
    "--prefetch-factor",
    "2",
    "--selection-metric",
    "ndcg_5pct",
    "--patience",
    "3",
    "--train-seed",
    "42",
    "--feature-set",
    "extended",
    "--group-by-prefix",
    "24",
    "--split-bucket-size",
    "32",
    "--weight-mode",
    "linear",
    "--weight-scale",
    "4.0",
    "--ranking-loss-weight",
    "0.2",
    "--ranking-margin",
    "0.02",
    "--ranking-pairs",
    "128",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run feature ablation experiments.")
    parser.add_argument("--output-root", type=Path, default=Path("examples/gnn/experiments/feature_ablation"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, choices=["leave_one_out", "only_one"], required=True)
    parser.add_argument("--groups", type=str, default=",".join(DEFAULT_EXTENDED_GROUPS))
    return parser.parse_args()


def build_group_sets(mode: str, groups: tuple[str, ...]) -> list[tuple[str, tuple[str, ...]]]:
    """Build group subsets for one ablation mode."""
    if mode == "leave_one_out":
        return [
            (f"drop_{group_name}", tuple(value for value in groups if value != group_name))
            for group_name in groups
        ]
    return [(f"only_{group_name}", (group_name,)) for group_name in groups]


def run_one(output_dir: Path, device: str, feature_groups: tuple[str, ...]) -> dict[str, float]:
    """Run one training command and return test metrics."""
    command = [
        *BASE_COMMAND,
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--feature-groups",
        ",".join(feature_groups),
    ]
    subprocess.run(command, check=True)
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    return metrics["test_metrics"]


def main() -> None:
    """Run ablation jobs and write a summary."""
    arguments = parse_args()
    groups = tuple(part.strip() for part in arguments.groups.split(",") if part.strip())
    experiment_sets = build_group_sets(arguments.mode, groups)
    arguments.output_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    for name, feature_groups in experiment_sets:
        output_dir = arguments.output_root / name
        test_metrics = run_one(output_dir, arguments.device, feature_groups)
        summary.append(
            {
                "name": name,
                "feature_groups": list(feature_groups),
                **test_metrics,
            }
        )

    summary_path = arguments.output_root / f"{arguments.mode}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

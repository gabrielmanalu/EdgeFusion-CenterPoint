"""
Accuracy / latency / power Pareto frontier.

Aggregates results from all compression experiments and plots the
accuracy vs latency Pareto frontier with power encoded as bubble size.
The chosen operating point is annotated with the constraint budget.

Usage:
    python compression/pareto.py \
        --results-dir compression/results/ \
        --latency-csv deployment/benchmarks/results/latency.csv \
        --map-budget  47.0 \
        --latency-budget 50.0 \
        --power-budget 15.0
"""

import argparse
# import json
from pathlib import Path

# import matplotlib.pyplot as plt
import mlflow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="compression/results/")
    p.add_argument("--latency-csv", default=None)
    p.add_argument(
        "--map-budget", type=float, default=None, help="Minimum acceptable mAP"
    )
    p.add_argument(
        "--latency-budget",
        type=float,
        default=None,
        help="Maximum acceptable p99 latency (ms)",
    )
    p.add_argument(
        "--power-budget", type=float, default=15.0, help="Power envelope (W)"
    )
    p.add_argument("--out", default="compression/results/pareto.png")
    return p.parse_args()


def load_all_variants(results_dir: Path) -> list:
    """
    Walk results_dir for experiment JSON files and build a list of
    {name, mAP, NDS, latency_ms, power_W} dicts.
    """
    # TODO: discover ptq/, qat/, pruning/, distillation/ JSON files
    raise NotImplementedError


def is_pareto_optimal(variants: list) -> list:
    """Return a boolean mask indicating which variants lie on the Pareto front."""
    # TODO: standard 2-objective Pareto filtering (max mAP, min latency)
    raise NotImplementedError


def plot_pareto(
    variants: list, pareto_mask: list, budgets: dict, out_path: str
) -> None:
    """
    Scatter plot of mAP vs latency_ms.
    Bubble size encodes power_W; Pareto-optimal points are highlighted.
    Operating point (satisfies all budgets) is annotated with a star.
    """
    # TODO: matplotlib scatter; draw budget contour lines; annotate operating point
    raise NotImplementedError


def main() -> None:
    args = parse_args()

    budgets = {
        "mAP": args.map_budget,
        "latency": args.latency_budget,
        "power": args.power_budget,
    }

    with mlflow.start_run(run_name="pareto"):
        variants = load_all_variants(Path(args.results_dir))
        pareto_mask = is_pareto_optimal(variants)
        plot_pareto(variants, pareto_mask, budgets, args.out)
        mlflow.log_artifact(args.out)
        print(f"[pareto] Frontier saved to {args.out}")


if __name__ == "__main__":
    main()

"""
Structured channel pruning of the CenterPoint 2D backbone.

For each sparsity ratio in a sweep, prune backbone Conv2d channels using
L1-norm importance scoring, then fine-tune to recover accuracy. Each
pruned variant is evaluated and becomes a Pareto candidate.

Usage:
    python compression/pruning.py \
        --config     configs/centerpoint_pillar02_circlenms_nus.py \
        --checkpoint /workspace/data/centerpoint/centerpoint_nuscenes.pth \
        --ratios 0.25 0.40 0.55
"""

import argparse
from pathlib import Path

import mlflow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.40, 0.55])
    p.add_argument("--finetune-epochs", type=int, default=3)
    p.add_argument("--out", default="compression/results/pruning/")
    return p.parse_args()


def prune_backbone(model, sparsity_ratio: float):
    """
    Apply structured channel pruning to backbone Conv2d layers.
    Uses torch-pruning L1NormImportance scoring.
    """
    # TODO:
    #   import torch_pruning as tp
    #   pruner = tp.MagnitudePruner(model, ..., sparsity=sparsity_ratio)
    #   pruner.step()
    raise NotImplementedError


def finetune(model, loader_train, epochs: int, lr: float) -> dict:
    """Fine-tune a pruned model and return val metrics."""
    # TODO: standard training loop (fewer epochs than QAT)
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for ratio in args.ratios:
        with mlflow.start_run(run_name=f"prune_r{int(ratio * 100)}"):
            mlflow.log_params(
                {
                    "method": "structured_pruning",
                    "sparsity_ratio": ratio,
                    "finetune_epochs": args.finetune_epochs,
                }
            )
            # TODO: load → prune → finetune → eval → log → save ckpt
            raise NotImplementedError


if __name__ == "__main__":
    main()

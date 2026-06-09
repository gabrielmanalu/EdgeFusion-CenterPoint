"""
Per-layer quantization sensitivity analysis.

For each layer in the backbone and head, quantize only that layer to INT8
and measure the resulting mAP drop on a fast val subset. Produces a ranked
sensitivity table that drives the mixed-precision strategy for QAT.

Usage:
    python compression/sensitivity.py \
        --config     configs/centerpoint_pillar02_circlenms_nus.py \
        --checkpoint /workspace/data/centerpoint/centerpoint_nuscenes.pth \
        --fast
"""

import argparse
import json
from pathlib import Path

import mlflow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--fast", action="store_true", help="Use a 500-sample val subset for speed"
    )
    p.add_argument("--out", default="compression/results/sensitivity.json")
    return p.parse_args()


def get_quantizable_layers(model) -> list:
    """Return (name, module) pairs eligible for per-layer INT8 quantization."""
    # TODO: filter for nn.Conv2d and nn.Linear in backbone / neck / head
    raise NotImplementedError


def eval_with_layer_quantized(model, layer_name: str, loader) -> dict:
    """
    Temporarily quantize a single named layer, run eval, restore, return metrics.
    Context manager approach: quantize → eval → dequantize.
    """
    # TODO: use a context manager that:
    #   1. inserts TensorQuantizer on layer_name only
    #   2. runs mmdet3d evaluation (subset)
    #   3. removes quantizer and restores original weights
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="sensitivity_analysis"):
        # TODO: load model, iterate layers, eval each, build ranked table
        results: dict = {}  # layer_name → {"mAP_drop": float, "NDS_drop": float}
        raise NotImplementedError

        out.write_text(json.dumps(results, indent=2))
        mlflow.log_artifact(str(out))
        print(f"[sensitivity] Ranked table saved to {out}")


if __name__ == "__main__":
    main()

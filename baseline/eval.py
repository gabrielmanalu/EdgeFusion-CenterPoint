"""
FP32 baseline evaluation on nuScenes val.

Wraps mmdetection3d tools/test.py and logs mAP / NDS to MLflow.
Results are written to baseline/results/fp32_baseline.json.

Usage:
    python baseline/eval.py \
        --config configs/centerpoint_pillar02_circlenms_nus.py \
        --checkpoint /workspace/data/centerpoint/centerpoint_nuscenes.pth
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow

MMDET3D_ROOT = Path("/workspace/mmdetection3d")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FP32 CenterPoint baseline")
    p.add_argument("--config", required=True, help="mmdet3d config path")
    p.add_argument("--checkpoint", required=True, help=".pth checkpoint path")
    p.add_argument("--out", default=str(RESULTS_DIR / "fp32_baseline.json"))
    p.add_argument("--gpu-id", default="0")
    return p.parse_args()


def run_mmdet3d_test(config: str, checkpoint: str, out: str, gpu_id: str) -> dict:
    """Delegate to mmdet3d tools/test.py and capture JSON results."""
    cmd = [
        sys.executable,
        str(MMDET3D_ROOT / "tools" / "test.py"),
        config,
        checkpoint,
        "--task",
        "lidar_det",
        "--out",
        out,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    subprocess.run(cmd, env=env, check=True)

    with open(out) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    with mlflow.start_run(run_name="fp32_baseline"):
        mlflow.log_params(
            {
                "config": args.config,
                "checkpoint": args.checkpoint,
                "precision": "fp32",
            }
        )

        metrics = run_mmdet3d_test(args.config, args.checkpoint, args.out, args.gpu_id)

        # TODO: parse mAP / NDS from metrics dict and log as MLflow metrics
        mlflow.log_dict(metrics, "fp32_metrics.json")
        print(f"[eval] Results saved to {args.out}")


if __name__ == "__main__":
    main()

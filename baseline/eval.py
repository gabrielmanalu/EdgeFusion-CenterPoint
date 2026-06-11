"""
FP32 baseline evaluation on nuScenes val.

Wraps mmdetection3d tools/test.py and logs mAP / NDS to MLflow.
Results are written to baseline/results/fp32_baseline.json.

Usage:
    # set these shell vars for brevity:
    #   CFG=/workspace/mmdetection3d/configs/centerpoint
    #   CKPT=/workspace/data/centerpoint
    python baseline/eval.py \
        --config  $CFG/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
        --checkpoint $CKPT/<2022_checkpoint>.pth
    # FP32 baseline: mAP 48.15  NDS 59.22
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
    """Delegate to mmdet3d tools/test.py and capture results."""
    cmd = [
        sys.executable,
        str(MMDET3D_ROOT / "tools" / "test.py"),
        config,
        checkpoint,
        "--task", "lidar_det",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)

    # Parse mAP and NDS from stdout
    metrics = {}
    for line in result.stdout.splitlines():
        if line.strip().startswith("mAP:"):
            metrics["mAP"] = float(line.split(":")[1].strip())
        elif line.strip().startswith("NDS:"):
            metrics["NDS"] = float(line.split(":")[1].strip())

    # Save parsed metrics
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    args = parse_args()

    with mlflow.start_run(run_name="fp32_baseline"):
        mlflow.log_params({
            "config": args.config,
            "checkpoint": args.checkpoint,
            "precision": "fp32",
        })

        metrics = run_mmdet3d_test(
            args.config, args.checkpoint, args.out, args.gpu_id
        )

        # TODO: parse mAP / NDS from metrics dict and log as MLflow metrics
        mlflow.log_dict(metrics, "fp32_metrics.json")
        print(f"[eval] Results saved to {args.out}")


if __name__ == "__main__":
    main()

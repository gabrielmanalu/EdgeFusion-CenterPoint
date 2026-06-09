"""
Export a trained CenterPoint .pth to Autoware's two-ONNX format.

Produces:
    pts_voxel_encoder_centerpoint.onnx
    pts_backbone_neck_head_centerpoint.onnx

Uses centerpoint_onnx_converter.py from the autowarefoundation/mmdetection3d fork.

Usage:
    python baseline/export_onnx.py \
        --cfg   configs/centerpoint_pillar02_circlenms_nus.py \
        --ckpt  /workspace/data/centerpoint/centerpoint_nuscenes.pth \
        --out   baseline/results/onnx_export/
"""

import argparse
import subprocess
import sys
from pathlib import Path

CONVERTER = Path(
    "/workspace/mmdetection3d/projects/AutowareCenterPoint/"
    "centerpoint_onnx_converter.py"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export CenterPoint to Autoware ONNX")
    p.add_argument("--cfg", required=True, help="mmdet3d config")
    p.add_argument("--ckpt", required=True, help=".pth checkpoint")
    p.add_argument("--out", default="baseline/results/onnx_export/")
    return p.parse_args()


def validate_onnx(onnx_dir: Path) -> None:
    """
    Run a forward pass through both exported ONNX models and compare outputs
    against the PyTorch model to confirm numerical parity.

    TODO: implement once export is confirmed working.
    """
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--cfg",
            args.cfg,
            "--ckpt",
            args.ckpt,
            "--work-dir",
            str(out),
        ],
        check=True,
    )
    print(f"[export_onnx] ONNX models written to {out}")
    # validate_onnx(out)   # uncomment once validate_onnx is implemented


if __name__ == "__main__":
    main()

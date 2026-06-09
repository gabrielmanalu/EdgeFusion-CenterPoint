"""
QAT (Quantization-Aware Training) fine-tuning.

Starts from the FP32 checkpoint, inserts fake-quant nodes guided by the
sensitivity analysis, and fine-tunes for a small number of epochs to
recover the accuracy gap left by PTQ.

Design notes (see docs/design_decisions.md):
    - Layers ranked as sensitive in sensitivity.py keep FP16 precision.
    - Learning rate is linearly scaled with batch size.
    - Uses pytorch-quantization (NVIDIA) fake-quant nodes.

Usage:
    python compression/qat.py \
        --config      configs/centerpoint_pillar02_circlenms_nus.py \
        --checkpoint  /workspace/data/centerpoint/centerpoint_nuscenes.pth \
        --sensitivity compression/results/sensitivity.json \
        --epochs 5 --batch-size 8
"""

import argparse

import mlflow

BASE_LR = 1e-4  # reference LR at batch_size = 4
BASE_BATCH = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--sensitivity", required=True, help="JSON output of compression/sensitivity.py"
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default="compression/results/qat/")
    return p.parse_args()


def build_qat_model(checkpoint: str, sensitivity_path: str):
    """
    Load checkpoint, insert fake-quant on non-sensitive layers,
    disable quant on the top-K layers flagged in sensitivity_path.
    """
    # TODO:
    #   1. from pytorch_quantization import quant_modules; quant_modules.initialize()
    #   2. load model via mmdet3d API
    #   3. read sensitivity_path, disable quant on sensitive layers
    raise NotImplementedError


def train_qat(model, loader_train, loader_val, epochs: int, lr: float) -> dict:
    """
    Fine-tune the QAT model.
    Logs per-epoch mAP, NDS, and loss to the active MLflow run.
    Returns best val metrics dict.
    """
    # TODO: training loop — can use mmdet3d Runner or lightweight custom loop
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    lr = BASE_LR * (args.batch_size / BASE_BATCH)

    with mlflow.start_run(run_name=f"qat_bs{args.batch_size}_ep{args.epochs}"):
        mlflow.log_params(
            {
                "method": "qat",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": lr,
            }
        )
        # TODO: build_qat_model → build loaders → train_qat → save best ckpt
        raise NotImplementedError


if __name__ == "__main__":
    main()

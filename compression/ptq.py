"""
PTQ (Post-Training Quantization) INT8 calibration.

Establishes the naive INT8 accuracy point before QAT fine-tuning.

Steps:
    1. Load FP32 model and insert quantization observers.
    2. Run calibration forward passes (~500 samples from nuScenes train set).
    3. Evaluate on val set and record accuracy drop vs FP32 baseline.
    4. Log all results to MLflow.

Usage:
    python compression/ptq.py \
        --config     configs/centerpoint_pillar02_circlenms_nus.py \
        --checkpoint /workspace/data/centerpoint/centerpoint_nuscenes.pth
"""

import argparse

import mlflow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--calib-size", type=int, default=512, help="Number of calibration samples"
    )
    p.add_argument("--out", default="compression/results/ptq/")
    return p.parse_args()


def build_calibration_loader(data_root: str, n_samples: int):
    """Return a DataLoader over the first n_samples of the nuScenes train set."""
    # TODO: build mmdet3d NuScenesDataset restricted to n_samples
    raise NotImplementedError


def insert_quantization_observers(model):
    """
    Insert pytorch-quantization TensorQuantizer observers into the model.
    Equivalent to torch.quantization.prepare for the NVIDIA toolkit.
    """
    # TODO:
    #   from pytorch_quantization import quant_modules
    #   quant_modules.initialize()
    #   from pytorch_quantization.tensor_quant import QuantDescriptor
    #   ...
    raise NotImplementedError


def calibrate(model, loader) -> None:
    """Run calibration forward passes (no backward pass)."""
    # TODO: disable gradients, iterate over loader, collect activation histograms
    raise NotImplementedError


def main() -> None:
    args = parse_args()

    with mlflow.start_run(run_name="ptq_int8"):
        mlflow.log_params(
            {
                "method": "ptq",
                "calib_size": args.calib_size,
            }
        )
        # TODO: load model → insert_quantization_observers → calibrate → eval → log
        raise NotImplementedError


if __name__ == "__main__":
    main()

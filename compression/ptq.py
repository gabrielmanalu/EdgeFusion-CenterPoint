"""
PTQ (Post-Training Quantization) INT8 calibration.

Uses NVIDIA pytorch-quantization to insert TensorQuantizer observers,
collect activation statistics over calibration samples from the nuScenes
train set, and evaluate the resulting INT8 model on the val set.

Install (once, if not present):
    pip install pytorch-quantization --extra-index-url \
        https://pypi.ngc.nvidia.com

Note: quant_modules.initialize() must be called before init_model() so
that quantized layer variants (QuantConv2d, QuantLinear, etc.) are
instantiated instead of the standard ones.

Usage (run from /workspace/mmdetection3d):
    CFG=configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py
    CKPT=/workspace/data/centerpoint/<2022_checkpoint>.pth
    python /path/to/compression/ptq.py --config $CFG --checkpoint $CKPT
"""

import argparse
import json
from pathlib import Path

import mlflow
import torch
from mmdet3d.apis import init_model
from mmengine.config import Config
from mmengine.evaluator import Evaluator as MMEval
from mmengine.runner import Runner
from tqdm import tqdm

try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import quant_modules
    HAS_PQ = True
except ImportError:
    HAS_PQ = False

RESULTS_DIR = Path(__file__).parent / 'results' / 'ptq'
MAP_KEY = 'NuScenes metric/pred_instances_3d_NuScenes/mAP'
NDS_KEY = 'NuScenes metric/pred_instances_3d_NuScenes/NDS'
FP32 = {'mAP': 0.4815, 'NDS': 0.5922}


def _enable_calibration(model: torch.nn.Module) -> None:
    """Switch all TensorQuantizers to calibration mode."""
    for _, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()


def _disable_calibration(model: torch.nn.Module) -> None:
    """Load calibration amax values and switch to fake-quant mode."""
    for _, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
                module.load_calib_amax()
            else:
                module.enable()


def build_calib_loader(cfg: Config):
    """DataLoader over nuScenes train set using the test (no-aug) pipeline."""
    calib_cfg = cfg.copy()
    calib_cfg.test_dataloader.dataset.ann_file = 'nuscenes_infos_train.pkl'
    calib_cfg.test_dataloader.dataset.test_mode = True
    calib_cfg.test_dataloader.batch_size = 1
    calib_cfg.test_dataloader.num_workers = 2
    return Runner.build_dataloader(calib_cfg.test_dataloader)


def run_calibration(
    model: torch.nn.Module, loader, n_calib: int
) -> None:
    """Run n_calib forward passes to collect per-layer activation stats."""
    model.eval()
    _enable_calibration(model)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Calibrating', total=n_calib)):
            if i >= n_calib:
                break
            model.test_step(batch)
    _disable_calibration(model)
    print(f'Calibration done over {n_calib} samples.')


def evaluate_ptq(model: torch.nn.Module, cfg: Config) -> dict:
    """Run full nuScenes val evaluation on the INT8 model."""
    test_loader = Runner.build_dataloader(cfg.test_dataloader)
    evaluator = MMEval(cfg.test_evaluator)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating INT8'):
            outputs = model.test_step(batch)
            evaluator.process(data_batch=batch, data_samples=outputs)
    return evaluator.evaluate(len(test_loader.dataset))


def log_results(metrics: dict, calib_size: int, out: Path) -> None:
    map_ptq = metrics.get(MAP_KEY)
    nds_ptq = metrics.get(NDS_KEY)

    if map_ptq is not None and nds_ptq is not None:
        mlflow.log_metrics({
            'ptq_mAP': map_ptq,
            'ptq_NDS': nds_ptq,
            'mAP_drop': round(FP32['mAP'] - map_ptq, 4),
            'NDS_drop': round(FP32['NDS'] - nds_ptq, 4),
        })
        print(f'\nPTQ INT8  mAP {map_ptq:.4f}  NDS {nds_ptq:.4f}')
        print(f'FP32      mAP {FP32["mAP"]:.4f}  NDS {FP32["NDS"]:.4f}')
        print(f'Drop      mAP {FP32["mAP"] - map_ptq:.4f}  '
              f'NDS {FP32["NDS"] - nds_ptq:.4f}')

    results_path = out / 'ptq_metrics.json'
    with open(results_path, 'w') as f:
        json.dump({'metrics': metrics, 'fp32_baseline': FP32}, f, indent=2)
    mlflow.log_artifact(str(results_path))
    print(f'Results saved to {results_path}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='PTQ INT8 calibration for CenterPoint')
    p.add_argument('--config', required=True, help='mmdet3d config path')
    p.add_argument('--checkpoint', required=True, help='FP32 .pth checkpoint')
    p.add_argument('--calib-size', type=int, default=512,
                   help='Number of calibration samples (default: 512)')
    p.add_argument('--no-eval', action='store_true',
                   help='Skip val evaluation — calibrate and save only')
    p.add_argument('--out', default=str(RESULTS_DIR))
    return p.parse_args()


def main() -> None:
    if not HAS_PQ:
        raise ImportError(
            'pytorch-quantization not installed.\n'
            '  pip install pytorch-quantization '
            '--extra-index-url https://pypi.ngc.nvidia.com'
        )

    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Must be called before init_model so quantized layer variants are used
    quant_modules.initialize()

    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device='cuda:0')
    model.eval()

    calib_loader = build_calib_loader(cfg)

    with mlflow.start_run(run_name=f'ptq_int8_calib{args.calib_size}'):
        mlflow.log_params({
            'method': 'ptq',
            'calib_size': args.calib_size,
            'fp32_mAP': FP32['mAP'],
            'fp32_NDS': FP32['NDS'],
        })

        run_calibration(model, calib_loader, args.calib_size)

        ckpt_path = str(out / 'ptq_calibrated.pth')
        torch.save({'state_dict': model.state_dict(), 'ptq': True}, ckpt_path)
        mlflow.log_artifact(ckpt_path)
        print(f'Calibrated checkpoint saved to {ckpt_path}')

        if not args.no_eval:
            metrics = evaluate_ptq(model, cfg)
            log_results(metrics, args.calib_size, out)
        else:
            print('Skipping eval (--no-eval). '
                  'Run with the saved checkpoint to evaluate later.')


if __name__ == '__main__':
    main()

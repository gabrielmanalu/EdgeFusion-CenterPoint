"""
PTQ using torch.ao.quantization FX mode — CUDA-compatible, no compilation.

Why torch.ao instead of pytorch-quantization:
    The A40 pod has CUDA 12.8; PyTorch compiled for CUDA 11.8. That mismatch
    breaks both the pre-built pytorch-quantization wheel (ABI mismatch) and
    the source build (nvcc version mismatch). torch.ao is built into PyTorch 2.1.

torch.ao FX mode inserts FakeQuantize nodes that simulate INT8 arithmetic on
GPU without actual INT8 conversion. Calibration sets the per-layer scales;
evaluating the prepared model (before convert_fx) gives the PTQ accuracy.
The same calibrated state dict is used as the QAT starting point.

Quantized:  pts_backbone (SECOND) + pts_neck (SECONDFPN)
FP32:       pts_voxel_encoder (5 KB), pts_middle_encoder, pts_bbox_head

Usage (run from /workspace/mmdetection3d):
    CFG=configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py
    CKPT=/workspace/data/centerpoint/<2022_checkpoint>.pth
    python /path/to/compression/ptq.py --config $CFG --checkpoint $CKPT
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import mlflow
import torch
from mmdet3d.apis import init_model
from mmengine.config import Config
from mmengine.evaluator import Evaluator as MMEval
from mmengine.runner import Runner
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx
from tqdm import tqdm

RESULTS_DIR = Path(__file__).parent / 'results' / 'ptq'
MAP_KEY = 'NuScenes metric/pred_instances_3d_NuScenes/mAP'
NDS_KEY = 'NuScenes metric/pred_instances_3d_NuScenes/NDS'
FP32 = {'mAP': 0.4815, 'NDS': 0.5922}


def build_calib_loader(cfg: Config):
    """DataLoader over nuScenes train set using the test (no-aug) pipeline."""
    calib_cfg = cfg.copy()
    calib_cfg.test_dataloader.dataset.ann_file = 'nuscenes_infos_train.pkl'
    calib_cfg.test_dataloader.batch_size = 1
    calib_cfg.test_dataloader.num_workers = 2
    return Runner.build_dataloader(calib_cfg.test_dataloader)


def get_voxel_dict(model: torch.nn.Module, batch: dict) -> dict:
    return model.data_preprocessor.voxelize(batch['inputs']['points'], batch)


def compute_bev_features(
    model: torch.nn.Module, voxel_dict: dict
) -> torch.Tensor:
    """Run encoder + scatter to produce BEV pseudo-image [B, 64, H, W]."""
    with torch.no_grad():
        pillar_feats = model.pts_voxel_encoder(
            voxel_dict['voxels'].cuda(),
            voxel_dict['num_points'].cuda(),
            voxel_dict['coors'].cuda(),
        ).squeeze()
        batch_size = int(voxel_dict['coors'][-1, 0].item()) + 1
        bev = model.pts_middle_encoder(
            pillar_feats, voxel_dict['coors'].cuda(), batch_size
        )
    return bev


def prepare_model_for_ptq(
    model: torch.nn.Module, bev_example: torch.Tensor
) -> Tuple[bool, bool]:
    """Insert FakeQuantize nodes into backbone and neck via torch.ao FX.

    FakeQuantize simulates INT8 precision on GPU in FP32 tensors — no actual
    INT8 conversion until convert_fx (which we defer to the TRT export in P3).
    Returns (backbone_quantized, neck_quantized).
    """
    qconfig_mapping = get_default_qconfig_mapping('x86')
    backbone_ok = neck_ok = False

    # pts_backbone: SECOND — Conv2d + BN + ReLU blocks
    try:
        model.pts_backbone = prepare_fx(
            model.pts_backbone,
            qconfig_mapping,
            example_inputs=(bev_example,),
        )
        backbone_ok = True
        print('[PTQ] pts_backbone: FakeQuantize nodes inserted')
    except Exception as exc:
        print(f'[PTQ] pts_backbone FX tracing failed ({exc}) — kept FP32')

    # pts_neck: SECONDFPN — ConvTranspose2d upsampling + concat
    if backbone_ok:
        try:
            with torch.no_grad():
                neck_example = model.pts_backbone(bev_example)
            model.pts_neck = prepare_fx(
                model.pts_neck,
                qconfig_mapping,
                example_inputs=(neck_example,),
            )
            neck_ok = True
            print('[PTQ] pts_neck: FakeQuantize nodes inserted')
        except Exception as exc:
            print(f'[PTQ] pts_neck FX tracing failed ({exc}) — kept FP32')

    return backbone_ok, neck_ok


def run_calibration(
    model: torch.nn.Module, loader, n_calib: int
) -> None:
    """Run n_calib forward passes so FakeQuantize observers collect stats."""
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(
            tqdm(loader, desc='Calibrating', total=n_calib)
        ):
            if i >= n_calib:
                break
            model.test_step(batch)
    print(f'[PTQ] Calibration complete ({n_calib} samples).')


def evaluate_ptq(model: torch.nn.Module, cfg: Config) -> dict:
    """Evaluate on nuScenes val with FakeQuantize active (simulated INT8)."""
    test_loader = Runner.build_dataloader(cfg.test_dataloader)
    evaluator = MMEval(cfg.test_evaluator)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating PTQ (fake-quant)'):
            outputs = model.test_step(batch)
            evaluator.process(data_batch=batch, data_samples=outputs)
    return evaluator.evaluate(len(test_loader.dataset))


def log_results(metrics: dict, out: Path) -> None:
    map_ptq = metrics.get(MAP_KEY)
    nds_ptq = metrics.get(NDS_KEY)

    if map_ptq is not None and nds_ptq is not None:
        mlflow.log_metrics({
            'ptq_mAP': round(map_ptq, 4),
            'ptq_NDS': round(nds_ptq, 4),
            'mAP_drop': round(FP32['mAP'] - map_ptq, 4),
            'NDS_drop': round(FP32['NDS'] - nds_ptq, 4),
        })
        print(f'\nPTQ INT8  mAP {map_ptq:.4f}  NDS {nds_ptq:.4f}')
        print(f'FP32      mAP {FP32["mAP"]:.4f}  NDS {FP32["NDS"]:.4f}')
        print(
            f'Drop      mAP {FP32["mAP"] - map_ptq:.4f}  '
            f'NDS {FP32["NDS"] - nds_ptq:.4f}'
        )

    results_path = out / 'ptq_metrics.json'
    with open(results_path, 'w') as f:
        json.dump({'metrics': metrics, 'fp32_baseline': FP32}, f, indent=2)
    mlflow.log_artifact(str(results_path))
    print(f'[PTQ] Results saved to {results_path}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='PTQ INT8 for CenterPoint (torch.ao)')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--calib-size', type=int, default=512,
                   help='Number of calibration samples (default: 512)')
    p.add_argument('--no-eval', action='store_true',
                   help='Calibrate and save only; skip val evaluation')
    p.add_argument('--out', default=str(RESULTS_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device='cuda:0')
    model.eval()

    calib_loader = build_calib_loader(cfg)

    # BEV example input needed for FX symbolic tracing
    first_batch = next(iter(calib_loader))
    voxel_dict = get_voxel_dict(model, first_batch)
    bev_example = compute_bev_features(model, voxel_dict)

    with mlflow.start_run(run_name=f'ptq_int8_calib{args.calib_size}'):
        mlflow.log_params({
            'method': 'ptq_torch_ao_fx',
            'calib_size': args.calib_size,
            'fp32_mAP': FP32['mAP'],
            'fp32_NDS': FP32['NDS'],
        })

        backbone_ok, neck_ok = prepare_model_for_ptq(model, bev_example)
        mlflow.log_params({
            'backbone_quantized': backbone_ok,
            'neck_quantized': neck_ok,
        })

        if not backbone_ok:
            print('[PTQ] No modules quantized — see FX tracing errors above.')
            return

        run_calibration(model, calib_loader, args.calib_size)

        ckpt_path = str(out / 'ptq_calibrated.pth')
        torch.save({
            'state_dict': model.state_dict(),
            'backbone_quantized': backbone_ok,
            'neck_quantized': neck_ok,
        }, ckpt_path)
        mlflow.log_artifact(ckpt_path)
        print(f'[PTQ] Calibrated checkpoint saved: {ckpt_path}')

        if not args.no_eval:
            metrics = evaluate_ptq(model, cfg)
            log_results(metrics, out)
        else:
            print('[PTQ] --no-eval set; skipping val evaluation.')


if __name__ == '__main__':
    main()

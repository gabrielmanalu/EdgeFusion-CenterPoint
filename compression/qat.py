"""
QAT (Quantization-Aware Training) fine-tuning.

Starts from ptq_calibrated.pth (backbone + neck with calibrated FakeQuantize
scales) and fine-tunes model weights while keeping quantization scales fixed.
PyTorch's FakeQuantize uses Straight-Through Estimator (STE) for gradients —
quantization noise propagates through backward automatically.

Key settings:
    Observer stats:  FROZEN  — scales stay fixed from PTQ calibration
    FakeQuantize:    ENABLED — INT8 noise active during forward/backward
    LR:              1e-5    — 0.1× original training LR (1e-4)

Estimate: 5 epochs, batch=4, no CBGS ≈ 5-6 hrs on A40.

Usage (run from /workspace/mmdetection3d):
    CFG=configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py
    python .../compression/qat.py \
        --config   $CFG \
        --ptq-ckpt .../compression/results/ptq/ptq_calibrated.pth
"""

import argparse
import copy
import json
import warnings
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from mmdet3d.apis import init_model
from mmengine.config import Config
from mmengine.evaluator import Evaluator as MMEval
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from torch.ao.quantization import (
    FakeQuantize,
    QConfig,
    HistogramObserver,
    PerChannelMinMaxObserver,
    QConfigMapping,
)
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from tqdm import tqdm

warnings.filterwarnings('ignore', message='Please use quant_min and quant_max')

RESULTS_DIR = Path(__file__).parent / 'results' / 'qat'
MAP_KEY = 'NuScenes metric/pred_instances_3d_NuScenes/mAP'
NDS_KEY = 'NuScenes metric/pred_instances_3d_NuScenes/NDS'
FP32 = {'mAP': 0.4815, 'NDS': 0.5922}

BASE_LR = 1e-5      # 0.1x of original training LR (1e-4)
BASE_BATCH = 4      # reference batch for LR scaling


# ── Utilities (mirrored from ptq.py) ─────────────────────────────────────────

def _build_qconfig_mapping() -> QConfigMapping:
    """TRT-compatible INT8 qconfig using FakeQuantize for GPU-side simulation.

    FakeQuantize.with_args(observer=...) is required — prepare_qat_fx only
    inserts FakeQuantize nodes when the QConfig explicitly uses FakeQuantize,
    not raw observer classes.
    """
    qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=HistogramObserver,
            quant_min=-128, quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        ),
        weight=FakeQuantize.with_args(
            observer=PerChannelMinMaxObserver,
            quant_min=-128, quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        ),
    )
    mapping = QConfigMapping()
    mapping.set_global(qconfig)
    return mapping


def get_voxel_dict(model: nn.Module, batch: dict) -> dict:
    return model.data_preprocessor.voxelize(batch['inputs']['points'], batch)


def compute_bev_features(model: nn.Module, voxel_dict: dict) -> torch.Tensor:
    """Run encoder + scatter → BEV pseudo-image [B, 64, H, W]."""
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


def _prepare_fx_on_model(model: nn.Module, bev_example: torch.Tensor) -> None:
    """Insert FakeQuantize nodes into backbone + neck (same as ptq.py)."""
    qconfig_mapping = _build_qconfig_mapping()

    model.pts_backbone = prepare_qat_fx(
        model.pts_backbone,
        qconfig_mapping,
        example_inputs=(bev_example,),
    )

    from mmdet3d.models.necks.second_fpn import SECONDFPN as _SFPN
    _orig = _SFPN.forward

    def _traceable(self, x):
        outs = [self.deblocks[i](x[i]) for i in range(len(self.deblocks))]
        out = torch.cat(outs, dim=1) if len(self.deblocks) > 1 else outs[0]
        return [out]

    _SFPN.forward = _traceable
    try:
        with torch.no_grad():
            neck_ex = model.pts_backbone(bev_example)
        model.pts_neck = prepare_qat_fx(
            model.pts_neck, qconfig_mapping, example_inputs=(neck_ex,)
        )
    finally:
        _SFPN.forward = _orig


# ── QAT-specific ──────────────────────────────────────────────────────────────

def freeze_observer_stats(model: nn.Module) -> None:
    """Freeze FakeQuantize observer stats — only weights update during QAT."""
    for module in model.modules():
        if isinstance(module, FakeQuantizeBase):
            module.disable_observer()
            module.enable_fake_quant()


def prepare_and_load_ptq_checkpoint(
    cfg: Config, fp32_ckpt: str, ptq_ckpt: str, calib_batch: dict
) -> nn.Module:
    """Init from FP32 checkpoint (sets train_cfg), apply FX, load PTQ scales."""
    model = init_model(cfg, checkpoint=fp32_ckpt, device='cuda:0')
    model.eval()
    voxel_dict = get_voxel_dict(model, calib_batch)
    bev_example = compute_bev_features(model, voxel_dict)
    _prepare_fx_on_model(model, bev_example)
    ckpt = torch.load(ptq_ckpt, map_location='cuda:0')
    state_dict = ckpt.get('state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'[QAT] PTQ checkpoint loaded '
          f'({len(missing)} missing, {len(unexpected)} unexpected keys)')
    return model


def build_train_loader(cfg: Config, batch_size: int):
    """Train loader without CBGSDataset for faster QAT iteration."""
    train_cfg = copy.deepcopy(cfg.train_dataloader)
    if train_cfg.dataset.get('type') == 'CBGSDataset':
        train_cfg.dataset = train_cfg.dataset.dataset
    train_cfg.batch_size = batch_size
    train_cfg.num_workers = 4
    return Runner.build_dataloader(train_cfg)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> float:
    """One QAT training epoch. Returns mean loss."""
    model.train()
    freeze_observer_stats(model)   # re-freeze each epoch (train() re-enables them)

    total_loss = 0.0
    for batch in tqdm(loader, desc=f'QAT epoch {epoch}'):
        data = model.data_preprocessor(batch, True)
        losses = model(**data, mode='loss')
        loss = sum(v for k, v in losses.items() if 'loss' in k.lower())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 35.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_qat(model: nn.Module, cfg: Config) -> dict:
    """Evaluate on nuScenes val with FakeQuantize active (INT8 simulation)."""
    test_loader = Runner.build_dataloader(cfg.test_dataloader)
    evaluator = MMEval(cfg.test_evaluator)
    evaluator.dataset_meta = test_loader.dataset.metainfo
    model.eval()
    freeze_observer_stats(model)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating QAT'):
            outputs = model.test_step(batch)
            evaluator.process(data_batch=batch, data_samples=outputs)
    return evaluator.evaluate(len(test_loader.dataset))


def log_results(metrics: dict, ptq_map: float, out: Path) -> None:
    map_qat = metrics.get(MAP_KEY)
    nds_qat = metrics.get(NDS_KEY)

    if map_qat is not None and nds_qat is not None:
        mlflow.log_metrics({
            'qat_mAP': round(map_qat, 4),
            'qat_NDS': round(nds_qat, 4),
            'recovery_vs_ptq': round(map_qat - ptq_map, 4),
            'drop_vs_fp32': round(FP32['mAP'] - map_qat, 4),
        })
        print(f'\nQAT INT8  mAP {map_qat:.4f}  NDS {nds_qat:.4f}')
        print(f'PTQ INT8  mAP {ptq_map:.4f}')
        print(f'FP32      mAP {FP32["mAP"]:.4f}  NDS {FP32["NDS"]:.4f}')
        print(f'Recovery vs PTQ: +{map_qat - ptq_map:.4f}')

    out_path = out / 'qat_metrics.json'
    with open(out_path, 'w') as f:
        json.dump({'metrics': metrics, 'fp32_baseline': FP32}, f, indent=2)
    mlflow.log_artifact(str(out_path))
    print(f'[QAT] Results saved to {out_path}')


def apply_sensitivity_mask(model: nn.Module, sensitivity_path: str) -> int:
    """Keep sensitive nodes in FP16 by disabling their FakeQuantize.

    Reads sensitivity.json produced by sensitivity.py and disables fake-quant
    on the top-K most sensitive nodes so they remain in FP32/FP16 during QAT.
    Returns the number of nodes kept in FP16.
    """
    with open(sensitivity_path) as f:
        data = json.load(f)
    sensitive = set(data.get('sensitive_nodes', []))
    kept_fp16 = 0
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantizeBase) and name in sensitive:
            module.disable_fake_quant()
            kept_fp16 += 1
    print(f'[QAT] Mixed precision: {kept_fp16} nodes kept in FP16 '
          f'({len(sensitive)} sensitive nodes in sensitivity.json)')
    return kept_fp16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='QAT fine-tuning for CenterPoint')
    p.add_argument('--config', required=True)
    p.add_argument('--fp32-ckpt', required=True,
                   help='Original FP32 .pth checkpoint (for model init + train_cfg)')
    p.add_argument('--ptq-ckpt', required=True,
                   help='ptq_calibrated.pth from compression/ptq.py')
    p.add_argument('--sensitivity', default=None,
                   help='sensitivity.json from compression/sensitivity.py — '
                        'sensitive nodes are kept in FP16 (mixed precision)')
    p.add_argument('--ptq-map', type=float, default=0.0,
                   help='PTQ mAP for logging (from ptq_metrics.json)')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--eval-interval', type=int, default=5,
                   help='Evaluate every N epochs (default: last epoch only)')
    p.add_argument('--out', default=str(RESULTS_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lr = BASE_LR * (args.batch_size / BASE_BATCH)

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    # Need one batch for BEV shape inference during FX tracing
    calib_loader = Runner.build_dataloader(
        copy.deepcopy(cfg.test_dataloader)
    )
    calib_batch = next(iter(calib_loader))

    model = prepare_and_load_ptq_checkpoint(
        cfg, args.fp32_ckpt, args.ptq_ckpt, calib_batch
    )

    if args.sensitivity:
        apply_sensitivity_mask(model, args.sensitivity)

    train_loader = build_train_loader(cfg, args.batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=lr * 0.01
    )

    run_name = f'qat_ep{args.epochs}_bs{args.batch_size}'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'method': 'qat_torch_ao_fx',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': lr,
            'fp32_mAP': FP32['mAP'],
            'ptq_mAP': args.ptq_map,
        })

        best_map = 0.0
        for epoch in range(1, args.epochs + 1):
            avg_loss = train_one_epoch(model, train_loader, optimizer, epoch)
            scheduler.step()
            mlflow.log_metrics({
                'loss': round(avg_loss, 4),
                'lr': scheduler.get_last_lr()[0],
            }, step=epoch)
            print(f'[QAT] epoch {epoch}/{args.epochs}  '
                  f'loss {avg_loss:.4f}  lr {scheduler.get_last_lr()[0]:.2e}')

            ckpt_path = str(out / f'qat_epoch{epoch}.pth')
            torch.save({'state_dict': model.state_dict()}, ckpt_path)

            if epoch % args.eval_interval == 0 or epoch == args.epochs:
                metrics = evaluate_qat(model, cfg)
                log_results(metrics, args.ptq_map, out)
                map_val = metrics.get(MAP_KEY, 0.0)
                if map_val > best_map:
                    best_map = map_val
                    best_path = str(out / 'qat_best.pth')
                    torch.save({'state_dict': model.state_dict()}, best_path)
                    mlflow.log_artifact(best_path)
                    print(f'[QAT] New best: mAP {map_val:.4f} → {best_path}')


if __name__ == '__main__':
    main()

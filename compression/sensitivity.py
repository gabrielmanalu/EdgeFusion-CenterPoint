"""
Per-layer quantization sensitivity analysis.

For each FakeQuantize node in pts_backbone and pts_neck:
    1. Disable ALL fake-quant (model runs in FP32).
    2. Enable ONLY this node (single quantizer active).
    3. Compute training loss on n_samples (default 500, ~10s per node).
    4. Record loss increase vs the FP32 reference on the same samples.

Training loss is used instead of mAP because NuScenesEval requires predictions
for the complete val split — partial subsets fail its token-set assertion.
Loss is a valid sensitivity proxy: it directly measures how much quantization
noise hurts the model's ability to produce correct heatmap + regression outputs.

Outputs sensitivity.json with per-node rankings and a sensitive_nodes list.
QAT reads this file to keep sensitive nodes in FP16 (mixed precision).

Estimate: ~40 nodes x 10s = ~7 min on A40.

Usage (run from /workspace/mmdetection3d):
    CFG=configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py
    python .../compression/sensitivity.py \
        --config  $CFG \
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
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from torch.ao.quantization import (
    QConfig,
    HistogramObserver,
    PerChannelMinMaxObserver,
    QConfigMapping,
)
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.ao.quantization.quantize_fx import prepare_fx
from tqdm import tqdm

warnings.filterwarnings('ignore', message='Please use quant_min and quant_max')

RESULTS_DIR = Path(__file__).parent / 'results' / 'sensitivity'

# Loss-based threshold: flag node as sensitive if loss increases by more
# than this fraction of the FP32 reference loss.
SENSITIVITY_THRESHOLD = 0.02   # 2% relative loss increase


# ── Utilities (shared with ptq.py / qat.py) ───────────────────────────────────

def _build_qconfig_mapping() -> QConfigMapping:
    qconfig = QConfig(
        activation=HistogramObserver.with_args(
            quant_min=-128, quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        ),
        weight=PerChannelMinMaxObserver.with_args(
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
    with torch.no_grad():
        pillar_feats = model.pts_voxel_encoder(
            voxel_dict['voxels'].cuda(),
            voxel_dict['num_points'].cuda(),
            voxel_dict['coors'].cuda(),
        ).squeeze()
        batch_size = int(voxel_dict['coors'][-1, 0].item()) + 1
        return model.pts_middle_encoder(
            pillar_feats, voxel_dict['coors'].cuda(), batch_size
        )


def _apply_fx_preparation(model: nn.Module, bev_example: torch.Tensor) -> None:
    """Insert FakeQuantize into backbone + neck (same as ptq.py / qat.py)."""
    qconfig_mapping = _build_qconfig_mapping()

    model.pts_backbone = prepare_fx(
        model.pts_backbone, qconfig_mapping, example_inputs=(bev_example,)
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
        model.pts_neck = prepare_fx(
            model.pts_neck, qconfig_mapping, example_inputs=(neck_ex,)
        )
    finally:
        _SFPN.forward = _orig


def prepare_and_load_ptq_checkpoint(
    cfg: Config, ptq_ckpt: str, sample_batch: dict
) -> nn.Module:
    model = init_model(cfg, checkpoint=None, device='cuda:0')
    model.eval()

    voxel_dict = get_voxel_dict(model, sample_batch)
    bev_example = compute_bev_features(model, voxel_dict)
    _apply_fx_preparation(model, bev_example)

    ckpt = torch.load(ptq_ckpt, map_location='cuda:0')
    state_dict = ckpt.get('state_dict', ckpt)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        print(f'[Sensitivity] strict=True failed ({exc}); using strict=False')
        model.load_state_dict(state_dict, strict=False)

    return model


# ── Sensitivity helpers ───────────────────────────────────────────────────────

def get_fake_quant_nodes(model: nn.Module) -> dict:
    """Return {name: module} for every FakeQuantize node in the model."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, FakeQuantizeBase)
    }


def set_all_fake_quant(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, FakeQuantizeBase):
            if enabled:
                module.enable_fake_quant()
            else:
                module.disable_fake_quant()


def build_train_loader(cfg: Config, batch_size: int = 1):
    """Train loader without CBGSDataset — provides GT for loss computation."""
    train_cfg = copy.deepcopy(cfg.train_dataloader)
    if train_cfg.dataset.get('type') == 'CBGSDataset':
        train_cfg.dataset = train_cfg.dataset.dataset
    train_cfg.batch_size = batch_size
    train_cfg.num_workers = 2
    return Runner.build_dataloader(train_cfg)


def compute_loss_proxy(
    model: nn.Module, loader, n_samples: int
) -> float:
    """Mean training loss over n_samples as a sensitivity proxy.

    Using loss instead of mAP avoids NuScenesEval's requirement for
    predictions on the complete val split. Loss is computed in eval mode
    (BN uses running stats) with no_grad for speed.
    """
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_samples:
                break
            data = model.data_preprocessor(batch, True)
            losses = model(**data, mode='loss')
            total += sum(
                v.item() for k, v in losses.items() if 'loss' in k.lower()
            )
            count += 1
    return total / max(count, 1)


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_sensitivity_analysis(
    model: nn.Module,
    train_loader,
    n_samples: int,
    threshold: float,
    out: Path,
) -> dict:
    fq_nodes = get_fake_quant_nodes(model)
    print(f'[Sensitivity] {len(fq_nodes)} FakeQuantize nodes found.')

    print('[Sensitivity] Computing FP32 reference loss...')
    set_all_fake_quant(model, False)
    ref_loss = compute_loss_proxy(model, train_loader, n_samples)
    print(f'[Sensitivity] FP32 reference loss: {ref_loss:.4f}')

    results = {}
    for node_name, fq_module in tqdm(fq_nodes.items(), desc='Sensitivity'):
        set_all_fake_quant(model, False)
        fq_module.enable_fake_quant()
        node_loss = compute_loss_proxy(model, train_loader, n_samples)
        abs_increase = node_loss - ref_loss
        rel_increase = abs_increase / max(ref_loss, 1e-8)
        results[node_name] = {
            'loss': round(node_loss, 4),
            'loss_increase': round(abs_increase, 4),
            'rel_increase': round(rel_increase, 4),
            'sensitive': rel_increase > threshold,
        }

    set_all_fake_quant(model, True)

    ranked = dict(sorted(
        results.items(), key=lambda x: x[1]['loss_increase'], reverse=True
    ))

    sensitive_nodes = [k for k, v in ranked.items() if v['sensitive']]
    print(f'\n[Sensitivity] Sensitive nodes '
          f'(rel increase > {threshold:.2f}): {len(sensitive_nodes)}')
    for name in sensitive_nodes[:10]:
        v = ranked[name]
        print(f'  {name:70s}  +{v["rel_increase"]:.3f}')

    output = {
        'metric': 'training_loss_proxy',
        'fp32_ref_loss': round(ref_loss, 4),
        'n_samples': n_samples,
        'sensitivity_threshold': threshold,
        'sensitive_nodes': sensitive_nodes,
        'layers': ranked,
    }

    out_path = out / 'sensitivity.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'[Sensitivity] Results saved to {out_path}')
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Per-layer sensitivity analysis for CenterPoint PTQ'
    )
    p.add_argument('--config', required=True)
    p.add_argument('--ptq-ckpt', required=True)
    p.add_argument('--fast-samples', type=int, default=500)
    p.add_argument('--threshold', type=float, default=SENSITIVITY_THRESHOLD,
                   help='Relative loss increase threshold (default: 0.02)')
    p.add_argument('--out', default=str(RESULTS_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    # One val batch for BEV shape inference during FX tracing
    val_loader = Runner.build_dataloader(copy.deepcopy(cfg.test_dataloader))
    sample_batch = next(iter(val_loader))

    model = prepare_and_load_ptq_checkpoint(cfg, args.ptq_ckpt, sample_batch)
    print('[Sensitivity] Model prepared and PTQ checkpoint loaded.')

    train_loader = build_train_loader(cfg, batch_size=1)

    with mlflow.start_run(run_name=f'sensitivity_{args.fast_samples}samp'):
        mlflow.log_params({
            'method': 'loss_proxy_sensitivity',
            'fast_samples': args.fast_samples,
            'threshold': args.threshold,
        })

        results = run_sensitivity_analysis(
            model, train_loader, args.fast_samples, args.threshold, out
        )

        mlflow.log_metrics({
            'fp32_ref_loss': results['fp32_ref_loss'],
            'n_sensitive_nodes': len(results['sensitive_nodes']),
        })
        mlflow.log_artifact(str(out / 'sensitivity.json'))


if __name__ == '__main__':
    main()

"""
Per-layer quantization sensitivity analysis.

For each FakeQuantize node in pts_backbone and pts_neck:
    1. Disable ALL fake-quant (model runs in FP32).
    2. Enable ONLY this node (single quantizer active).
    3. Evaluate on a fast val subset (default 500 samples, ~70s per node).
    4. Record mAP drop vs the FP32 reference on the same subset.

Outputs sensitivity.json with per-node rankings.
QAT reads this file to keep sensitive nodes in FP16 (mixed precision).

Estimate: ~40 nodes x 70s = ~50 min on A40.

Usage (run from /workspace/mmdetection3d):
    CFG=configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py
    python .../compression/sensitivity.py \
        --config  $CFG \
        --ptq-ckpt .../compression/results/ptq/ptq_calibrated.pth \
        --fast-samples 500
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
MAP_KEY = 'NuScenes metric/pred_instances_3d_NuScenes/mAP'
NDS_KEY = 'NuScenes metric/pred_instances_3d_NuScenes/NDS'
FP32 = {'mAP': 0.4815, 'NDS': 0.5922}

# Layers with drop above this threshold are flagged as sensitive
SENSITIVITY_THRESHOLD = 0.005   # 0.5% mAP drop


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


def fast_eval(
    model: nn.Module, loader, cfg: Config, n_samples: int
) -> dict:
    """Evaluate on the first n_samples of the val set.

    mAP is lower than full eval but relative rankings between nodes are valid
    since the same samples are used for every node.
    """
    evaluator = MMEval(cfg.test_evaluator)
    evaluator.dataset_meta = loader.dataset.metainfo
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_samples:
                break
            outputs = model.test_step(batch)
            evaluator.process(data_batch=batch, data_samples=outputs)
    return evaluator.evaluate(min(n_samples, len(loader.dataset)))


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_sensitivity_analysis(
    model: nn.Module,
    val_loader,
    cfg: Config,
    n_samples: int,
    out: Path,
) -> dict:
    """Per-node sensitivity sweep.

    For each FakeQuantize node: quantize only that node, measure mAP drop.
    """
    fq_nodes = get_fake_quant_nodes(model)
    print(f'[Sensitivity] Found {len(fq_nodes)} FakeQuantize nodes.')

    # FP32 reference on the same n_samples subset
    print('[Sensitivity] Computing FP32 reference...')
    set_all_fake_quant(model, False)
    ref_metrics = fast_eval(model, val_loader, cfg, n_samples)
    ref_map = ref_metrics.get(MAP_KEY, 0.0)
    ref_nds = ref_metrics.get(NDS_KEY, 0.0)
    print(f'[Sensitivity] FP32 reference ({n_samples} samples): '
          f'mAP {ref_map:.4f}  NDS {ref_nds:.4f}')

    results = {}
    for node_name, fq_module in tqdm(fq_nodes.items(), desc='Sensitivity'):
        # Quantize only this node
        set_all_fake_quant(model, False)
        fq_module.enable_fake_quant()

        metrics = fast_eval(model, val_loader, cfg, n_samples)
        node_map = metrics.get(MAP_KEY, 0.0)
        drop = ref_map - node_map

        results[node_name] = {
            'mAP': round(node_map, 4),
            'mAP_drop': round(drop, 4),
            'sensitive': drop > SENSITIVITY_THRESHOLD,
        }

    # Re-enable all fake quant
    set_all_fake_quant(model, True)

    # Sort by drop descending
    ranked = dict(sorted(
        results.items(), key=lambda x: x[1]['mAP_drop'], reverse=True
    ))

    sensitive_nodes = [k for k, v in ranked.items() if v['sensitive']]
    print(f'\n[Sensitivity] Sensitive nodes '
          f'(drop > {SENSITIVITY_THRESHOLD:.3f}): {len(sensitive_nodes)}')
    for name in sensitive_nodes[:10]:
        print(f'  {name:60s}  drop {ranked[name]["mAP_drop"]:.4f}')

    output = {
        'fp32_map_fast': round(ref_map, 4),
        'fp32_nds_fast': round(ref_nds, 4),
        'fp32_map_full': FP32['mAP'],
        'n_samples': n_samples,
        'sensitivity_threshold': SENSITIVITY_THRESHOLD,
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
    p.add_argument('--ptq-ckpt', required=True,
                   help='ptq_calibrated.pth from compression/ptq.py')
    p.add_argument('--fast-samples', type=int, default=500,
                   help='Val samples per node (default 500, ~70s/node)')
    p.add_argument('--threshold', type=float, default=SENSITIVITY_THRESHOLD,
                   help='mAP drop threshold to flag a node as sensitive')
    p.add_argument('--out', default=str(RESULTS_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    # One sample for BEV shape inference during FX tracing
    sample_loader = Runner.build_dataloader(copy.deepcopy(cfg.test_dataloader))
    sample_batch = next(iter(sample_loader))

    model = prepare_and_load_ptq_checkpoint(cfg, args.ptq_ckpt, sample_batch)
    print('[Sensitivity] Model prepared and PTQ checkpoint loaded.')

    # Rebuild val loader (separate from sample_loader to reset iteration)
    val_loader = Runner.build_dataloader(copy.deepcopy(cfg.test_dataloader))

    with mlflow.start_run(run_name=f'sensitivity_{args.fast_samples}samp'):
        mlflow.log_params({
            'method': 'per_node_sensitivity',
            'fast_samples': args.fast_samples,
            'threshold': args.threshold,
        })

        results = run_sensitivity_analysis(
            model, val_loader, cfg, args.fast_samples, out
        )

        mlflow.log_metrics({
            'fp32_map_fast': results['fp32_map_fast'],
            'n_sensitive_nodes': len(results['sensitive_nodes']),
        })
        mlflow.log_artifact(str(out / 'sensitivity.json'))


if __name__ == '__main__':
    main()
    
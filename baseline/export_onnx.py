"""
Multi-task ONNX export for the open-mmlab 10-class CenterPoint model.

Autoware's centerpoint_onnx_converter.py cannot be used here because:
  1. It overrides the encoder type to PillarFeatureNetAutoware, incompatible
     with our standard PillarFeatureNet checkpoint.
  2. CenterHeadONNX.forward() only processes task_heads[0], exporting only
     the first task head (car, 1 class).

This script produces the same two Autoware-named ONNX files by:
  - Capturing pre-PFN features via a forward hook (encoder export).
  - Concatenating all 6 task head outputs channel-wise (backbone export).

Output schemas:
  encoder:  input_features [N, max_pts, 11] -> pillar_features [N, 1, 64]
  backbone: spatial_features [B, 64, H, W]
            heatmap [B, 10, H, W]  reg    [B, 12, H, W]
            height  [B,  6, H, W]  dim    [B, 18, H, W]
            rot     [B, 12, H, W]  vel    [B, 12, H, W]

Heatmap channel layout (for CUDA postprocessing):
  ch 0      car
  ch 1-2    truck, construction_vehicle
  ch 3-4    bus, trailer
  ch 5      barrier
  ch 6-7    motorcycle, bicycle
  ch 8-9    pedestrian, traffic_cone

These output schemas are IDENTICAL across all compression variants — pruning/
distillation only change internal backbone/neck/shared_conv channel counts
(invisible at the ONNX I/O boundary); task_heads (and therefore output shapes)
are untouched. The encoder is also unchanged by pruning (pts_voxel_encoder is
outside the pruned backbone+neck+shared_conv boundary) — its ONNX is identical
across variants, but exported per-variant anyway for a complete, self-contained
pair per checkpoint.

Two loading modes:

  --checkpoint  FP32 .pth checkpoint. Architecture matches --config exactly;
                loaded via init_model(cfg, checkpoint). Use for the FP32
                baseline.

  --model-path  Full model object (.pt, saved via torch.save(model, path)).
                Architecture does NOT match --config (pruned/distilled channel
                counts differ) — init_model would raise a shape-mismatch error.
                Loaded directly via torch.load(). Use for pruned_model_*.pt /
                distilled_model_*.pt. --config is still required (provides the
                dataset/dataloader for the calibration batch; the MODEL fields
                in cfg are ignored in this mode).

Usage:
    # FP32 baseline
    # CFG=/workspace/mmdetection3d/configs/centerpoint
    # CKPT=/workspace/data/centerpoint
    python baseline/export_onnx.py \
        --config  $CFG/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
        --checkpoint $CKPT/<2022_checkpoint>.pth \
        --out     baseline/results/onnx_export/

    # Pruned 25% (or any pruned_model_*.pt / distilled_model_*.pt)
    python baseline/export_onnx.py \
        --config $CFG/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
        --model-path compression/results/pruning/ratio_25/pruned_model_25_recalib.pt \
        --variant pruned25 \
        --out compression/results/onnx_export/pruned25/

    # --synthetic-input: skip the real dataloader entirely (no /data/nuscenes/
    # needed). ONNX export only needs correctly-shaped tensors for tracing,
    # not real point clouds. Use this on a fresh pod where only the conda env
    # + checkpoint/.pt files have been restored (e.g. from Dropbox), without
    # re-extracting the multi-GB nuScenes dataset.
    python baseline/export_onnx.py \
        --config $CFG/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
        --model-path compression/results/pruning/ratio_25/pruned_model_25_recalib.pt \
        --variant pruned25 --synthetic-input \
        --out compression/results/onnx_export/pruned25/
"""

import argparse
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmdet3d.apis import init_model
from mmdet3d.structures import Det3DDataSample
from mmengine.config import Config
from mmengine.runner import Runner

OUTPUT_NAMES = ['heatmap', 'reg', 'height', 'dim', 'rot', 'vel']


class PillarEncoderONNX(nn.Module):
    """Wraps PFN layers for ONNX tracing.

    Receives pre-processed pillar features (relative features appended and
    mask applied on CPU via forward hook) and runs only the MLP chain.
    Mirrors Autoware's PillarFeatureNetONNX split but works with the
    standard PillarFeatureNet checkpoint.
    """

    def __init__(self, pfn_layers: nn.ModuleList) -> None:
        super().__init__()
        self.pfn_layers = pfn_layers

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        for pfn in self.pfn_layers:
            input_features = pfn(input_features)
        return input_features


class BackboneNeckHeadONNX(nn.Module):
    """Wraps SECOND backbone + SECONDFPN neck + all 6 CenterHead task heads.

    Concatenates per-task predictions channel-wise so the ONNX has one
    output tensor per type spanning all 10 nuScenes classes.
    """

    def __init__(self, backbone, neck, head) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        feat = self.head.shared_conv(
            x[0] if isinstance(x, (list, tuple)) else x
        )
        preds: List[Dict[str, torch.Tensor]] = [
            task_head(feat) for task_head in self.head.task_heads
        ]
        return (
            torch.cat([p['heatmap'] for p in preds], dim=1),
            torch.cat([p['reg'] for p in preds], dim=1),
            torch.cat([p['height'] for p in preds], dim=1),
            torch.cat([p['dim'] for p in preds], dim=1),
            torch.cat([p['rot'] for p in preds], dim=1),
            torch.cat([p['vel'] for p in preds], dim=1),
        )


def _infer_point_dim(cfg: Config) -> int:
    """Infer raw point feature dim from LoadPointsFromFile's use_dim/load_dim.

    Falls back to 5 (x, y, z, intensity, timestamp — standard for nuScenes)
    if the transform isn't found, e.g. in a minimal/inference-only config.
    """
    pipelines = []
    if 'test_pipeline' in cfg:
        pipelines.append(cfg.test_pipeline)
    try:
        pipelines.append(cfg.test_dataloader.dataset.pipeline)
    except (KeyError, AttributeError):
        pass

    for pipeline in pipelines:
        for transform in pipeline:
            if transform.get('type') == 'LoadPointsFromFile':
                use_dim = transform.get('use_dim', transform.get('load_dim', 5))
                return use_dim if isinstance(use_dim, int) else len(use_dim)
    return 5


def create_synthetic_batch(
    cfg: Config, batch_size: int = 1,
    num_points: int = 20000, device: str = 'cuda:0'
) -> dict:
    """Generate a synthetic point cloud batch for ONNX export tracing.

    ONNX export only needs tensors with correct shapes/dtypes to trace the
    forward pass — not real data. This avoids needing /data/nuscenes/ (a
    multi-GB extraction) when only exporting ONNX, e.g. on a fresh pod with
    just the conda env + checkpoint/.pt files restored from Dropbox.

    Points are uniform-random within the model's point_cloud_range; feature
    dim is inferred via _infer_point_dim (default 5: x,y,z,intensity,
    timestamp). data_samples are empty Det3DDataSample placeholders — the
    voxelize() call only needs them for batch-size bookkeeping, not metadata,
    since export never reaches extract_feat()/pts_bbox_head (which are the
    only callers that need populated metainfo).
    """
    voxel_layer = cfg.model.get('data_preprocessor', {}).get('voxel_layer', {})
    pc_range = voxel_layer.get(
        'point_cloud_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    )
    point_dim = _infer_point_dim(cfg)
    print(f'[synthetic] point_cloud_range={pc_range}  point_dim={point_dim}  '
          f'num_points={num_points}')

    points_list = []
    for _ in range(batch_size):
        xyz = torch.rand(num_points, 3, device=device)
        xyz[:, 0] = xyz[:, 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        xyz[:, 1] = xyz[:, 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        xyz[:, 2] = xyz[:, 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
        extra = torch.rand(num_points, max(point_dim - 3, 0), device=device)
        points_list.append(torch.cat([xyz, extra], dim=1))

    data_samples = [Det3DDataSample() for _ in range(batch_size)]
    return {'inputs': {'points': points_list}, 'data_samples': data_samples}


def capture_pfn_input(model: nn.Module, voxel_dict: dict) -> torch.Tensor:
    """Capture pre-PFN features via forward pre-hook on pfn_layers[0].

    Fires after relative feature concatenation and mask application inside
    PillarFeatureNet.forward() — exactly the tensor that enters the MLP.
    """
    captured: dict = {}

    def _hook(module, args) -> None:
        captured['features'] = args[0].detach()

    handle = model.pts_voxel_encoder.pfn_layers[0].register_forward_pre_hook(
        _hook
    )
    try:
        with torch.no_grad():
            model.pts_voxel_encoder(
                voxel_dict['voxels'].cuda(),
                voxel_dict['num_points'].cuda(),
                voxel_dict['coors'].cuda(),
            )
    finally:
        handle.remove()

    return captured['features']


def compute_bev_features(model: nn.Module, voxel_dict: dict) -> torch.Tensor:
    """Run encoder + pillar scatter → BEV pseudo-image [B, 64, H, W]."""
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


def export_encoder(model: nn.Module, voxel_dict: dict, out_dir: str, variant: str) -> None:
    pfn_input = capture_pfn_input(model, voxel_dict)
    enc = PillarEncoderONNX(model.pts_voxel_encoder.pfn_layers).cuda().eval()

    out_path = os.path.join(
        out_dir, f'pts_voxel_encoder_centerpoint_{variant}.onnx'
    )
    torch.onnx.export(
        enc,
        (pfn_input,),
        f=out_path,
        input_names=['input_features'],
        output_names=['pillar_features'],
        dynamic_axes={
            'input_features': {0: 'num_voxels', 1: 'num_max_points'},
            'pillar_features': {0: 'num_voxels'},
        },
        opset_version=11,
    )
    print(f'Saved encoder ONNX: {out_path}')


def export_backbone_neck_head(
    model: nn.Module, voxel_dict: dict, out_dir: str, variant: str
) -> None:
    bev = compute_bev_features(model, voxel_dict)
    bnk = BackboneNeckHeadONNX(
        model.pts_backbone, model.pts_neck, model.pts_bbox_head
    ).cuda().eval()

    dyn: dict = {'spatial_features': {0: 'batch_size', 2: 'H', 3: 'W'}}
    for name in OUTPUT_NAMES:
        dyn[name] = {0: 'batch_size', 2: 'H', 3: 'W'}

    out_path = os.path.join(
        out_dir, f'pts_backbone_neck_head_centerpoint_{variant}.onnx'
    )
    torch.onnx.export(
        bnk,
        (bev,),
        f=out_path,
        input_names=['spatial_features'],
        output_names=OUTPUT_NAMES,
        dynamic_axes=dyn,
        opset_version=11,
    )
    print(f'Saved backbone-neck-head ONNX: {out_path}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Export 10-class CenterPoint to Autoware two-ONNX format'
    )
    p.add_argument('--config', required=True, help='mmdet3d config path')
    p.add_argument(
        '--checkpoint',
        help='FP32 .pth checkpoint — architecture matches --config '
             '(loaded via init_model). Use for the FP32 baseline.'
    )
    p.add_argument(
        '--model-path',
        help='Pruned/distilled full model object (.pt, saved via '
             'torch.save(model, path)) — architecture differs from '
             '--config, loaded directly via torch.load(). '
             'Mutually exclusive with --checkpoint.'
    )
    p.add_argument(
        '--variant', default='custom',
        help='Suffix for output filenames, e.g. pruned25, '
             'distilled25 (default: custom)'
    )
    p.add_argument('--out', default='baseline/results/onnx_export/')
    p.add_argument(
        '--synthetic-input', action='store_true',
        help='Generate a synthetic point cloud batch for tracing '
             'instead of loading from cfg.test_dataloader — avoids '
             'needing /data/nuscenes/ extracted. See module docstring.'
    )
    p.add_argument(
        '--num-points', type=int, default=20000,
        help='Points per synthetic point cloud (default: 20000, '
             'only used with --synthetic-input)'
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.checkpoint and args.model_path:
        raise ValueError('--checkpoint and --model-path are mutually exclusive')

    cfg = Config.fromfile(args.config)

    if args.model_path:
        print(f'Loading compressed model object from {args.model_path}...')
        model = torch.load(args.model_path, map_location='cuda:0')
        model.cuda()
    elif args.checkpoint:
        model = init_model(cfg, args.checkpoint, device='cuda:0')
    else:
        raise ValueError('One of --checkpoint or --model-path is required')
    model.eval()

    if args.synthetic_input:
        print('Generating synthetic point cloud batch for tracing '
              '(no dataset required)...')
        batch = create_synthetic_batch(
            cfg, batch_size=1, num_points=args.num_points
        )
    else:
        dataloader = Runner.build_dataloader(cfg.test_dataloader)
        batch = next(iter(dataloader))

    voxel_dict = model.data_preprocessor.voxelize(
        batch['inputs']['points'], batch
    )

    export_encoder(model, voxel_dict, args.out, args.variant)
    export_backbone_neck_head(model, voxel_dict, args.out, args.variant)
    print('Export complete.')


if __name__ == '__main__':
    main()

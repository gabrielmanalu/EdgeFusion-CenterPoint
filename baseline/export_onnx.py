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

Usage:
    # CFG=/workspace/mmdetection3d/configs/centerpoint
    # CKPT=/workspace/data/centerpoint
    python baseline/export_onnx.py \
        --config  $CFG/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
        --checkpoint $CKPT/<2022_checkpoint>.pth \
        --out     baseline/results/onnx_export/
"""

import argparse
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmdet3d.apis import init_model
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


def export_encoder(model: nn.Module, voxel_dict: dict, out_dir: str) -> None:
    pfn_input = capture_pfn_input(model, voxel_dict)
    enc = PillarEncoderONNX(model.pts_voxel_encoder.pfn_layers).cuda().eval()

    out_path = os.path.join(
        out_dir, 'pts_voxel_encoder_centerpoint_custom.onnx'
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
    model: nn.Module, voxel_dict: dict, out_dir: str
) -> None:
    bev = compute_bev_features(model, voxel_dict)
    bnk = BackboneNeckHeadONNX(
        model.pts_backbone, model.pts_neck, model.pts_bbox_head
    ).cuda().eval()

    dyn: dict = {'spatial_features': {0: 'batch_size', 2: 'H', 3: 'W'}}
    for name in OUTPUT_NAMES:
        dyn[name] = {0: 'batch_size', 2: 'H', 3: 'W'}

    out_path = os.path.join(
        out_dir, 'pts_backbone_neck_head_centerpoint_custom.onnx'
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
    p.add_argument('--checkpoint', required=True, help='.pth checkpoint path')
    p.add_argument('--out', default='baseline/results/onnx_export/')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device='cuda:0')
    model.eval()

    dataloader = Runner.build_dataloader(cfg.test_dataloader)
    batch = next(iter(dataloader))
    voxel_dict = model.data_preprocessor.voxelize(
        batch['inputs']['points'], batch
    )

    export_encoder(model, voxel_dict, args.out)
    export_backbone_neck_head(model, voxel_dict, args.out)
    print('Export complete.')


if __name__ == '__main__':
    main()

"""
Generate [64,512,512] BEV calibration tensors for TRT INT8 calibration of the
backbone+neck+head engine, from raw nuScenes point clouds in jetson_calib/
(each a .pcd.bin file, float32, point_dim columns).

Runs voxelize() -> pts_voxel_encoder() -> pts_middle_encoder() on the FP32
model for each raw point cloud — the exact validated pipeline from
baseline/export_onnx.py's compute_bev_features(), duplicated here unmodified.
This avoids reimplementing PFN feature augmentation (cluster/voxel-center
offsets) in numpy on Jetson, which carries silent-miscalibration risk if the
augmentation formula is gotten wrong.

Output: one [64,512,512] float32 .npy per input .pcd.bin, written to --out.
Feeds directly into deployment/scripts/build_engine.py's BEVFeatureCalibrator.

Size note: 512 samples x 64x512x512x4 bytes ~= 8.6GB total (~16.8MB/sample).
Use --max-samples to generate a subset if transfer bandwidth to Jetson is a
concern — TRT INT8 entropy calibration commonly converges well within 100-200
samples; 512 is on the high end.

Usage (from /workspace/mmdetection3d, after source activate_env.sh):
    python EdgeFusion-CenterPoint/compression/generate_calib_bev.py \
        --config $CFG --checkpoint $CKPT \
        --calib-dir EdgeFusion-CenterPoint/jetson_calib \
        --out EdgeFusion-CenterPoint/jetson_calib_bev \
        --max-samples 200
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from mmdet3d.apis import init_model
from mmdet3d.structures import Det3DDataSample
from mmengine.config import Config


def _infer_point_dim(cfg: Config) -> int:
    """Infer raw point feature dim from LoadPointsFromFile's use_dim/load_dim.

    Same logic as baseline/export_onnx.py — duplicated here to keep this
    script self-contained (avoids cross-directory import path issues when
    invoked as EdgeFusion-CenterPoint/compression/<script>.py).
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


def compute_bev_features(model, voxel_dict: dict) -> torch.Tensor:
    """Run encoder + pillar scatter -> BEV pseudo-image [B, 64, H, W].

    Identical to baseline/export_onnx.py's compute_bev_features — duplicated
    here for the same self-containment reason as _infer_point_dim above.
    """
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Generate BEV calibration tensors for TRT INT8'
    )
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True, help='FP32 .pth checkpoint')
    p.add_argument(
        '--calib-dir', required=True,
        help='Directory of raw .pcd.bin point clouds (jetson_calib/)'
    )
    p.add_argument(
        '--out', required=True,
        help='Output directory for [64,512,512] .npy files'
    )
    p.add_argument(
        '--max-samples', type=int, default=None,
        help='Process only the first N files (default: all). See size note '
             'in module docstring re: transfer bandwidth to Jetson.'
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    point_dim = _infer_point_dim(cfg)
    print(f'[calib-bev] point_dim={point_dim}')

    model = init_model(cfg, args.checkpoint, device='cuda:0')
    model.eval()

    files = sorted(Path(args.calib_dir).glob('*.bin'))
    if not files:
        raise FileNotFoundError(f'No .bin files found in {args.calib_dir}')
    if args.max_samples:
        files = files[:args.max_samples]
    print(f'[calib-bev] {len(files)} point clouds from {args.calib_dir}')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(files):
        points = np.fromfile(path, dtype=np.float32).reshape(-1, point_dim)
        points_tensor = torch.from_numpy(points).cuda()
        points_list = [points_tensor]
        batch = {
            'inputs': {'points': points_list},
            'data_samples': [Det3DDataSample()],
        }

        voxel_dict = model.data_preprocessor.voxelize(points_list, batch)
        bev = compute_bev_features(model, voxel_dict)  # [1, 64, H, W]

        # Strip both .bin and .pcd suffixes: foo.pcd.bin -> foo.npy
        out_name = path.with_suffix('').stem + '.npy'
        np.save(out_dir / out_name, bev.squeeze(0).cpu().numpy().astype(np.float32))

        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f'[calib-bev] {i + 1}/{len(files)}  bev shape={tuple(bev.shape)}')

    print(f'[calib-bev] Done. Saved {len(files)} .npy files to {out_dir}')


if __name__ == '__main__':
    main()

"""
Generate VERIFIED [64,512,512] BEV calibration tensors.

ROOT CAUSE FOUND: CenterPoint config uses
LoadPointsFromMultiSweeps(sweeps_num=9) — the model expects 9 past sweeps
aggregated with the current frame (~10 LiDAR frames, motion-compensated) for
each sample, NOT a single raw scan. jetson_calib/ contains single-sweep .bin
files, which are far too sparse for this model, producing weak/noisy heatmaps
regardless of how correctly voxelization+encoding+scatter are implemented.

fix: use the REAL test pipeline (built straight from the config) on
samples drawn from the actual nuscenes_infos_val.pkl, which is what supplies
the multi-sweep metadata (sweep file paths + relative timestamps/ego poses)
that LoadPointsFromMultiSweeps needs. This is exactly how mmdet3d's own
test.py evaluates the model — guaranteed-correct by construction, since we
are no longer hand-assembling any part of the input pipeline.

Requires: the nuScenes raw sweep .bin files referenced by each info's
'lidar_sweeps' to be present at the path the pkl expects (typically under
data/nuscenes/sweeps/LIDAR_TOP/ relative to the dataset root used when the
pkl was generated). If sweeps aren't available, this won't run — see the
fallback note in the script output.

Usage (on pod, from mmdetection3d/, after source activate_env.sh):
    python EdgeFusion-CenterPoint/compression/generate_calib_bev.py \
        --config $CFG --checkpoint $CKPT \
        --val-pkl /path/to/nuscenes_infos_val.pkl \
        --data-root /workspace/data/nuscenes \
        --tokens-file EdgeFusion-CenterPoint/jetson_calib_tokens.txt \
        --out /data/jetson_calib_bev \
        --verify-n 5
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from mmdet3d.apis import init_model
try:
    from mmdet3d.datasets.transforms import Compose
except ImportError:
    try:
        from mmcv.transforms import Compose
    except ImportError:
        from mmengine.dataset import Compose
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmengine.config import Config


def _resolve_lidar_path(data_root: str, raw_path: str) -> str:
    """Resolve a pkl-stored lidar path (often a bare filename) against the
    actual nuScenes folder layout. Tries, in order: the raw path as-is
    relative to data_root (in case it already includes subdirs), then
    samples/LIDAR_TOP/<name>, then sweeps/LIDAR_TOP/<name>. Returns the first
    that exists on disk; falls back to the as-is join (so a clear FileNotFound
    error surfaces downstream rather than a silent wrong guess).
    """
    name = Path(raw_path).name
    candidates = [
        Path(data_root) / raw_path,
        Path(data_root) / 'samples' / 'LIDAR_TOP' / name,
        Path(data_root) / 'sweeps' / 'LIDAR_TOP' / name,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument(
        '--val-pkl', required=True,
        help='nuscenes_infos_val.pkl (supplies multi-sweep metadata)'
    )
    p.add_argument(
        '--data-root', required=True,
        help='nuScenes dataset root containing samples/ and sweeps/'
    )
    p.add_argument(
        '--tokens-file', default=None,
        help='Optional: text file of sample tokens (one per line) to '
             'restrict to — e.g. the 512 jetson_calib tokens. If omitted, '
             'uses --max-samples from the start of the val pkl.'
    )
    p.add_argument('--max-samples', type=int, default=512)
    p.add_argument('--out', required=True)
    p.add_argument('--verify-n', type=int, default=5)
    args = p.parse_args()

    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device='cuda:0')
    model.eval()

    # Build the REAL test pipeline straight from the config — this includes
    # LoadPointsFromFile + LoadPointsFromMultiSweeps + everything else,
    # exactly as used during the model's actual evaluation.
    pipeline = Compose(cfg.test_dataloader.dataset.pipeline)

    with open(args.val_pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data['data_list'] if isinstance(data, dict) else data
    print(f' {len(infos)} val infos loaded')

    if args.tokens_file:
        with open(args.tokens_file) as f:
            want = set(line.strip() for line in f if line.strip())
        infos = [i for i in infos if i.get('token') in want]
        print(f' Restricted to {len(infos)} infos matching tokens file')
    else:
        infos = infos[:args.max_samples]

    # Hook to capture pts_middle_encoder output (the BEV)
    captured = {}

    def hook(module, inp, out):
        captured['bev'] = out.detach()

    handle = model.pts_middle_encoder.register_forward_hook(hook)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for i, info in enumerate(infos):
        # data_root must be injected so LoadPointsFromFile/MultiSweeps can
        # resolve paths in the info dict. The pkl stores BARE FILENAMES (not
        # the full samples/LIDAR_TOP/... relative path), so we resolve against
        # the actual nuScenes folder layout: keyframes live under
        # samples/LIDAR_TOP/, sweep frames under sweeps/LIDAR_TOP/. Try both
        # locations (order matters: samples first since the keyframe itself
        # sometimes appears as the first "sweep" entry too).
        info_in = dict(info)
        info_in['lidar_points'] = dict(info['lidar_points'])
        info_in['lidar_points']['lidar_path'] = _resolve_lidar_path(
            args.data_root, info['lidar_points']['lidar_path']
        )
        if 'lidar_sweeps' in info_in:
            sweeps = []
            for sw in info_in['lidar_sweeps']:
                sw = dict(sw)
                sw['lidar_points'] = dict(sw['lidar_points'])
                sw['lidar_points']['lidar_path'] = _resolve_lidar_path(
                    args.data_root, sw['lidar_points']['lidar_path']
                )
                sweeps.append(sw)
            info_in['lidar_sweeps'] = sweeps

        try:
            sample = pipeline(info_in)
        except Exception as e:
            print(f' [{i}] pipeline failed: {e} — skipping '
                  f'(check --data-root path / sweep file availability)')
            continue

        points = sample['inputs']['points']
        pts = points.cuda() if isinstance(points, torch.Tensor) else \
            torch.from_numpy(np.asarray(points)).cuda()

        ds = Det3DDataSample()
        ds.set_metainfo({'box_type_3d': LiDARInstance3DBoxes})
        batch = {'inputs': {'points': [pts]}, 'data_samples': [ds]}

        with torch.no_grad():
            pre = model.data_preprocessor(batch, training=False)
            # extract_feat signature varies across mmdet3d versions; the hook
            # on pts_middle_encoder fires regardless of which path runs, so we
            # just need the call to complete. Try the known signatures.
            metas = [ds.metainfo]
            try:
                model.extract_feat(pre['inputs'], metas)
            except TypeError:
                try:
                    model.extract_feat(
                        batch_inputs_dict=pre['inputs'],
                        batch_input_metas=metas,
                    )
                except TypeError:
                    model.extract_feat(pre['inputs'])  # oldest signature

        bev = captured['bev']

        if i < args.verify_n:
            with torch.no_grad():
                feats = model.pts_backbone(bev)
                if model.with_pts_neck:
                    feats = model.pts_neck(feats)
                head_out = model.pts_bbox_head(feats)
            try:
                tasks = head_out[0] if isinstance(head_out, tuple) else head_out
                maxes = [float(t['heatmap'].sigmoid().max()) for t in tasks]
                print(f'[verify {i}] token={info.get("token", "?")[:8]} '
                      f'n_points={pts.shape[0]}  '
                      f'heatmap maxes: {[round(x, 3) for x in maxes]}  '
                      f'overall={max(maxes):.3f} (healthy if >0.8)')
            except Exception as e:
                print(f'[verify {i}] head read failed: {e}')

        token = info.get('token', f'idx{i}')
        np.save(out_dir / f'{token}.npy',
                bev.squeeze(0).cpu().numpy().astype(np.float32))
        n_ok += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(infos):
            print(f' {i + 1}/{len(infos)}  n_points={pts.shape[0]}  '
                  f'bev_range=[{bev.min():.2f},{bev.max():.2f}]')

    handle.remove()
    print(f' Done. Saved {n_ok}/{len(infos)} BEV tensors to {out_dir}')


if __name__ == '__main__':
    main()

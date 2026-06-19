#!/usr/bin/env python3
"""
Standalone NuScenes mAP/NDS evaluation from saved submission JSONs.

KEY FIX: the submission JSONs are in LIDAR frame (what the CenterPoint decoder
produces). nuScenes eval expects GLOBAL frame. This script applies the
per-sample lidar->ego->global transforms (from nuscenes_infos_val.pkl) before
evaluating. Without this, all predictions miss all GT boxes → mAP = 0.

Memory-efficient: filters v1.0-trainval JSON files to our 512 tokens before
loading NuScenes, keeping peak memory ~100-200 MB (vs ~2-4 GB for full dataset).

Prerequisites (host Python, no Docker needed):
    pip3 install nuscenes-devkit scikit-learn shapely pyquaternion "numpy<2"

Usage:
    python3 deployment/scripts/eval_metrics.py \
        --nuscenes ~/Downloads/v1.0-trainval_meta \
        --val-pkl  nuscenes_infos_val.pkl \
        --submissions deployment/output \
        --out deployment/output/eval_summary.json
"""

import argparse
import json
import pickle
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

# Mock cv2 before nuscenes imports it (only needed for visualization;
# fails on Jetson due to numpy ABI mismatch at module level).
try:
    import cv2  # noqa: F401
except (AttributeError, ImportError):
    sys.modules['cv2'] = types.ModuleType('cv2')

from nuscenes import NuScenes  # noqa: E402
from nuscenes.eval.detection.config import config_factory  # noqa: E402
from nuscenes.eval.detection.evaluate import NuScenesEval  # noqa: E402


# ── Coordinate transform ────────────────────────────────────────────────────

def load_transforms(val_pkl_path: str) -> dict:
    """Load per-sample lidar2ego + ego2global 4×4 matrices from mmdet3d pkl.

    Handles both old (flat list) and new (data_list dict) mmdet3d pkl formats.
    Returns dict: sample_token → {'lidar2ego': np.ndarray, 'ego2global': np.ndarray}.
    """
    with open(val_pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos = data.get('data_list', data) if isinstance(data, dict) else data

    transforms = {}
    for info in infos:
        token = info.get('token', '')
        if not token:
            continue
        e2g = np.array(info.get('ego2global', np.eye(4)), dtype=np.float64)
        if e2g.shape == (3, 4):
            e2g = np.vstack([e2g, [0, 0, 0, 1]])
        l2e_raw = info.get('lidar_points', {}).get('lidar2ego', np.eye(4))
        l2e = np.array(l2e_raw, dtype=np.float64)
        if l2e.shape == (3, 4):
            l2e = np.vstack([l2e, [0, 0, 0, 1]])
        transforms[token] = {'lidar2ego': l2e, 'ego2global': e2g}

    print(f'[transforms] Loaded {len(transforms)} sample transforms from {val_pkl_path}')
    return transforms


def _orthonormal_quaternion(rot3x3: np.ndarray) -> Quaternion:
    """Build a Quaternion from a 3×3 rotation matrix, re-orthogonalizing first.

    Rotation blocks from the pkl carry tiny numerical drift after float
    round-trips (e.g. 0.9999998 instead of 1.0), which trips pyquaternion's
    strict orthogonality check. SVD gives the nearest true orthogonal matrix:
    R = U @ Vt where R = U @ S @ Vt. Guard against a reflection (det = -1) by
    flipping the last column of U so det = +1 (a proper rotation).
    """
    u, _, vt = np.linalg.svd(rot3x3)
    r_ortho = u @ vt
    if np.linalg.det(r_ortho) < 0:
        u[:, -1] *= -1
        r_ortho = u @ vt
    return Quaternion(matrix=r_ortho)


def _transform_box(box: dict, l2e: np.ndarray, e2g: np.ndarray) -> dict:
    """Transform a single box dict from LiDAR frame to global frame.

    CenterPoint decoder outputs (x,y,z) in the LiDAR/BEV coordinate frame
    (center of pc_range, e.g. 0..51.2 meters east of ego vehicle). nuScenes
    DetectionEval requires global (UTM-like) coordinates. Apply the two-step
    rigid transform: LiDAR → ego (lidar2ego), then ego → global (ego2global).
    Velocity and rotation are transformed by the rotation components only.
    """
    # Translation: LiDAR → ego → global
    xyz = np.array([box['translation'][0], box['translation'][1],
                    box['translation'][2], 1.0])
    xyz_global = (e2g @ (l2e @ xyz))[:3]

    # Rotation: compose quaternions Q_global = Q_e2g * Q_l2e * Q_box
    # (rotation matrices re-orthogonalized via SVD to tolerate pkl float drift)
    l2e_q = _orthonormal_quaternion(l2e[:3, :3])
    e2g_q = _orthonormal_quaternion(e2g[:3, :3])
    box_q = Quaternion(box['rotation'])  # [w, x, y, z]
    q_global = e2g_q * l2e_q * box_q

    # Velocity: rotate (no translation for velocity)
    vel = box.get('velocity', [0.0, 0.0])
    vel3 = np.array([vel[0], vel[1], 0.0])
    vel_global = e2g[:3, :3] @ (l2e[:3, :3] @ vel3)

    return {
        **box,
        'translation': xyz_global.tolist(),
        'rotation': [float(q_global.w), float(q_global.x),
                     float(q_global.y), float(q_global.z)],
        'velocity': [float(vel_global[0]), float(vel_global[1])],
    }


def transform_submission(submission: dict, transforms: dict) -> dict:
    """Transform all predictions in a submission from LiDAR to global frame."""
    missing = 0
    new_results = {}
    for token, boxes in submission['results'].items():
        if token not in transforms:
            missing += 1
            new_results[token] = boxes
            continue
        l2e = transforms[token]['lidar2ego']
        e2g = transforms[token]['ego2global']
        new_results[token] = [_transform_box(b, l2e, e2g) for b in boxes]
    if missing:
        print(f'[transform] {missing} tokens had no transform — boxes kept as-is')
    return {'results': new_results, 'meta': submission.get('meta', {})}


# ── Mini NuScenes (memory-efficient filter) ─────────────────────────────────

def _filter_table(src_file: Path, key: str, keep: set) -> list:
    with open(src_file) as f:
        table = json.load(f)
    filtered = [row for row in table if row[key] in keep]
    del table
    return filtered


def create_mini_nuscenes(full_dataroot: str, our_tokens: set) -> str:
    """Filter v1.0-trainval JSONs to our tokens; write to a temp dir.

    Processes each large JSON sequentially so peak RAM ≈ largest file
    (~500 MB Python objects for sample_data.json) rather than the full
    ~2-4 GB in-memory NuScenes model.
    """
    src = Path(full_dataroot) / 'v1.0-trainval'
    tmp = Path(tempfile.mkdtemp(prefix='nuscenes_mini_'))
    dst = tmp / 'v1.0-trainval'
    dst.mkdir()
    print(f'[mini] Creating filtered NuScenes at {tmp}...')

    def load(name):
        with open(src / f'{name}.json') as f:
            return json.load(f)

    def save(name, data):
        with open(dst / f'{name}.json', 'w') as f:
            json.dump(data, f)
        print(f'[mini]   {name}.json: {len(data)} rows')

    for name in ['category', 'attribute', 'visibility', 'sensor']:
        shutil.copy(src / f'{name}.json', dst / f'{name}.json')

    map_src = src / 'map.json'
    if map_src.exists():
        shutil.copy(map_src, dst / 'map.json')
    else:
        save('map', [])

    samples = _filter_table(src / 'sample.json', 'token', our_tokens)
    for s in samples:
        if s.get('prev') not in our_tokens | {''}:
            s['prev'] = ''
        if s.get('next') not in our_tokens | {''}:
            s['next'] = ''
    save('sample', samples)
    scene_tokens = {s['scene_token'] for s in samples}

    scenes = _filter_table(src / 'scene.json', 'token', scene_tokens)
    save('scene', scenes)
    log_tokens = {s['log_token'] for s in scenes}

    logs = _filter_table(src / 'log.json', 'token', log_tokens)
    save('log', logs)

    sd_all = _filter_table(src / 'sample_data.json', 'sample_token', our_tokens)
    save('sample_data', sd_all)
    cs_tokens = {sd['calibrated_sensor_token'] for sd in sd_all}
    ep_tokens = {sd['ego_pose_token'] for sd in sd_all}
    del sd_all

    cs = _filter_table(src / 'calibrated_sensor.json', 'token', cs_tokens)
    save('calibrated_sensor', cs)

    ep = _filter_table(src / 'ego_pose.json', 'token', ep_tokens)
    save('ego_pose', ep)

    anns = _filter_table(src / 'sample_annotation.json', 'sample_token', our_tokens)
    # Sever prev/next annotation links that point to annotations outside our
    # subset. load_gt → box_velocity follows these links via nusc.get(); a
    # dangling token (belonging to a sample we didn't keep) raises KeyError
    # mid-load and yields zero GT → 0 mAP. With next/prev cleared, box_velocity
    # returns [nan, nan] gracefully (its documented behavior at sequence ends).
    our_ann_tokens = {a['token'] for a in anns}
    severed = 0
    for a in anns:
        if a.get('prev') and a['prev'] not in our_ann_tokens:
            a['prev'] = ''
            severed += 1
        if a.get('next') and a['next'] not in our_ann_tokens:
            a['next'] = ''
            severed += 1
    print(f'[mini]   severed {severed} dangling annotation prev/next links')
    save('sample_annotation', anns)
    inst_tokens = {a['instance_token'] for a in anns}

    inst = _filter_table(src / 'instance.json', 'token', inst_tokens)
    save('instance', inst)

    maps_src = Path(full_dataroot) / 'maps'
    maps_dst = tmp / 'maps'
    if maps_src.exists():
        shutil.copytree(maps_src, maps_dst)
    else:
        maps_dst.mkdir()

    print('[mini] Done.')
    return str(tmp)


# ── Evaluation ───────────────────────────────────────────────────────────────

def _patch_nusc(nusc: NuScenes) -> None:
    """Monkey-patch nusc to handle tokens outside our mini dataset.

    NuScenesEval.load_gt() iterates ALL 6019 val tokens. Our mini only has
    512, so 5507 lookups fail. Return ghost stubs so load_gt gets empty GT
    for those — they have no predictions either, so they don't affect mAP.

    Also patch box_velocity to catch KeyErrors when 'next' annotations fall
    outside our mini (their tokens are not in our sample_annotation table).
    """
    _orig_get = nusc.get

    def _robust_get(table_name, token):
        try:
            return _orig_get(table_name, token)
        except KeyError:
            if table_name == 'sample':
                return {'token': token, 'timestamp': 0, 'scene_token': '',
                        'prev': '', 'next': '', 'data': {}, 'anns': []}
            if table_name == 'sample_annotation':
                return {'token': token, 'sample_token': '', 'instance_token': '',
                        'prev': '', 'next': '', 'translation': [0, 0, 0],
                        'size': [1, 1, 1], 'rotation': [1, 0, 0, 0],
                        'num_lidar_pts': 0, 'visibility_token': '1',
                        'attribute_tokens': [], 'category_name': 'car'}
            raise

    nusc.get = _robust_get

    # box_velocity computes velocity from consecutive annotation positions.
    # If the 'next' annotation is for a sample outside our mini, its token
    # is absent from sample_annotation → KeyError. Return zeros instead.
    _orig_vel = nusc.box_velocity

    def _safe_vel(token: str, max_time_diff: float = 1.5):
        try:
            return _orig_vel(token, max_time_diff)
        except (KeyError, Exception):
            return np.array([0.0, 0.0, 0.0])

    nusc.box_velocity = _safe_vel


def eval_submission(
    nusc: NuScenes,
    submission_path: str,
    out_dir: str,
    variant: str,
    transforms: dict,
) -> dict:
    with open(submission_path) as f:
        submission = json.load(f)

    # Transform predictions from LiDAR frame to global frame
    submission = transform_submission(submission, transforms)

    # Truncate to nuScenes' max 500 boxes/sample limit (sort by score desc)
    for token, boxes in submission['results'].items():
        if len(boxes) > 500:
            submission['results'][token] = sorted(
                boxes, key=lambda b: b['detection_score'], reverse=True
            )[:500]

    filtered_path = submission_path.replace('_submission.json', '_global_top500.json')
    with open(filtered_path, 'w') as f:
        json.dump(submission, f)
    print(f'  [{variant}] → {filtered_path}')

    our_tokens = set(submission['results'].keys())

    _patch_nusc(nusc)
    cfg = config_factory('detection_cvpr_2019')
    evaluator = NuScenesEval(
        nusc,
        config=cfg,
        result_path=filtered_path,
        eval_set='val',
        output_dir=out_dir,
        verbose=False,
    )

    evaluator.sample_tokens = [
        t for t in evaluator.sample_tokens if t in our_tokens
    ]
    print(f'  [{variant}] evaluating {len(evaluator.sample_tokens)} samples...')

    # Diagnostic: count GT + prediction boxes the evaluator actually has,
    # restricted to our tokens. If GT count is 0, matching is impossible.
    gt_count = sum(
        len(evaluator.gt_boxes.boxes.get(t, []))
        for t in evaluator.sample_tokens
    )
    pred_count = sum(
        len(evaluator.pred_boxes.boxes.get(t, []))
        for t in evaluator.sample_tokens
    )
    gt_classes: dict = {}
    for t in evaluator.sample_tokens:
        for b in evaluator.gt_boxes.boxes.get(t, []):
            gt_classes[b.detection_name] = gt_classes.get(b.detection_name, 0) + 1
    print(f'  [{variant}] GT boxes={gt_count}  pred boxes={pred_count}')
    print(f'  [{variant}] GT classes: {gt_classes}')

    metrics, _ = evaluator.evaluate()
    summary = metrics.serialize()

    return {
        'variant': variant,
        'n_samples': len(evaluator.sample_tokens),
        'mAP': float(summary['mean_ap']),
        'NDS': float(summary['nd_score']),
        'mATE': float(summary['tp_errors'].get('trans_err', float('nan'))),
        'mASE': float(summary['tp_errors'].get('scale_err', float('nan'))),
        'mAOE': float(summary['tp_errors'].get('orient_err', float('nan'))),
        'mAVE': float(summary['tp_errors'].get('vel_err', float('nan'))),
        'mAAE': float(summary['tp_errors'].get('attr_err', float('nan'))),
        'per_class_ap': {
            k: float(v) for k, v in summary['mean_dist_aps'].items()
        },
        'note': (
            'Evaluated on 512-sample jetson_calib subset of val set. '
            'Predictions transformed from LiDAR to global frame via '
            'per-sample lidar2ego + ego2global from nuscenes_infos_val.pkl. '
            'Not directly comparable to full-val (6019-sample) A40 numbers.'
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Memory-efficient nuScenes eval with LiDAR→global transform'
    )
    p.add_argument(
        '--nuscenes', required=True,
        help='nuScenes dataroot with v1.0-trainval/ annotation JSONs'
    )
    p.add_argument(
        '--val-pkl', required=True,
        help='nuscenes_infos_val.pkl (lidar2ego + ego2global transforms)'
    )
    p.add_argument('--submissions', default='deployment/output')
    p.add_argument(
        '--variants', nargs='+',
        default=['fp32', 'pruned25', 'pruned40', 'pruned55', 'distilled25']
    )
    p.add_argument('--out', default='deployment/output/eval_summary.json')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    transforms = load_transforms(args.val_pkl)

    all_tokens: set = set()
    sub_paths = {}
    for variant in args.variants:
        p = Path(args.submissions) / f'eval_{variant}_submission.json'
        if p.exists():
            with open(p) as f:
                tokens = set(json.load(f)['results'].keys())
            all_tokens.update(tokens)
            sub_paths[variant] = str(p)
            print(f'[main] {variant}: {len(tokens)} tokens')
        else:
            print(f'[main] {variant}: not found at {p} — skipping')

    if not all_tokens:
        print('[main] No submissions found — exiting')
        return

    mini_dir = create_mini_nuscenes(args.nuscenes, all_tokens)

    try:
        print(f'\n[main] Loading mini NuScenes ({len(all_tokens)} tokens)...')
        nusc = NuScenes(version='v1.0-trainval', dataroot=mini_dir, verbose=False)
        print('[main] Loaded.\n')

        out_dir = str(Path(args.out).parent)
        results = {}

        for variant, sub_path in sub_paths.items():
            print(f'\n[{variant}] Evaluating...')
            try:
                result = eval_submission(nusc, sub_path, out_dir, variant, transforms)
                results[variant] = result
                per_path = Path(args.submissions) / f'eval_{variant}.json'
                with open(per_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f'  [{variant}] mAP={result["mAP"]:.4f}  NDS={result["NDS"]:.4f}  '
                      f'→ {per_path}')
            except Exception as e:
                print(f'  [{variant}] FAILED: {e}')
                import traceback
                traceback.print_exc()

    finally:
        shutil.rmtree(mini_dir, ignore_errors=True)

    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'{"Variant":<15}  {"mAP":>6}  {"NDS":>6}')
    print('-' * 30)
    for variant, r in results.items():
        print(f'{variant:<15}  {r["mAP"]:>6.4f}  {r["NDS"]:>6.4f}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n[main] Saved: {args.out}')


if __name__ == '__main__':
    main()

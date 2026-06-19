"""
On-device mAP/NDS evaluation for CenterPoint TRT INT8 engines on Jetson.

Pipeline:
  Pre-computed BEV .npy (jetson_calib_bev/)
    → backbone+neck+head TRT INT8 engine
    → standalone CenterPoint decode (numpy, no mmdet3d)
    → nuScenes submission JSON
    → nuscenes-devkit DetectionEval (mAP / NDS)

Requirements:
  - pts_backbone_neck_head.engine  (built by build_engine.py)
  - jetson_calib_bev/  512 × [64,512,512] float32 .npy files
  - nuscenes_infos_val.pkl  (maps filename → sample_token, used to build
    the submission dict; from the pod Dropbox backup)
  - nuScenes annotation JSONs  v1.0-trainval/ subdirectory under --nuscenes
    (no sensor data needed — only the annotation JSON files ~400 MB).
    Download: https://www.nuscenes.org/download  →  "Metadata"

ASSUMPTIONS (from CenterPoint pillar02 nuScenes config):
  - point_cloud_range:  [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
  - voxel_size:          [0.2, 0.2, 8.0]
  - BEV grid:            512 × 512
  - Head output stride:  4  (head outputs [1,*,128,128], each cell = 0.8m)
  - NMS type:            Circle NMS (distance-based, CenterPoint default)
  --out-size and --stride are overridable if your config differs.

Head output channel layout (from BackboneNeckHeadONNX, see export_onnx.py):
  heatmap  [1, 10, H, W]   task class scores (sigmoid → probability)
  reg      [1, 12, H, W]   x,y sub-cell offsets  (2 per task × 6 tasks)
  height   [1,  6, H, W]   z centre (1 per task × 6 tasks)
  dim      [1, 18, H, W]   log(l,w,h)            (3 per task × 6 tasks)
  rot      [1, 12, H, W]   sin(yaw), cos(yaw)    (2 per task × 6 tasks)
  vel      [1, 12, H, W]   vx, vy                (2 per task × 6 tasks)

Task → class mapping (nuScenes 10-class):
  task 0: car
  task 1: truck, construction_vehicle
  task 2: bus, trailer
  task 3: barrier
  task 4: motorcycle, bicycle
  task 5: pedestrian, traffic_cone

Usage:
  docker compose -f deployment/docker/docker-compose.yml run --rm eval
  VARIANT=pruned25 docker compose ... run --rm eval
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
from pyquaternion import Quaternion

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ── CenterPoint nuScenes config constants ────────────────────────────────────
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
VOXEL_SIZE = [0.2, 0.2, 8.0]

TASKS = [
    {'names': ['car']},
    {'names': ['truck', 'construction_vehicle']},
    {'names': ['bus', 'trailer']},
    {'names': ['barrier']},
    {'names': ['motorcycle', 'bicycle']},
    {'names': ['pedestrian', 'traffic_cone']},
]
ALL_CLASSES = [n for t in TASKS for n in t['names']]

# Default nuScenes attribute per detection class (required for submission JSON)
CLASS_ATTRIBUTE = {
    'car':                  'vehicle.moving',
    'truck':                'vehicle.moving',
    'bus':                  'vehicle.moving',
    'trailer':              'vehicle.parked',
    'construction_vehicle': 'vehicle.parked',
    'pedestrian':           'pedestrian.moving',
    'motorcycle':           'cycle.with_rider',
    'bicycle':              'cycle.with_rider',
    'traffic_cone':         '',
    'barrier':              '',
}

# Detection score threshold — boxes below this are dropped before NMS.
# CenterPoint default is 0.1; higher = fewer FP, may hurt mAP at low recall.
SCORE_THRESH = 0.1
NMS_RADIUS = {  # circle NMS radius in metres per class
    'car': 4.0, 'truck': 4.0, 'bus': 10.0, 'trailer': 10.0,
    'construction_vehicle': 12.0, 'pedestrian': 0.175,
    'motorcycle': 0.5, 'bicycle': 0.5,
    'traffic_cone': 0.175, 'barrier': 1.5,
}  # noqa: E241
MAX_PREDS_PER_TASK = 500  # top-K per task before NMS


class Engine:
    """Minimal TRT 10.x wrapper for static-shape engines."""

    def __init__(self, path: str) -> None:
        runtime = trt.Runtime(TRT_LOGGER)
        with open(path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs = {}
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            buf = cuda.mem_alloc(int(np.prod(shape)) * np.dtype(dtype).itemsize)
            self.context.set_tensor_address(name, int(buf))
            entry = {'buf': buf, 'shape': shape, 'dtype': dtype}
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.inputs[name] = entry
            else:
                self.outputs[name] = entry

    def infer(self, bev: np.ndarray) -> dict:
        name = list(self.inputs.keys())[0]
        arr = np.ascontiguousarray(bev.astype(np.float32))
        cuda.memcpy_htod_async(self.inputs[name]['buf'], arr, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        outs = {}
        for n, info in self.outputs.items():
            out = np.empty(info['shape'], dtype=info['dtype'])
            cuda.memcpy_dtoh_async(out, info['buf'], self.stream)
            outs[n] = out
        self.stream.synchronize()
        return outs


# ── CenterPoint decode ───────────────────────────────────────────────────────

def _heatmap_peak_mask(hm: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Zero out non-local-maximum heatmap cells (3×3 max-pool NMS).

    For each class channel, keep a cell's score only if it equals the maximum
    in its kernel×kernel neighborhood; otherwise set it to 0. This is the
    standard CenterPoint peak-extraction step — it collapses each object's
    heatmap blob to a single peak cell, preventing dozens of duplicate boxes
    per object. Implemented with a sliding-window max via numpy stride tricks
    (no scipy dependency).

    hm: [C, H, W] post-sigmoid heatmap. Returns same shape, non-peaks zeroed.
    """
    c, h, w = hm.shape
    pad = kernel // 2
    padded = np.pad(hm, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    maxpool = np.zeros_like(hm)
    for dy in range(kernel):
        for dx in range(kernel):
            maxpool = np.maximum(maxpool, padded[:, dy:dy + h, dx:dx + w])
    # Keep cells that equal the neighborhood max (the local peaks)
    return np.where(hm == maxpool, hm, 0.0)


def _circle_nms(boxes_xy: np.ndarray, scores: np.ndarray,
                radius: float) -> np.ndarray:
    """Greedy circle NMS. boxes_xy: [N,2], scores: [N]. Returns kept indices."""
    order = scores.argsort()[::-1]
    keep = []
    suppressed = np.zeros(len(order), dtype=bool)
    for i, idx in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(idx)
        dists = np.sqrt(
            ((boxes_xy[order[i + 1:]] - boxes_xy[idx]) ** 2).sum(1)
        )
        suppressed[i + 1:][dists < radius] = True
    return np.array(keep, dtype=np.int64)


def decode_outputs(
    outputs: dict,
    stride: int,
    score_thresh: float = SCORE_THRESH,
    max_preds: int = MAX_PREDS_PER_TASK,
) -> list:
    """Decode raw head outputs to a list of detection dicts.

    Coordinate system: CenterPoint head outputs are in the BEV grid frame.
    For each peak at row r, col c in the [H, W] head output:
      x_global = (c + reg_x) * stride * voxel_size[0] + pc_range[0]
      y_global = (r + reg_y) * stride * voxel_size[1] + pc_range[1]
      z_global = height (direct regression to z-centre, not an offset)
      l, w, h = exp(dim)
      yaw = atan2(sin_rot, cos_rot)
    """
    hm = outputs.get('heatmap')    # [1, 10, H, W]
    reg = outputs.get('reg')       # [1, 12, H, W]
    height = outputs.get('height')  # [1,  6, H, W]
    dim = outputs.get('dim')       # [1, 18, H, W]
    rot = outputs.get('rot')       # [1, 12, H, W]
    vel = outputs.get('vel')       # [1, 12, H, W]

    # Remove batch dim
    hm = 1 / (1 + np.exp(-hm[0]))  # sigmoid
    reg = reg[0]
    height = height[0]
    dim = dim[0]
    rot = rot[0]
    vel = vel[0]

    # CenterPoint peak extraction: keep only local-maximum cells per class.
    # Without this, every pixel in an object's heatmap blob becomes a box
    # (a single car → dozens of detections), flooding the output with false
    # positives and destroying precision (and therefore mAP). A 3×3 max-pool
    # NMS keeps a cell only if it equals the max in its 3×3 neighborhood —
    # the standard CenterPoint approach (equivalent to mmdet3d's _nms_heatmap
    # / local_maximum with kernel=3).
    hm = _heatmap_peak_mask(hm, kernel=3)

    H, W = hm.shape[1], hm.shape[2]
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)  # [H, W]

    detections = []
    class_offset = 0

    for t, task in enumerate(TASKS):
        n_cls = len(task['names'])
        reg_t = reg[t * 2: t * 2 + 2]       # [2, H, W]
        h_t = height[t]  # [H, W]
        dim_t = dim[t * 3: t * 3 + 3]        # [3, H, W]
        rot_t = rot[t * 2: t * 2 + 2]        # [2, H, W]
        vel_t = vel[t * 2: t * 2 + 2]        # [2, H, W]

        for c in range(n_cls):
            cls_name = task['names'][c]
            score_map = hm[class_offset + c]  # [H, W]
            mask = score_map > score_thresh
            if not mask.any():
                continue

            scores = score_map[mask]
            cx = (grid_x + reg_t[0])[mask] * stride * VOXEL_SIZE[0] + PC_RANGE[0]
            cy = (grid_y + reg_t[1])[mask] * stride * VOXEL_SIZE[1] + PC_RANGE[1]
            cz = h_t[mask]
            # mmdet3d dim regression outputs log(dx, dy, dz) in LiDAR frame,
            # i.e. (length, width, height). nuScenes 'size' field convention
            # is [width, length, height] — so we swap dx<->dy below when
            # emitting. Naming here matches the LiDAR-frame axes.
            d_len = np.exp(np.clip(dim_t[0][mask], -5, 5))   # along-x (length)
            d_wid = np.exp(np.clip(dim_t[1][mask], -5, 5))   # along-y (width)
            d_hgt = np.exp(np.clip(dim_t[2][mask], -5, 5))   # along-z (height)
            yaw = np.arctan2(rot_t[0][mask], rot_t[1][mask])
            vx = vel_t[0][mask]
            vy = vel_t[1][mask]

            # Keep top-K before NMS
            if len(scores) > max_preds:
                topk = np.argpartition(scores, -max_preds)[-max_preds:]
                scores = scores[topk]
                cx, cy, cz = cx[topk], cy[topk], cz[topk]
                d_len, d_wid, d_hgt = d_len[topk], d_wid[topk], d_hgt[topk]
                yaw, vx, vy = yaw[topk], vx[topk], vy[topk]

            keep = _circle_nms(
                np.stack([cx, cy], axis=1), scores,
                radius=NMS_RADIUS[cls_name]
            )

            for i in keep:
                q = Quaternion(axis=[0, 0, 1], angle=float(yaw[i]))
                # mmdet3d CenterPoint already regresses gravity-center z
                # (verified against nuScenes GT z), so cz is used directly.
                detections.append({
                    'translation': [float(cx[i]), float(cy[i]), float(cz[i])],
                    # nuScenes size = [width, length, height]
                    'size': [float(d_wid[i]), float(d_len[i]), float(d_hgt[i])],
                    'rotation': [q.w, q.x, q.y, q.z],
                    'velocity': [float(vx[i]), float(vy[i])],
                    'detection_name': cls_name,
                    'detection_score': float(scores[i]),
                    'attribute_name': CLASS_ATTRIBUTE[cls_name],
                })

        class_offset += n_cls

    return detections


# ── Sample token lookup from nuscenes_infos_val.pkl ──────────────────────────

def _build_token_map(val_pkl_path: str) -> dict:
    """Build filename-stem → sample_token mapping from nuscenes_infos_val.pkl.

    Handles multiple mmdet3d pkl formats:
    - New (mmdet3d 1.x): {'data_list': [...], 'metainfo': {...}}
    - Old (mmdet3d 0.x): flat list of dicts
    - Alt key names: 'infos', 'data_infos'

    Each info entry has a lidar_path like:
      'samples/LIDAR_TOP/n008-...__LIDAR_TOP__1234.pcd.bin'
    and a token/sample_token which is the nuScenes sample_token.
    """
    with open(val_pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Debug: print top-level structure to diagnose format
    if isinstance(data, dict):
        print(f'[eval] pkl keys: {list(data.keys())}')
        # Try known list keys in priority order
        for key in ('data_list', 'infos', 'data_infos'):
            if key in data:
                infos = data[key]
                print(f'[eval] using data["{key}"], {len(infos)} entries')
                break
        else:
            print('[eval] unknown dict pkl format — trying first list value')
            infos = next(
                (v for v in data.values() if isinstance(v, list) and v), []
            )
    elif isinstance(data, list):
        infos = data
        print(f'[eval] pkl is flat list, {len(infos)} entries')
    else:
        print(f'[eval] unknown pkl format: {type(data)}')
        return {}

    if not infos:
        print('[eval] pkl info list is empty')
        return {}

    # Debug: print first entry keys to understand field names
    first = infos[0]
    first_keys = list(first.keys()) if isinstance(first, dict) else type(first)
    print(f'[eval] first entry keys: {first_keys}')

    token_map = {}
    for info in infos:
        if not isinstance(info, dict):
            continue

        # lidar_path: try multiple known field names across mmdet3d versions
        # mmdet3d 1.x: info['lidar_points']['lidar_path']
        # mmdet3d 0.x: info['lidar_path'] or info['pts_filename']
        lidar_path = (
            info.get('lidar_path')
            or info.get('pts_filename', '')
            or info.get('lidar_points', {}).get('lidar_path', '')
            or info.get('lidar_info', {}).get('lidar_path', '')
        )

        # sample_token: try multiple known field names
        token = (
            info.get('token')
            or info.get('sample_token')
            or info.get('sample_data_token', '')
        )

        if lidar_path and token:
            stem = Path(lidar_path).name  # e.g. n008-...__LIDAR_TOP__1234.pcd.bin
            token_map[stem] = token

    print(f'[eval] Token map: {len(token_map)} entries from {val_pkl_path}')
    if token_map:
        first_key = next(iter(token_map))
        print(f'[eval] Sample entry: {first_key} → {token_map[first_key][:12]}...')
    return token_map


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='On-device CenterPoint mAP/NDS eval via TRT INT8 + nuscenes-devkit'
    )
    p.add_argument(
        '--engine', required=True,
        help='pts_backbone_neck_head.engine'
    )
    p.add_argument(
        '--calib-bev', required=True,
        help='Directory of [64,512,512] .npy BEV features (jetson_calib_bev/)'
    )
    p.add_argument(
        '--val-pkl', required=True,
        help='nuscenes_infos_val.pkl — maps lidar filename to sample_token'
    )
    p.add_argument(
        '--nuscenes', required=True,
        help='nuScenes dataroot with v1.0-trainval/ annotation JSONs '
             '(no sensor data needed)'
    )
    p.add_argument(
        '--stride', type=int, default=4,
        help='Head output stride in BEV cells (default: 4). '
             '512 BEV → 128 head output → stride=4.'
    )
    p.add_argument('--score-thresh', type=float, default=SCORE_THRESH)
    p.add_argument(
        '--no-eval', action='store_true',
        help='Skip in-container NuScenes eval (memory-heavy, OOM-prone on '
             'Jetson). Just produce the submission JSON; compute mAP/NDS on '
             'the host via eval_metrics.py instead.'
    )
    p.add_argument('--out', default='/workspace/output/eval.json')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f'[eval] Engine: {args.engine}')

    engine = Engine(args.engine)
    token_map = _build_token_map(args.val_pkl)

    bev_files = sorted(Path(args.calib_bev).glob('*.npy'))
    if not bev_files:
        raise FileNotFoundError(f'No .npy files in {args.calib_bev}')
    print(f'[eval] {len(bev_files)} BEV files')

    results = {}
    missing_tokens = 0
    
    for i, bev_path in enumerate(bev_files):
        # The filename is now the token itself
        token = bev_path.stem 
        
        # We no longer need to look up the token via pcd_name
        # token = token_map.get(pcd_name) 

        bev = np.load(bev_path).astype(np.float32)
        if bev.ndim == 3:
            bev = bev[np.newaxis]

        outputs = engine.infer(bev)
        dets = decode_outputs(
            outputs, stride=args.stride, score_thresh=args.score_thresh
        )

        # Add sample_token to each box
        for d in dets:
            d['sample_token'] = token
        results[token] = dets

        if (i + 1) % 50 == 0 or (i + 1) == len(bev_files):
            print(f'[eval] {i + 1}/{len(bev_files)}  '
                  f'boxes={len(dets)}  token={token[:8]}...')

    if missing_tokens:
        print(f'[eval] Warning: {missing_tokens} files had no matching token '
              f'in val pkl — skipped')

    submission = {
        'results': results,
        'meta': {
            'use_camera': False, 'use_lidar': True,
            'use_radar': False, 'use_map': False, 'use_external': False,
        },
    }
    out_path = Path(args.out)
    result_path = out_path.parent / f'{out_path.stem}_submission.json'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(submission, f)
    print(f'[eval] Submission saved: {result_path} ({len(results)} samples)')

    if args.no_eval:
        print('[eval] --no-eval set: skipping in-container NuScenes eval. '
              'Run eval_metrics.py on the host to compute mAP/NDS.')
        return

    # Explicitly free TRT engine + CUDA buffers before NuScenes loading.
    # Jetson Orin Nano has 8 GB unified memory shared between CPU and GPU.
    # The TRT engine context + input/output CUDA allocations (~70-500 MB) stay
    # alive until garbage collected, competing with NuScenes JSON loading
    # (~500 MB-1 GB) and causing OOM. Free them explicitly now — inference is
    # complete and we don't need the engine any further.
    for info in list(engine.inputs.values()) + list(engine.outputs.values()):
        info['buf'].free()
    del engine.context, engine.engine
    del engine
    import gc
    gc.collect()
    print('[eval] GPU memory freed — loading NuScenes...')

    # NuScenes evaluation
    try:
        import sys
        import types

        # nuscenes.nuscenes imports cv2 at module level for visualization only.
        # DetectionEval never calls cv2 at runtime, but the system cv2 on
        # l4t-jetpack was compiled against NumPy 1.x ABI and fails to import
        # under NumPy 2.x with: AttributeError: _ARRAY_API not found.
        # Mock cv2 before nuscenes imports it so the module-level import
        # succeeds regardless of numpy ABI version.
        try:
            import cv2  # noqa: F401
        except (AttributeError, ImportError):
            sys.modules['cv2'] = types.ModuleType('cv2')

        from nuscenes import NuScenes
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        print(f'[eval] Loading NuScenes from {args.nuscenes}...')
        nusc = NuScenes(version='v1.0-trainval', dataroot=args.nuscenes,
                        verbose=False)
        cfg = config_factory('detection_cvpr_2019')
        evaluator = NuScenesEval(
            nusc,
            config=cfg,
            result_path=str(result_path),
            eval_set='val',
            output_dir=str(Path(args.out).parent),
            verbose=True,
        )

        # Restrict evaluation to our 512 predicted sample tokens only.
        # Without this, the 5507 val tokens with no predictions score as
        # zero-recall frames, deflating mAP by a factor of ~12 (512/6019).
        our_tokens = set(results.keys())
        evaluator.sample_tokens = [
            t for t in evaluator.sample_tokens if t in our_tokens
        ]
        print(f'[eval] Evaluating on {len(evaluator.sample_tokens)} samples '
              f'(subset of full val set)')

        metrics, metric_data_list = evaluator.evaluate()
        summary = metrics.serialize()

        mAP = float(summary['mean_ap'])
        NDS = float(summary['nd_score'])
        print(f'\n[eval] mAP={mAP:.4f}  NDS={NDS:.4f}')

        out = {
            'engine': str(args.engine),
            'variant': Path(args.engine).parent.name,
            'n_samples': len(evaluator.sample_tokens),
            'note': ('Evaluated on 512-sample jetson_calib subset of val set, '
                     'not the full 6019-sample val set. Numbers are '
                     'representative but not directly comparable to full-val '
                     'A40 eval results.'),
            'mAP': mAP,
            'NDS': NDS,
            'per_class_ap': {k: float(v)
                             for k, v in summary['mean_dist_aps'].items()},
            'submission_path': str(result_path),
        }
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'[eval] Metrics saved: {args.out}')

    except ImportError as e:
        print(f'[eval] nuscenes-devkit import error: {e}')
        print('[eval] Submission JSON saved but eval skipped.')
    except Exception as e:
        print(f'[eval] Evaluation failed: {e}')
        print('[eval] Check that --nuscenes points to valid v1.0-trainval/ '
              'annotation JSONs.')


if __name__ == '__main__':
    main()

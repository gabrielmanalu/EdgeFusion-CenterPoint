"""
eval_cuda.py — on-device eval using the CUDA center-head postprocessor.

Drop-in replacement for deployment/scripts/eval.py that swaps the
host-side numpy decode for the CUDA postprocessing stage in
deployment/plugins/center_head_postprocess.cu.

Hot-path comparison
-------------------
eval.py (numpy decode):
  BEV → engine → memcpy_dtoh (all head outputs, ~1 MB) → sigmoid + peak
  NMS + box decode in numpy → circle NMS → detections

eval_cuda.py (CUDA decode):
  BEV → engine → [head outputs stay on GPU] → CUDA peak_finding kernel
  → CUDA box_decode kernel → stream.sync() → read managed memory (~20 KB,
  zero-copy on Jetson UMA) → per-class circle NMS (CPU, ~100 boxes max)

The key change: the large head output tensors (~1 MB per frame across all
channels) are never copied to CPU.  Only the small final detection list
crosses the bus, because it lives in CUDA managed memory that is directly
accessible from both CPU and GPU on Jetson's unified memory architecture.

Allocation-free hot path
------------------------
All GPU and managed-memory buffers are pre-allocated during __init__.
The per-frame loop calls:
  1. engine.infer_raw(bev)        — uploads BEV, runs TRT, returns GPU ptrs
  2. postproc.run(ptrs, stream)   — launches CUDA kernels on same stream
  3. stream.synchronize()
  4. postproc.collect_detections() — reads managed memory, circle NMS
No Python-side allocation in steps 1-4.

Prerequisites
-------------
  libcenter_head_postprocess.so must be compiled first:
    cd deployment/plugins && mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4

Usage
-----
  docker compose -f deployment/docker/docker-compose.yml run --rm eval-cuda

  Or directly (inside container):
    python3 deployment/scripts/eval_cuda.py \\
        --engine /workspace/output/engines/qat_best/pts_backbone_neck_head.engine \\
        --calib-bev /workspace/calib_bev \\
        --val-pkl /workspace/nuscenes_infos_val.pkl \\
        --so-path /workspace/plugins/libcenter_head_postprocess.so \\
        --no-eval
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

# Import the CUDA postprocessor wrapper
_PLUGINS = Path(__file__).parent.parent / 'plugins'
sys.path.insert(0, str(_PLUGINS))
from postprocess_wrapper import CenterHeadPostprocessor  # noqa: E402

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ── Engine: modified to expose raw GPU pointers without copying to CPU ────────

class Engine:
    """TRT 10.x wrapper — exposes GPU pointers so postproc runs without copy.

    Extends the Engine in eval.py with an infer_raw() method that skips the
    memcpy_dtoh of head outputs.  Only the BEV input is copied to GPU
    (unavoidable — it originates on the CPU as a .npy file).
    """

    def __init__(self, path: str) -> None:
        runtime = trt.Runtime(TRT_LOGGER)
        with open(path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()
        self.inputs  = {}
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            buf   = cuda.mem_alloc(int(np.prod(shape)) * np.dtype(dtype).itemsize)
            self.context.set_tensor_address(name, int(buf))
            entry = {'buf': buf, 'shape': shape, 'dtype': dtype}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = entry
            else:
                self.outputs[name] = entry

    def infer_raw(self, bev: np.ndarray) -> dict:
        """Upload BEV, run TRT, return dict of {name: int GPU ptr}.

        The output buffers stay on GPU — no memcpy_dtoh.  The returned
        integer pointers are valid for the duration of the stream (until
        the next call to infer_raw or explicit free).
        """
        name = next(iter(self.inputs))
        arr  = np.ascontiguousarray(bev.astype(np.float32))
        cuda.memcpy_htod_async(self.inputs[name]['buf'], arr, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        # Return GPU pointer addresses — NOT the values (no D→H copy)
        return {n: int(info['buf']) for n, info in self.outputs.items()}

    def free(self) -> None:
        """Explicit cleanup — call before loading large CPU data (OOM guard)."""
        for info in list(self.inputs.values()) + list(self.outputs.values()):
            info['buf'].free()
        del self.context, self.engine


# ── Token map (unchanged from eval.py) ───────────────────────────────────────

def _build_token_map(val_pkl_path: str) -> dict:
    with open(val_pkl_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        for key in ('data_list', 'infos', 'data_infos'):
            if key in data:
                infos = data[key]
                break
        else:
            infos = next((v for v in data.values()
                          if isinstance(v, list) and v), [])
    elif isinstance(data, list):
        infos = data
    else:
        return {}
    token_map = {}
    for info in infos:
        if not isinstance(info, dict):
            continue
        lidar_path = (
            info.get('lidar_path')
            or info.get('pts_filename', '')
            or info.get('lidar_points', {}).get('lidar_path', '')
        )
        token = (
            info.get('token')
            or info.get('sample_token')
            or info.get('sample_data_token', '')
        )
        if lidar_path and token:
            token_map[Path(lidar_path).name] = token
    return token_map


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='On-device eval using CUDA center-head postprocessor'
    )
    p.add_argument('--engine', required=True,
                   help='pts_backbone_neck_head.engine')
    p.add_argument('--calib-bev', required=True,
                   help='Directory of [64,512,512] .npy BEV features')
    p.add_argument('--val-pkl', required=True,
                   help='nuscenes_infos_val.pkl')
    p.add_argument('--nuscenes', default=None,
                   help='nuScenes dataroot (needed unless --no-eval)')
    p.add_argument('--so-path',
                   default='/workspace/plugins/libcenter_head_postprocess.so',
                   help='Path to libcenter_head_postprocess.so')
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--score-thresh', type=float, default=0.1)
    p.add_argument('--no-eval', action='store_true',
                   help='Skip in-container NuScenes eval; produce submission only')
    p.add_argument('--out', default='/workspace/output/eval_cuda.json')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f'[eval_cuda] Engine: {args.engine}')
    print(f'[eval_cuda] CUDA postproc: {args.so_path}')

    engine   = Engine(args.engine)
    postproc = CenterHeadPostprocessor(
        so_path=args.so_path,
        score_threshold=args.score_thresh,
    )

    bev_files = sorted(Path(args.calib_bev).glob('*.npy'))
    if not bev_files:
        raise FileNotFoundError(f'No .npy files in {args.calib_bev}')
    print(f'[eval_cuda] {len(bev_files)} BEV files')

    results = {}

    for i, bev_path in enumerate(bev_files):
        token = bev_path.stem

        bev = np.load(bev_path).astype(np.float32)
        if bev.ndim == 3:
            bev = bev[np.newaxis]

        # ── Hot path (allocation-free) ────────────────────────────────────
        gpu_ptrs = engine.infer_raw(bev)   # BEV → GPU, TRT executes
        postproc.run(gpu_ptrs, engine.stream)  # CUDA kernels on same stream
        engine.stream.synchronize()            # sync before reading managed mem
        dets = postproc.collect_detections()   # zero-copy managed mem + NMS
        # ─────────────────────────────────────────────────────────────────

        for d in dets:
            d['sample_token'] = token
        results[token] = dets

        if (i + 1) % 50 == 0 or (i + 1) == len(bev_files):
            print(f'[eval_cuda] {i + 1}/{len(bev_files)}  '
                  f'boxes={len(dets)}  token={token[:8]}...')

    submission = {
        'results': results,
        'meta': {
            'use_camera': False, 'use_lidar': True,
            'use_radar': False, 'use_map': False, 'use_external': False,
        },
    }
    out_path    = Path(args.out)
    # Submission named eval_cuda_{variant}_submission.json to stay distinct from
    # the numpy-decode submission (eval_{variant}_submission.json from eval.py).
    # Use: cp eval_cuda_*_submission.json eval_*_cuda_submission.json
    # then eval_metrics.py --variants *_cuda to evaluate without overwriting numpy.
    result_path = out_path.parent / f'{out_path.stem}_submission.json'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(submission, f)
    print(f'[eval_cuda] Submission saved: {result_path} ({len(results)} samples)')

    if args.no_eval:
        print('[eval_cuda] --no-eval set. Run eval_metrics.py on host for mAP/NDS.')
        return

    # Free engine before loading NuScenes (OOM guard on Jetson unified memory)
    engine.free()
    del engine
    import gc
    gc.collect()

    if not args.nuscenes:
        print('[eval_cuda] --nuscenes not provided; skipping in-container eval.')
        return

    try:
        import types
        try:
            import cv2  # noqa: F401
        except (AttributeError, ImportError):
            sys.modules['cv2'] = types.ModuleType('cv2')

        from nuscenes import NuScenes
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        nusc = NuScenes(version='v1.0-trainval', dataroot=args.nuscenes,
                        verbose=False)
        cfg  = config_factory('detection_cvpr_2019')
        evaluator = NuScenesEval(
            nusc, config=cfg, result_path=str(result_path),
            eval_set='val',
            output_dir=str(out_path.parent), verbose=True,
        )
        our_tokens = set(results.keys())
        evaluator.sample_tokens = [
            t for t in evaluator.sample_tokens if t in our_tokens
        ]
        metrics, _ = evaluator.evaluate()
        summary = metrics.serialize()
        mAP = float(summary['mean_ap'])
        NDS  = float(summary['nd_score'])
        print(f'\n[eval_cuda] mAP={mAP:.4f}  NDS={NDS:.4f}')

        out = {
            'engine':    str(args.engine),
            'variant':   Path(args.engine).parent.name,
            'postproc':  'CUDA (center_head_postprocess.cu)',
            'n_samples': len(evaluator.sample_tokens),
            'mAP': mAP, 'NDS': NDS,
            'per_class_ap': {k: float(v)
                             for k, v in summary['mean_dist_aps'].items()},
            'submission_path': str(result_path),
        }
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'[eval_cuda] Metrics saved: {args.out}')

    except Exception as e:
        print(f'[eval_cuda] Eval failed: {e}')


if __name__ == '__main__':
    main()

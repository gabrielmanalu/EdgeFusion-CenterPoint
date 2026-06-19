"""
postprocess_wrapper.py

Python interface to the CUDA center-head postprocessing kernels in
center_head_postprocess.cu / libcenter_head_postprocess.so.

Design
------
The hot path is allocation-free:

  1. TRT execute_async_v3() writes head outputs into pre-allocated GPU
     buffers (managed memory shared between CPU and GPU on Jetson UMA).
  2. ef_launch_peak_finding() runs on the TRT output buffers — no copy.
  3. ef_launch_box_decode() runs on the same buffers — no copy.
  4. stream.synchronize() — after this, managed memory is readable from CPU.
  5. CPU reads counts[0] (a single int) to know how many detections there
     are, then slices the small decoded-boxes array (≤ 20 KB at max_det=500)
     for per-class circle NMS.

On Jetson Orin (unified memory architecture) cudaMallocManaged allocations
are directly readable from CPU after the stream is synced, with no explicit
cudaMemcpy.  The TRT output buffers ARE passed as device pointers directly
to the CUDA kernels, so there is no copy of the large head tensors
(~1 MB across all channels) to CPU.  Only the small final detection list
crosses to CPU.

Usage
-----
See deployment/scripts/eval_cuda.py for a full end-to-end example.

Quick example::

    postproc = CenterHeadPostprocessor(so_path='deployment/plugins/libcenter_head_postprocess.so')

    # engine_ptrs: dict of {name: int GPU pointer} from the TRT engine
    postproc.run(engine_ptrs, stream)
    detections = postproc.collect_detections()

Dependencies
------------
  pycuda, numpy (already in the deployment container)
  libcenter_head_postprocess.so (built from center_head_postprocess.cu)
"""

import ctypes
from pathlib import Path
from typing import Dict, List

import numpy as np
import pycuda.driver as cuda  # needed for managed_zeros, Stream type hint
from pyquaternion import Quaternion

# ── CenterPoint nuScenes constants (must match eval.py) ──────────────────────

PC_RANGE   = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
VOXEL_SIZE = [0.2, 0.2, 8.0]
HEAD_STRIDE = 4
BEV_H = BEV_W = 128         # 512 / HEAD_STRIDE
NUM_CLASSES   = 10
MAX_DETECTIONS = 3000        # per frame; generous upper bound

SCORE_THRESHOLD = 0.1
POOL_RADIUS     = 1          # 3×3 max-pool NMS

# Class names in heatmap channel order (eval.py ALL_CLASSES)
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle',
    'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle',
    'pedestrian', 'traffic_cone',
]

# Circle NMS radius per class in metres (from eval.py NMS_RADIUS)
NMS_RADIUS = {
    'car': 4.0, 'truck': 4.0, 'bus': 10.0, 'trailer': 10.0,
    'construction_vehicle': 12.0, 'pedestrian': 0.175,
    'motorcycle': 0.5, 'bicycle': 0.5,
    'traffic_cone': 0.175, 'barrier': 1.5,
}

# Default nuScenes attribute per class (required in submission JSON)
CLASS_ATTRIBUTE = {
    'car': 'vehicle.moving', 'truck': 'vehicle.moving',
    'bus': 'vehicle.moving', 'trailer': 'vehicle.parked',
    'construction_vehicle': 'vehicle.parked',
    'pedestrian': 'pedestrian.moving',
    'motorcycle': 'cycle.with_rider', 'bicycle': 'cycle.with_rider',
    'traffic_cone': '', 'barrier': '',
}

# Per-class top-K before NMS (matches eval.py MAX_PREDS_PER_TASK)
MAX_PREDS_PER_CLASS = 500


# ── ctypes loader ─────────────────────────────────────────────────────────────

def _load_lib(so_path: str) -> ctypes.CDLL:
    so_path = Path(so_path)
    if not so_path.exists():
        raise FileNotFoundError(
            f'CUDA postproc library not found: {so_path}\n'
            f'Build it first:\n'
            f'  cd deployment/plugins && mkdir -p build && cd build\n'
            f'  cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4'
        )
    lib = ctypes.CDLL(str(so_path))

    # ── ef_launch_peak_finding ─────────────────────────────────────────────
    lib.ef_launch_peak_finding.restype = None
    lib.ef_launch_peak_finding.argtypes = [
        ctypes.c_void_p,   # heatmap  (device ptr, const float*)
        ctypes.c_void_p,   # peaks_out
        ctypes.c_void_p,   # scores_out
        ctypes.c_void_p,   # counts
        ctypes.c_int,      # B
        ctypes.c_int,      # C
        ctypes.c_int,      # H
        ctypes.c_int,      # W
        ctypes.c_int,      # max_det
        ctypes.c_float,    # score_thr
        ctypes.c_int,      # pool_radius
        ctypes.c_void_p,   # cudaStream_t (opaque handle)
    ]

    # ── ef_launch_box_decode ───────────────────────────────────────────────
    lib.ef_launch_box_decode.restype = None
    lib.ef_launch_box_decode.argtypes = [
        ctypes.c_void_p,   # reg
        ctypes.c_void_p,   # height
        ctypes.c_void_p,   # dim
        ctypes.c_void_p,   # rot
        ctypes.c_void_p,   # vel
        ctypes.c_void_p,   # peaks
        ctypes.c_void_p,   # counts
        ctypes.c_void_p,   # boxes_out
        ctypes.c_int,      # B
        ctypes.c_int,      # H
        ctypes.c_int,      # W
        ctypes.c_int,      # max_det
        ctypes.c_float,    # voxel_x
        ctypes.c_float,    # voxel_y
        ctypes.c_float,    # x_min
        ctypes.c_float,    # y_min
        ctypes.c_float,    # head_stride
        ctypes.c_void_p,   # cudaStream_t
    ]
    return lib


# ── Circle NMS (CPU, small array) ─────────────────────────────────────────────

def _circle_nms(xy: np.ndarray, scores: np.ndarray, radius: float) -> np.ndarray:
    """Greedy circle NMS.  xy: [N,2], scores: [N].  Returns kept indices.

    Identical to _circle_nms() in eval.py — used as the reference in
    parity tests and as the production NMS after CUDA decode.
    """
    order = scores.argsort()[::-1]
    keep = []
    suppressed = np.zeros(len(order), dtype=bool)
    for i, idx in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(idx)
        dists = np.sqrt(((xy[order[i + 1:]] - xy[idx]) ** 2).sum(1))
        suppressed[i + 1:][dists < radius] = True
    return np.array(keep, dtype=np.int64)


# ── Main class ────────────────────────────────────────────────────────────────

class CenterHeadPostprocessor:
    """Allocation-free CUDA postprocessor for CenterPoint head outputs.

    Call once at engine-init time, then call run() + collect_detections()
    on every frame.  All large buffers are pre-allocated; the hot path
    does no Python-side allocation.

    Parameters
    ----------
    so_path : str
        Path to libcenter_head_postprocess.so
    bev_h, bev_w : int
        Head output spatial dimensions (BEV grid / stride, default 128×128)
    num_classes : int
        Number of heatmap channels (default 10 for nuScenes)
    max_detections : int
        Upper bound on peaks per frame across all classes.  3000 is safe
        (a healthy CenterPoint head yields ~50-300 peaks at thresh=0.1).
    score_threshold : float
    pool_radius : int  (1 → 3×3 window)
    """

    def __init__(
        self,
        so_path: str = 'deployment/plugins/libcenter_head_postprocess.so',
        bev_h: int = BEV_H,
        bev_w: int = BEV_W,
        num_classes: int = NUM_CLASSES,
        max_detections: int = MAX_DETECTIONS,
        score_threshold: float = SCORE_THRESHOLD,
        pool_radius: int = POOL_RADIUS,
    ) -> None:
        self._lib = _load_lib(so_path)
        self.bev_h           = bev_h
        self.bev_w           = bev_w
        self.num_classes     = num_classes
        self.max_det         = max_detections
        self.score_threshold = score_threshold
        self.pool_radius     = pool_radius

        # Pre-allocate GPU output buffers (cuda.mem_alloc — device memory).
        # cuda.managed_zeros was tried first but segfaults when a TRT engine
        # occupies the CUDA context; plain device allocation is more robust.
        # All buffers are pre-allocated once here; the hot path does no
        # allocation per frame.
        self._peaks_gpu  = cuda.mem_alloc(max_detections * 3 * 4)   # int32
        self._scores_gpu = cuda.mem_alloc(max_detections * 4)        # float32
        self._counts_gpu = cuda.mem_alloc(4)                         # int32 [1]
        self._boxes_gpu  = cuda.mem_alloc(max_detections * 10 * 4)  # float32

        # Pre-allocated CPU arrays for reading results after dtoh copy.
        # The copy is ~170 KB total — negligible latency.
        self._peaks_cpu  = np.zeros(max_detections * 3, dtype=np.int32)
        self._scores_cpu = np.zeros(max_detections,     dtype=np.float32)
        self._counts_cpu = np.zeros(1,                  dtype=np.int32)
        self._boxes_cpu  = np.zeros(max_detections * 10, dtype=np.float32)

    # ── Hot path ──────────────────────────────────────────────────────────────

    def run(self, gpu_ptrs: Dict[str, int], stream: cuda.Stream) -> None:
        """Launch peak-finding + box-decode on the TRT engine output buffers.

        Parameters
        ----------
        gpu_ptrs : dict
            {tensor_name: int_device_pointer} from the TRT engine.
            Expected keys: 'heatmap', 'reg', 'height', 'dim', 'rot', 'vel'
        stream : pycuda.driver.Stream
            The SAME stream used by engine.execute_async_v3() — kernels
            are queued behind the engine on the same stream, so no explicit
            synchronization is needed between them.
        """
        # Reset the atomic count to 0 via a tiny host-to-device copy.
        # Plain numpy write to _counts_cpu then htod is safe regardless of
        # CUDA context state (no managed-memory assumptions).
        self._counts_cpu[0] = 0
        cuda.memcpy_htod_async(self._counts_gpu, self._counts_cpu, stream)

        peaks_ptr  = int(self._peaks_gpu)
        scores_ptr = int(self._scores_gpu)
        counts_ptr = int(self._counts_gpu)
        boxes_ptr  = int(self._boxes_gpu)

        stream_handle = ctypes.c_void_p(stream.handle)

        # 1. Peak finding — operates on TRT output GPU buffer (no copy)
        self._lib.ef_launch_peak_finding(
            ctypes.c_void_p(gpu_ptrs['heatmap']),
            ctypes.c_void_p(peaks_ptr),
            ctypes.c_void_p(scores_ptr),
            ctypes.c_void_p(counts_ptr),
            ctypes.c_int(1),                          # B = 1
            ctypes.c_int(self.num_classes),
            ctypes.c_int(self.bev_h),
            ctypes.c_int(self.bev_w),
            ctypes.c_int(self.max_det),
            ctypes.c_float(self.score_threshold),
            ctypes.c_int(self.pool_radius),
            stream_handle,
        )

        # 2. Box decode — still on GPU, same stream
        self._lib.ef_launch_box_decode(
            ctypes.c_void_p(gpu_ptrs['reg']),
            ctypes.c_void_p(gpu_ptrs['height']),
            ctypes.c_void_p(gpu_ptrs['dim']),
            ctypes.c_void_p(gpu_ptrs['rot']),
            ctypes.c_void_p(gpu_ptrs['vel']),
            ctypes.c_void_p(peaks_ptr),
            ctypes.c_void_p(counts_ptr),
            ctypes.c_void_p(boxes_ptr),
            ctypes.c_int(1),                          # B = 1
            ctypes.c_int(self.bev_h),
            ctypes.c_int(self.bev_w),
            ctypes.c_int(self.max_det),
            ctypes.c_float(VOXEL_SIZE[0]),
            ctypes.c_float(VOXEL_SIZE[1]),
            ctypes.c_float(PC_RANGE[0]),
            ctypes.c_float(PC_RANGE[1]),
            ctypes.c_float(HEAD_STRIDE),
            stream_handle,
        )
        # caller must call stream.synchronize() before collect_detections()

    def collect_detections(self) -> List[dict]:
        """Copy GPU results to CPU and apply per-class circle NMS.

        Must be called AFTER stream.synchronize().  Copies ~170 KB of result
        data from GPU to pre-allocated CPU arrays (negligible latency), then
        applies per-class top-K + circle NMS on CPU.
        """
        cuda.memcpy_dtoh(self._counts_cpu, self._counts_gpu)
        n = int(min(self._counts_cpu[0], self.max_det))
        if n == 0:
            return []

        cuda.memcpy_dtoh(self._peaks_cpu[:n * 3], self._peaks_gpu)
        cuda.memcpy_dtoh(self._scores_cpu[:n],    self._scores_gpu)
        cuda.memcpy_dtoh(self._boxes_cpu[:n * 10], self._boxes_gpu)

        peaks  = self._peaks_cpu[:n * 3].reshape(n, 3)
        scores = self._scores_cpu[:n]
        boxes  = self._boxes_cpu[:n * 10].reshape(n, 10)

        # Per-class top-K + circle NMS (CPU, fast: ~100 boxes max per class)
        detections = []
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            mask = peaks[:, 2] == cls_id
            if not mask.any():
                continue

            cls_scores = scores[mask]
            cls_boxes  = boxes[mask]

            # Top-K per class before NMS (matches eval.py MAX_PREDS_PER_TASK)
            if len(cls_scores) > MAX_PREDS_PER_CLASS:
                topk = np.argpartition(cls_scores, -MAX_PREDS_PER_CLASS)[
                    -MAX_PREDS_PER_CLASS:
                ]
                cls_scores = cls_scores[topk]
                cls_boxes  = cls_boxes[topk]

            # Circle NMS
            xy   = cls_boxes[:, :2]          # cx, cy
            keep = _circle_nms(xy, cls_scores, NMS_RADIUS[cls_name])

            for i in keep:
                cx, cy, cz        = cls_boxes[i, 0], cls_boxes[i, 1], cls_boxes[i, 2]
                d_len, d_wid, d_hgt = cls_boxes[i, 3], cls_boxes[i, 4], cls_boxes[i, 5]
                yaw               = cls_boxes[i, 6]
                vx, vy            = cls_boxes[i, 7], cls_boxes[i, 8]

                q = Quaternion(axis=[0, 0, 1], angle=float(yaw))
                detections.append({
                    'translation': [float(cx), float(cy), float(cz)],
                    # nuScenes size = [width, length, height]
                    'size': [float(d_wid), float(d_len), float(d_hgt)],
                    'rotation': [q.w, q.x, q.y, q.z],
                    'velocity': [float(vx), float(vy)],
                    'detection_name': cls_name,
                    'detection_score': float(cls_scores[i]),
                    'attribute_name': CLASS_ATTRIBUTE[cls_name],
                })

        return detections

    # ── Raw access for parity tests ───────────────────────────────────────────

    def raw_peaks(self, n: int) -> np.ndarray:
        """Return the first n peak records as (row, col, class_id) array.
        Call AFTER stream.synchronize()."""
        cuda.memcpy_dtoh(self._peaks_cpu[:n * 3], self._peaks_gpu)
        return self._peaks_cpu[:n * 3].reshape(n, 3).copy()

    def raw_scores(self, n: int) -> np.ndarray:
        cuda.memcpy_dtoh(self._scores_cpu[:n], self._scores_gpu)
        return self._scores_cpu[:n].copy()

    def raw_boxes(self, n: int) -> np.ndarray:
        cuda.memcpy_dtoh(self._boxes_cpu[:n * 10], self._boxes_gpu)
        return self._boxes_cpu[:n * 10].reshape(n, 10).copy()

    @property
    def n_detected(self) -> int:
        cuda.memcpy_dtoh(self._counts_cpu, self._counts_gpu)
        return int(min(self._counts_cpu[0], self.max_det))

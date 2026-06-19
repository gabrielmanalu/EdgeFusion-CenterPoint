"""
Parity tests: CUDA center-head postprocessing vs numpy reference.

Compares the CUDA kernels in center_head_postprocess.cu against the
validated numpy implementation in deployment/scripts/eval.py (the same
code that produced the 0.4265 mAP eval result).

Test strategy
-------------
We import the reference functions from eval.py directly — not reimplementing
them here — so that parity tests verify the CUDA kernel against the ground
truth, not against a potentially-divergent second numpy copy.

Three categories:
  1. Unit — synthetic heatmaps / regression maps with known analytic answers.
     These run in CI without CUDA (CUDA-dependent tests auto-skip).
  2. Parity — random realistic tensors; CUDA output must match numpy output
     within TOLERANCE = 1e-3 absolute (fp32 rounding budget).
  3. End-to-end — full frame decode; distribution of box coordinates and
     counts must be plausible.

Running
-------
  # On Jetson, with libcenter_head_postprocess.so already compiled:
  cd EdgeFusion-CenterPoint
  pytest deployment/plugins/test_parity.py -v

  # In CI (no CUDA / no .so): CUDA tests auto-skip, numpy tests run.
  pytest deployment/plugins/test_parity.py -v -k "not cuda"
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Add deployment/scripts to path so we can import eval.py's reference fns
SCRIPTS = Path(__file__).parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPTS))

# ── Conditionally mock GPU modules for environments without CUDA ──────────────
# eval.py imports pycuda.autoinit and tensorrt at module level. If those
# packages are genuinely available (inside Docker), let them load normally —
# eval.py works as-is and the CUDA parity tests can run. If not available
# (host machine without pycuda), mock them so we can still import eval.py's
# pure-numpy reference functions for the numpy-only tests.
import types as _types


def _make_mock(name: str):
    m = _types.ModuleType(name)
    sys.modules[name] = m
    return m


try:
    import pycuda.autoinit   # noqa: F401  — initialises CUDA context in Docker
    import pycuda.driver     # noqa: F401
except ImportError:
    # Host machine without CUDA runtime: mock so eval.py numpy fns are importable
    _make_mock('pycuda')
    _make_mock('pycuda.autoinit')
    _make_mock('pycuda.driver')

try:
    import tensorrt          # noqa: F401
except ImportError:
    _trt = _make_mock('tensorrt')

    class _MockLogger:       # eval.py calls trt.Logger(trt.Logger.WARNING)
        WARNING = 0

        def __init__(self, *a): pass  # noqa: E704

    _trt.Logger = _MockLogger

from eval import (   # noqa: E402  (import after sys.path manipulation)
    _heatmap_peak_mask,
    _circle_nms,
    decode_outputs,
    SCORE_THRESH,
    PC_RANGE,
    VOXEL_SIZE,
)

# ── Constants matching eval.py / postprocess_wrapper.py ──────────────────────
H, W   = 128, 128      # head output spatial size
C      = 10            # heatmap channels (10 classes)
STRIDE = 4
TOLERANCE = 1e-3       # absolute tolerance for float32 comparisons

SO_PATH = Path(__file__).parent / 'libcenter_head_postprocess.so'
CUDA_AVAILABLE = False
postproc = None

def _try_load_cuda():
    """Attempt to load the CUDA wrapper; set module-level flags."""
    global CUDA_AVAILABLE, postproc
    if not SO_PATH.exists():
        print(f'[test_parity] .so not found at {SO_PATH} — CUDA tests will skip')
        return
    # Detect whether pycuda is real or our eval.py mock.
    # Real pycuda.driver always has Stream; our empty mock does not.
    import pycuda.driver as _drv
    if not hasattr(_drv, 'Stream'):
        print('[test_parity] pycuda is mocked (host machine, not in Docker) '
              '— CUDA tests will skip. Run inside the Docker container to '
              'test the CUDA kernels.')
        return
    try:
        import pycuda.autoinit  # noqa: F401
        sys.path.insert(0, str(Path(__file__).parent))
        from postprocess_wrapper import CenterHeadPostprocessor
        postproc = CenterHeadPostprocessor(so_path=str(SO_PATH))
        CUDA_AVAILABLE = True
        print(f'[test_parity] CUDA available — .so loaded from {SO_PATH}')
    except Exception as e:
        print(f'[test_parity] CUDA not available: {e}')


_try_load_cuda()


cuda_only = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason='libcenter_head_postprocess.so not found or CUDA unavailable'
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_synthetic_heatmap(peaks_spec, shape=(C, H, W), background=-10.0):
    """Return a raw-logit heatmap with planted peaks at specified locations.

    peaks_spec: list of (row, col, cls, logit_value)
    Background raw logit = background (sigmoid ≈ 0 for background=-10).
    """
    hm = np.full(shape, background, dtype=np.float32)
    for row, col, cls, val in peaks_spec:
        hm[cls, row, col] = val
    return hm


def _numpy_decode_full(hm_raw, reg, height, dim, rot, vel, stride=STRIDE):
    """Run the full numpy reference decode from eval.py on synthetic tensors.

    Wraps inputs in a dict matching what decode_outputs() expects (with
    batch dim, as eval.py strips it internally).
    """
    outputs = {
        'heatmap': hm_raw[np.newaxis],   # add batch dim
        'reg':     reg[np.newaxis],
        'height':  height[np.newaxis],
        'dim':     dim[np.newaxis],
        'rot':     rot[np.newaxis],
        'vel':     vel[np.newaxis],
    }
    return decode_outputs(outputs, stride=stride, score_thresh=SCORE_THRESH)


def _random_regression_maps(seed=42):
    """Return realistic random regression maps matching head output shapes."""
    rng = np.random.default_rng(seed)
    return {
        'reg':    rng.standard_normal((12, H, W)).astype(np.float32) * 0.5,
        'height': rng.standard_normal((6,  H, W)).astype(np.float32),
        'dim':    rng.standard_normal((18, H, W)).astype(np.float32) * 0.3,
        'rot':    rng.standard_normal((12, H, W)).astype(np.float32),
        'vel':    rng.standard_normal((12, H, W)).astype(np.float32) * 2.0,
    }


# ── Category 1: numpy reference unit tests ───────────────────────────────────
# These run in CI without CUDA and verify that our understanding of eval.py's
# decode math is correct.  If these fail, the CUDA kernel is definitely wrong.

class TestNumpyReference:

    def test_heatmap_peak_mask_single_peak(self):
        """One dominant peak → only that cell survives."""
        hm = np.zeros((C, H, W), dtype=np.float32)
        hm[0, 64, 64] = 5.0          # strong peak for class 0 (car)
        result = _heatmap_peak_mask(hm, kernel=3)
        assert result[0, 64, 64] > 0, 'Peak cell must survive'
        assert result[0, 63, 64] == 0, 'Adjacent cell must be suppressed'
        assert result[0, 64, 63] == 0, 'Adjacent cell must be suppressed'

    def test_heatmap_peak_mask_two_adjacent_suppressed(self):
        """Two adjacent equal-strength peaks: only both survive if they don't
        dominate each other's window (edge case: equal neighbours both kept)."""
        hm = np.zeros((C, H, W), dtype=np.float32)
        hm[0, 60, 60] = 3.0
        hm[0, 60, 62] = 3.0   # 2 cells apart — outside 3×3 window
        result = _heatmap_peak_mask(hm, kernel=3)
        assert result[0, 60, 60] > 0, '3×3 windows dont overlap, both should survive'
        assert result[0, 60, 62] > 0, '3×3 windows dont overlap, both should survive'

    def test_heatmap_peak_mask_adjacent_pair_suppressed(self):
        """Stronger neighbour suppresses weaker adjacent cell."""
        hm = np.zeros((C, H, W), dtype=np.float32)
        hm[0, 64, 64] = 5.0    # dominant
        hm[0, 64, 65] = 4.0    # weaker, inside window of dominant
        result = _heatmap_peak_mask(hm, kernel=3)
        assert result[0, 64, 64] > 0, 'Dominant must survive'
        assert result[0, 64, 65] == 0, 'Weaker neighbour must be suppressed'

    def test_heatmap_peak_mask_boundary(self):
        """Peak at grid boundary (row=0, col=0) must be handled correctly."""
        hm = np.zeros((C, H, W), dtype=np.float32)
        hm[0, 0, 0] = 5.0
        result = _heatmap_peak_mask(hm, kernel=3)
        assert result[0, 0, 0] > 0

    def test_circle_nms_keeps_highest_score(self):
        """Two boxes within radius: only higher score kept."""
        xy = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        scores = np.array([0.9, 0.6])
        keep = _circle_nms(xy, scores, radius=2.0)
        assert 0 in keep
        assert 1 not in keep

    def test_circle_nms_keeps_both_if_far(self):
        """Two boxes beyond radius: both kept."""
        xy = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        scores = np.array([0.9, 0.6])
        keep = _circle_nms(xy, scores, radius=2.0)
        assert len(keep) == 2

    def test_box_decode_coordinate_math(self):
        """Zero-offset regression → box centre at voxel-grid coordinate."""
        # Single class=0 (car, task=0) peak at row=r, col=c
        # reg[0, r, c] = reg[1, r, c] = 0  → offset is zero
        # cx = c * stride * voxel_x + x_min
        r, c = 64, 64
        reg = np.zeros((12, H, W), dtype=np.float32)
        height = np.zeros((6, H, W), dtype=np.float32)
        height[0, r, c] = 1.5    # z-centre
        dim = np.zeros((18, H, W), dtype=np.float32)  # exp(0)=1 → 1m box
        rot = np.zeros((12, H, W), dtype=np.float32)  # sin=0,cos=0→yaw=0
        vel = np.zeros((12, H, W), dtype=np.float32)

        # Plant a single strong peak (class=0, task=0)
        hm_raw = np.full((C, H, W), -10.0, dtype=np.float32)
        hm_raw[0, r, c] = 5.0

        dets = _numpy_decode_full(hm_raw, reg, height, dim, rot, vel)

        assert len(dets) >= 1
        det = next(d for d in dets if d['detection_name'] == 'car')
        expected_cx = c * STRIDE * VOXEL_SIZE[0] + PC_RANGE[0]
        expected_cy = r * STRIDE * VOXEL_SIZE[1] + PC_RANGE[1]
        assert abs(det['translation'][0] - expected_cx) < TOLERANCE
        assert abs(det['translation'][1] - expected_cy) < TOLERANCE
        assert abs(det['translation'][2] - 1.5) < TOLERANCE

    def test_box_decode_size_log_space(self):
        """Positive dim offset → exp gives dimensions > 1m; negative → < 1m."""
        r, c = 32, 32
        reg = np.zeros((12, H, W), dtype=np.float32)
        height = np.zeros((6, H, W), dtype=np.float32)
        dim = np.zeros((18, H, W), dtype=np.float32)
        dim[0, r, c] = math.log(4.5)   # task 0, l → exp → 4.5
        dim[1, r, c] = math.log(2.0)   # task 0, w → exp → 2.0
        dim[2, r, c] = math.log(1.6)   # task 0, h → exp → 1.6
        rot = np.zeros((12, H, W), dtype=np.float32)
        vel = np.zeros((12, H, W), dtype=np.float32)

        hm_raw = np.full((C, H, W), -10.0, dtype=np.float32)
        hm_raw[0, r, c] = 5.0

        dets = _numpy_decode_full(hm_raw, reg, height, dim, rot, vel)
        det = next(d for d in dets if d['detection_name'] == 'car')
        # nuScenes size = [width, length, height] → [2.0, 4.5, 1.6]
        assert abs(det['size'][0] - 2.0) < TOLERANCE   # width
        assert abs(det['size'][1] - 4.5) < TOLERANCE   # length
        assert abs(det['size'][2] - 1.6) < TOLERANCE   # height

    def test_box_decode_yaw(self):
        """atan2(sin=1, cos=0) = pi/2 ≈ 1.5708."""
        r, c = 50, 50
        rot = np.zeros((12, H, W), dtype=np.float32)
        rot[0, r, c] = 1.0   # task 0, sin_yaw = 1
        rot[1, r, c] = 0.0   # task 0, cos_yaw = 0

        hm_raw = np.full((C, H, W), -10.0, dtype=np.float32)
        hm_raw[0, r, c] = 5.0

        dets = _numpy_decode_full(
            hm_raw, np.zeros((12, H, W), dtype=np.float32),
            np.zeros((6, H, W), dtype=np.float32),
            np.zeros((18, H, W), dtype=np.float32),
            rot,
            np.zeros((12, H, W), dtype=np.float32),
        )
        det = next(d for d in dets if d['detection_name'] == 'car')
        q = det['rotation']          # [w, x, y, z]
        # atan2(1, 0) = pi/2 → Quaternion(axis=[0,0,1], angle=pi/2)
        # → q.w = cos(pi/4) ≈ 0.7071
        assert abs(q[0] - math.cos(math.pi / 4)) < TOLERANCE


# ── Category 2: CUDA parity tests ────────────────────────────────────────────

class TestCudaParity:

    @cuda_only
    def test_peak_locations_single_peak(self):
        """CUDA finds the same single peak as numpy on a synthetic heatmap."""
        import pycuda.driver as cuda

        hm_raw = _make_synthetic_heatmap(
            [(64, 64, 0, 5.0)], shape=(C, H, W)
        )

        # Numpy reference
        hm_sig = 1.0 / (1.0 + np.exp(-hm_raw))
        hm_peak_ref = _heatmap_peak_mask(hm_sig, kernel=3)
        assert hm_peak_ref[0, 64, 64] > SCORE_THRESH, 'Numpy ref must detect the planted peak'

        # CUDA — save all DeviceAllocation objects so Python GC does not free
        # them before the kernel finishes.  int(cuda.mem_alloc(...)) inline
        # extracts the pointer but immediately frees the allocation → crash.
        stream = cuda.Stream()
        hm_gpu   = cuda.mem_alloc(hm_raw.nbytes)
        reg_gpu  = cuda.mem_alloc(12 * H * W * 4)
        ht_gpu   = cuda.mem_alloc(6  * H * W * 4)
        dim_gpu  = cuda.mem_alloc(18 * H * W * 4)
        rot_gpu  = cuda.mem_alloc(12 * H * W * 4)
        vel_gpu  = cuda.mem_alloc(12 * H * W * 4)

        cuda.memcpy_htod_async(hm_gpu, hm_raw, stream)

        postproc.run({
            'heatmap': int(hm_gpu),
            'reg':     int(reg_gpu),
            'height':  int(ht_gpu),
            'dim':     int(dim_gpu),
            'rot':     int(rot_gpu),
            'vel':     int(vel_gpu),
        }, stream)
        stream.synchronize()

        n = postproc.n_detected
        assert n >= 1, f'Expected at least 1 peak, got {n}'
        peaks_cuda = postproc.raw_peaks(n)   # [n, 3]: (row, col, cls)

        # Check that (row=64, col=64, cls=0) is in CUDA output
        found = any(
            (p[0] == 64 and p[1] == 64 and p[2] == 0) for p in peaks_cuda
        )
        assert found, f'Expected peak at (64,64,cls=0), CUDA found: {peaks_cuda}'

    @cuda_only
    def test_peak_count_matches_numpy(self):
        """CUDA and numpy agree on the number of peaks for a sparse heatmap.

        Uses a very-negative background (sigmoid ≈ 0) so only the 15 planted
        peaks survive the score threshold.  The numpy count is therefore well
        below MAX_DETECTIONS=3000, making the equality assertion meaningful.
        (A dense random heatmap can produce >3000 peaks; when numpy_count >
        MAX_DETECTIONS the CUDA kernel correctly caps and the counts diverge
        by design, not by a kernel bug.)
        """
        import pycuda.driver as cuda

        rng = np.random.default_rng(0)
        # Background well below threshold: sigmoid(-10) ≈ 4.5e-5 << 0.1
        hm_raw = np.full((C, H, W), -10.0, dtype=np.float32)
        # Plant 15 strong, spatially separated peaks across different classes
        planted = set()
        for i in range(15):
            while True:
                r = int(rng.integers(5, H - 5))
                c = int(rng.integers(5, W - 5))
                cls = int(rng.integers(0, C))
                # Ensure peaks are at least 3 cells apart so none suppress each other
                if all(abs(r - pr) > 2 or abs(c - pc) > 2 for pr, pc, _ in planted):
                    break
            hm_raw[cls, r, c] = 4.0
            planted.add((r, c, cls))

        # Numpy count (expected: exactly 15 — no two peaks are within 3×3 of each other)
        hm_sig = 1.0 / (1.0 + np.exp(-hm_raw))
        hm_peak = _heatmap_peak_mask(hm_sig, kernel=3)
        numpy_count = int((hm_peak > SCORE_THRESH).sum())
        assert numpy_count == 15, f'Test setup error: expected 15 numpy peaks, got {numpy_count}'

        # CUDA count
        stream  = cuda.Stream()
        hm_gpu  = cuda.mem_alloc(hm_raw.nbytes)
        reg_gpu = cuda.mem_alloc(12 * H * W * 4)
        ht_gpu  = cuda.mem_alloc(6  * H * W * 4)
        dim_gpu = cuda.mem_alloc(18 * H * W * 4)
        rot_gpu = cuda.mem_alloc(12 * H * W * 4)
        vel_gpu = cuda.mem_alloc(12 * H * W * 4)

        cuda.memcpy_htod_async(hm_gpu, hm_raw, stream)

        postproc.run({
            'heatmap': int(hm_gpu),
            'reg':     int(reg_gpu),
            'height':  int(ht_gpu),
            'dim':     int(dim_gpu),
            'rot':     int(rot_gpu),
            'vel':     int(vel_gpu),
        }, stream)
        stream.synchronize()

        cuda_count = postproc.n_detected
        assert cuda_count == numpy_count, (
            f'Peak count mismatch: numpy={numpy_count}, CUDA={cuda_count}'
        )

    @cuda_only
    def test_box_decode_coordinates_match_numpy(self):
        """CUDA decoded box coordinates match numpy reference within TOLERANCE."""
        import pycuda.driver as cuda

        r, c_col = 48, 72
        maps = _random_regression_maps(seed=7)

        # Plant one isolated peak (class=5, barrier, task=3)
        hm_raw = np.full((C, H, W), -10.0, dtype=np.float32)
        hm_raw[5, r, c_col] = 5.0

        # Numpy reference decode
        ref_dets = _numpy_decode_full(
            hm_raw, maps['reg'], maps['height'],
            maps['dim'], maps['rot'], maps['vel'],
        )
        assert len(ref_dets) >= 1
        ref = next(d for d in ref_dets if d['detection_name'] == 'barrier')

        # CUDA decode
        stream = cuda.Stream()

        def _upload(arr):
            buf = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod_async(buf, arr.astype(np.float32), stream)
            return buf

        hm_gpu = _upload(hm_raw[np.newaxis])    # add batch dim [1,C,H,W]

        # TRT outputs have shape [1, channels, H, W]
        reg_gpu    = _upload(maps['reg'][np.newaxis])
        height_gpu = _upload(maps['height'][np.newaxis])
        dim_gpu    = _upload(maps['dim'][np.newaxis])
        rot_gpu    = _upload(maps['rot'][np.newaxis])
        vel_gpu    = _upload(maps['vel'][np.newaxis])

        postproc.run({
            'heatmap': int(hm_gpu),
            'reg':     int(reg_gpu),
            'height':  int(height_gpu),
            'dim':     int(dim_gpu),
            'rot':     int(rot_gpu),
            'vel':     int(vel_gpu),
        }, stream)
        stream.synchronize()

        cuda_dets = postproc.collect_detections()
        assert len(cuda_dets) >= 1
        cuda_det = next(
            (d for d in cuda_dets if d['detection_name'] == 'barrier'), None
        )
        assert cuda_det is not None, 'barrier detection missing from CUDA output'

        for axis, name in enumerate(['x', 'y', 'z']):
            diff = abs(cuda_det['translation'][axis] - ref['translation'][axis])
            assert diff < TOLERANCE, (
                f'translation[{name}] mismatch: '
                f'cuda={cuda_det["translation"][axis]:.5f}, '
                f'numpy={ref["translation"][axis]:.5f}, '
                f'diff={diff:.2e}'
            )

        for i, dim_name in enumerate(['width', 'length', 'height']):
            diff = abs(cuda_det['size'][i] - ref['size'][i])
            assert diff < TOLERANCE, (
                f'size[{dim_name}] mismatch: cuda={cuda_det["size"][i]:.5f}, '
                f'numpy={ref["size"][i]:.5f}, diff={diff:.2e}'
            )

    @cuda_only
    def test_all_classes_decode(self):
        """Plant one peak per class; all 10 must be found and decoded."""
        import pycuda.driver as cuda

        maps = _random_regression_maps(seed=99)
        hm_raw = np.full((C, H, W), -10.0, dtype=np.float32)
        planted = []
        for cls_id in range(C):
            row = 10 + cls_id * 10
            col = 10 + cls_id * 10
            hm_raw[cls_id, row, col] = 5.0
            planted.append((row, col, cls_id))

        stream = cuda.Stream()
        keep_alive = []  # hold DeviceAllocation objects until after sync

        def _upload(arr):
            buf = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod_async(buf, arr.astype(np.float32), stream)
            keep_alive.append(buf)
            return int(buf)

        postproc.run({
            'heatmap': _upload(hm_raw[np.newaxis]),
            'reg':     _upload(maps['reg'][np.newaxis]),
            'height':  _upload(maps['height'][np.newaxis]),
            'dim':     _upload(maps['dim'][np.newaxis]),
            'rot':     _upload(maps['rot'][np.newaxis]),
            'vel':     _upload(maps['vel'][np.newaxis]),
        }, stream)
        stream.synchronize()

        n = postproc.n_detected
        assert n >= C, f'Expected >= {C} peaks (one per class), got {n}'
        found_classes = set(postproc.raw_peaks(n)[:, 2])
        assert found_classes == set(range(C)), (
            f'Not all 10 classes found. Missing: {set(range(C)) - found_classes}'
        )


# ── Category 3: End-to-end plausibility ──────────────────────────────────────

class TestEndToEndPlausibility:

    @cuda_only
    def test_realistic_frame_box_count(self):
        """Realistic random heatmap → plausible number of detections (<500)."""
        import pycuda.driver as cuda

        rng = np.random.default_rng(123)
        hm_raw = (rng.standard_normal((C, H, W)) - 3.0).astype(np.float32)
        maps = _random_regression_maps(seed=123)

        stream = cuda.Stream()
        keep_alive = []

        def _upload(arr):
            buf = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod_async(buf, arr.astype(np.float32), stream)
            keep_alive.append(buf)
            return int(buf)

        postproc.run({
            'heatmap': _upload(hm_raw[np.newaxis]),
            'reg':     _upload(maps['reg'][np.newaxis]),
            'height':  _upload(maps['height'][np.newaxis]),
            'dim':     _upload(maps['dim'][np.newaxis]),
            'rot':     _upload(maps['rot'][np.newaxis]),
            'vel':     _upload(maps['vel'][np.newaxis]),
        }, stream)
        stream.synchronize()

        dets = postproc.collect_detections()
        assert len(dets) < 500, f'Implausibly many detections: {len(dets)}'

    @cuda_only
    def test_translations_within_point_cloud_range(self):
        """All decoded box centres must lie within the configured PC range."""
        import pycuda.driver as cuda

        rng = np.random.default_rng(42)
        hm_raw = (rng.standard_normal((C, H, W)) - 2.0).astype(np.float32)
        maps = _random_regression_maps(seed=42)

        stream = cuda.Stream()
        keep_alive = []

        def _upload(arr):
            buf = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod_async(buf, arr.astype(np.float32), stream)
            keep_alive.append(buf)
            return int(buf)

        postproc.run({
            'heatmap': _upload(hm_raw[np.newaxis]),
            'reg':     _upload(maps['reg'][np.newaxis]),
            'height':  _upload(maps['height'][np.newaxis]),
            'dim':     _upload(maps['dim'][np.newaxis]),
            'rot':     _upload(maps['rot'][np.newaxis]),
            'vel':     _upload(maps['vel'][np.newaxis]),
        }, stream)
        stream.synchronize()

        dets = postproc.collect_detections()
        for d in dets:
            tx, ty, tz = d['translation']
            assert PC_RANGE[0] - 1 <= tx <= PC_RANGE[3] + 1, (
                f'tx={tx} out of PC range'
            )
            assert PC_RANGE[1] - 1 <= ty <= PC_RANGE[4] + 1, (
                f'ty={ty} out of PC range'
            )

"""
benchmark_e2e.py — end-to-end pipeline latency and power benchmark.

Measures the COMPLETE hot path on Jetson:
  BEV htod  →  TRT engine  →  CUDA peak_finding  →  CUDA box_decode
  →  result dtoh  →  circle NMS

Complements benchmark.py (which times the engine only) with the full
deployment pipeline latency.

Timing approach
---------------
GPU-side portion (htod + engine + CUDA kernels):
    CUDA events — immune to OS scheduling jitter, ~microsecond accuracy.
    Recorded on the SAME stream as everything else, so ordering is exact.

CPU-side portion (result dtoh + circle NMS):
    time.perf_counter() after stream.synchronize(). Both are CPU operations
    that don't benefit from CUDA event timing.

Total latency = GPU CUDA event time + CPU tail time.

Allocation-free hot path
------------------------
All GPU buffers (engine I/O + postprocessor peaks/scores/boxes/counts) and
CPU result arrays are pre-allocated once at init.  The timed loop does:
  feed → run → run_postproc → sync → collect
with zero Python allocations per frame.

Output
------
Prints p50/p99 and jitter (p99−p50) for the GPU stage, the CPU tail,
and the total pipeline.  Saves full distributions to JSON alongside the
power numbers.

Prerequisites
-------------
  sudo jetson_clocks          # on HOST before starting container
  sudo nvpmodel -m 1          # 25W mode
  libcenter_head_postprocess.so compiled in deployment/plugins/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda

# ── Reuse Engine, TegrastatsMonitor, _load_bev_sample from benchmark.py ───────
sys.path.insert(0, str(Path(__file__).parent))
from benchmark import Engine, TegrastatsMonitor, _load_bev_sample  # noqa: E402

# ── Reuse CenterHeadPostprocessor from plugins/ ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / 'plugins'))
from postprocess_wrapper import CenterHeadPostprocessor  # noqa: E402

BEV_SHAPE = (1, 64, 512, 512)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='End-to-end pipeline benchmark (engine + CUDA postproc)'
    )
    p.add_argument('--engine', required=True,
                   help='pts_backbone_neck_head.engine')
    p.add_argument('--calib-bev', default=None,
                   help='Directory of [64,512,512] .npy BEV features')
    p.add_argument('--so-path',
                   default='/workspace/plugins/libcenter_head_postprocess.so',
                   help='Path to libcenter_head_postprocess.so')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--power-interval', type=int, default=100)
    p.add_argument('--out',
                   default='/workspace/output/benchmark_e2e.json')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    variant = Path(args.engine).parent.name

    # ── Init postprocessor FIRST (avoids TRT context ordering issue) ──────────
    print(f'[bench_e2e] Loading CUDA postprocessor from {args.so_path}')
    postproc = CenterHeadPostprocessor(so_path=args.so_path)

    # ── Load TRT engine ───────────────────────────────────────────────────────
    print(f'[bench_e2e] Loading engine: {args.engine}')
    eng = Engine(args.engine)
    stream = eng.stream
    in_name = list(eng.inputs.keys())[0]
    gpu_ptrs = {n: int(info['buf']) for n, info in eng.outputs.items()}

    # ── Pre-allocate BEV in pinned host memory (fastest htod) ─────────────────
    if args.calib_bev:
        bev_np = _load_bev_sample(args.calib_bev)
        print(f'[bench_e2e] Using BEV sample from {args.calib_bev}')
    else:
        bev_np = np.random.rand(*BEV_SHAPE).astype(np.float32)
        print('[bench_e2e] Using synthetic BEV input')

    bev_pinned = cuda.pagelocked_zeros(BEV_SHAPE, dtype=np.float32)
    bev_pinned[:] = bev_np

    # ── CUDA event objects (reused each iteration — no per-frame alloc) ───────
    ev_start   = cuda.Event()   # start of GPU portion (before htod)
    ev_postproc = cuda.Event()  # after CUDA kernels (before dtoh)
    ev_dtoh    = cuda.Event()   # after dtoh (end of GPU-side work)

    def hot_path():
        """One complete pipeline iteration.  Allocation-free hot path."""
        # ── GPU PORTION ───────────────────────────────────────────────────────
        ev_start.record(stream)
        # 1. BEV htod
        cuda.memcpy_htod_async(eng.inputs[in_name]['buf'], bev_pinned, stream)
        # 2. TRT engine
        eng.context.execute_async_v3(stream.handle)
        # 3. CUDA peak-finding + box-decode (queued on same stream)
        postproc.run(gpu_ptrs, stream)
        ev_postproc.record(stream)   # time up to here = GPU compute
        ev_dtoh.record(stream)       # placeholder (dtoh is synchronous below)

        # ── CPU PORTION (after GPU sync) ───────────────────────────────────────
        stream.synchronize()         # GPU work is done; managed mem readable

        t0 = time.perf_counter()
        dets = postproc.collect_detections()  # dtoh + circle NMS (both CPU)
        cpu_ms = (time.perf_counter() - t0) * 1000.0

        gpu_ms = ev_start.time_till(ev_postproc)   # htod + engine + kernels
        return gpu_ms, cpu_ms, len(dets)

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f'[bench_e2e] Warming up ({args.warmup} iters)...')
    for _ in range(args.warmup):
        hot_path()

    # ── Timed iterations ──────────────────────────────────────────────────────
    monitor = TegrastatsMonitor(interval_ms=args.power_interval)
    monitor.start()

    print(f'[bench_e2e] Timing {args.iterations} iterations...')
    gpu_lat, cpu_lat, n_dets_list = [], [], []

    for i in range(args.iterations):
        g, c, n = hot_path()
        gpu_lat.append(g)
        cpu_lat.append(c)
        n_dets_list.append(n)
        if (i + 1) % 50 == 0:
            total = g + c
            print(f'[bench_e2e] {i + 1}/{args.iterations}  '
                  f'gpu={g:.2f}ms  cpu={c:.2f}ms  total={total:.2f}ms  '
                  f'dets={n}')

    power = monitor.stop()

    gpu_arr   = np.array(gpu_lat)
    cpu_arr   = np.array(cpu_lat)
    total_arr = gpu_arr + cpu_arr

    def stats(arr):
        return {
            'mean': float(np.mean(arr)),
            'p50':  float(np.percentile(arr, 50)),
            'p90':  float(np.percentile(arr, 90)),
            'p99':  float(np.percentile(arr, 99)),
            'min':  float(np.min(arr)),
            'max':  float(np.max(arr)),
            'jitter_ms': float(np.percentile(arr, 99) - np.percentile(arr, 50)),
        }

    results = {
        'engine':     str(args.engine),
        'variant':    variant,
        'iterations': args.iterations,
        'avg_detections_per_frame': float(np.mean(n_dets_list)),
        'latency_ms': {
            'gpu_portion':   stats(gpu_arr),    # htod + engine + CUDA kernels
            'cpu_portion':   stats(cpu_arr),    # dtoh + circle NMS
            'total_pipeline': stats(total_arr),  # full frame latency
        },
        'fps': float(1000.0 / np.mean(total_arr)),
        'power': power,
    }

    if power.get('available'):
        vdd_in = power['vdd_in_w']['mean']
        results['mj_per_frame_vdd_in'] = float(
            vdd_in * np.mean(total_arr)
        )
        vdd_cv = power['vdd_cpu_gpu_cv_w']['mean']
        results['mj_per_frame_vdd_cv'] = float(
            vdd_cv * np.mean(total_arr)
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    g = results['latency_ms']['gpu_portion']
    c = results['latency_ms']['cpu_portion']
    t = results['latency_ms']['total_pipeline']

    print(f'\n[bench_e2e] Results for {variant}:')
    print('  GPU portion  (htod + engine + CUDA kernels):')
    print(f'    p50={g["p50"]:.2f}ms  p99={g["p99"]:.2f}ms  '
          f'jitter={g["jitter_ms"]:.2f}ms')
    print('  CPU portion  (dtoh + circle NMS):')
    print(f'    mean={c["mean"]:.2f}ms')
    print('  Total pipeline:')
    print(f'    p50={t["p50"]:.2f}ms  p99={t["p99"]:.2f}ms  '
          f'jitter={t["jitter_ms"]:.2f}ms  FPS={results["fps"]:.1f}')
    if power.get('available'):
        cv = power['vdd_cpu_gpu_cv_w']
        vi = power['vdd_in_w']
        print('  Power:')
        print(f'    {cv["mean"]:.2f}W VDD_CPU_GPU_CV  / '
              f'{vi["mean"]:.2f}W VDD_IN')
        print(f'    {results["mj_per_frame_vdd_in"]:.2f} mJ/frame (VDD_IN)')
    print(f'[bench_e2e] Saved: {args.out}')


if __name__ == '__main__':
    main()

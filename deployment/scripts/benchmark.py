"""
Benchmark TRT INT8 backbone+neck+head engine latency and power on Jetson.

Measures GPU inference time using CUDA events (accurate to ~microsecond,
immune to OS scheduling jitter that distorts time.perf_counter() on
multi-task Jetson workloads). Reports p50/p99/mean latency and mean power.

Power monitoring: parses tegrastats output (same approach as EdgeDrive
hardware_monitor.py, validated on this hardware). Reads VDD_CPU_GPU_CV
(CPU+GPU+CV rail — the inference-relevant number) and VDD_IN (total
module power). tegrastats must be accessible in the container — mount it
from the host via docker-compose or install via the l4t-jetpack image.

Prerequisites (on host, before docker run):
  sudo jetson_clocks   <- locks GPU to max clock; without this, GPU idles at
                          low frequency and latency numbers are not reproducible

Usage:
  docker compose -f deployment/docker/docker-compose.yml run --rm benchmark
  VARIANT=pruned25 docker compose -f deployment/docker/docker-compose.yml \
      run --rm benchmark
"""

import argparse
import json
import re
import subprocess
import threading
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# CenterPoint BEV grid shape: [1, 64, 512, 512]
BEV_SHAPE = (1, 64, 512, 512)

# Regex patterns from EdgeDrive hardware_monitor.py (validated on this hw)
_PWR_CV_RE = re.compile(r'\bVDD_CPU_GPU_CV\s+(\d+)mW')
_PWR_IN_RE = re.compile(r'\bVDD_IN\s+(\d+)mW')


class Engine:
    """Minimal TRT 10.x inference wrapper (static-shape BNH engine)."""

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

    def feed(self, name: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr.astype(self.inputs[name]['dtype']))
        cuda.memcpy_htod_async(self.inputs[name]['buf'], arr, self.stream)

    def run(self) -> None:
        self.context.execute_async_v3(self.stream.handle)

    def sync(self) -> None:
        self.stream.synchronize()


class TegrastatsMonitor:
    """Reads VDD_CPU_GPU_CV and VDD_IN from tegrastats in a background thread.

    Same parsing logic as EdgeDrive hardware_monitor.py, adapted to collect
    readings into a list instead of rendering a UI.
    """

    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = interval_ms
        self._readings_cv = []   # VDD_CPU_GPU_CV in W
        self._readings_in = []   # VDD_IN in W
        self._proc = None
        self._thread = None
        self._stop = threading.Event()

    def start(self) -> bool:
        try:
            self._proc = subprocess.Popen(
                ['tegrastats', '--interval', str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            self._thread = threading.Thread(target=self._read, daemon=True)
            self._thread.start()
            print(f'[bench] Power monitoring: tegrastats --interval '
                  f'{self.interval_ms}ms')
            return True
        except FileNotFoundError:
            print('[bench] Power monitoring: tegrastats not found in container '
                  '(mount /usr/bin/tegrastats from host or skip --power)')
            return False

    def _read(self) -> None:
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            cv = _PWR_CV_RE.search(line)
            if cv:
                self._readings_cv.append(int(cv.group(1)) / 1000.0)  # mW → W
            vi = _PWR_IN_RE.search(line)
            if vi:
                self._readings_in.append(int(vi.group(1)) / 1000.0)

    def stop(self) -> dict:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=3)

        if not self._readings_cv:
            return {'available': False}
        return {
            'available': True,
            'vdd_cpu_gpu_cv_w': {
                'mean': float(np.mean(self._readings_cv)),
                'max': float(np.max(self._readings_cv)),
                'n_samples': len(self._readings_cv),
            },
            'vdd_in_w': {
                'mean': (float(np.mean(self._readings_in))
                         if self._readings_in else None),
                'max': (float(np.max(self._readings_in))
                        if self._readings_in else None),
            },
        }


def _load_bev_sample(calib_bev_dir: str) -> np.ndarray:
    files = sorted(Path(calib_bev_dir).glob('*.npy'))
    if not files:
        raise FileNotFoundError(f'No .npy files in {calib_bev_dir}')
    arr = np.load(files[0]).astype(np.float32)
    return arr[np.newaxis] if arr.ndim == 3 else arr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Benchmark CenterPoint BNH TRT INT8 engine on Jetson'
    )
    p.add_argument(
        '--engine', required=True,
        help='Path to pts_backbone_neck_head.engine'
    )
    p.add_argument(
        '--calib-bev', default=None,
        help='Directory of [64,512,512] .npy BEV features. '
             'If omitted, uses synthetic random input.'
    )
    p.add_argument(
        '--warmup', type=int, default=20,
        help='Warmup iterations before timing (default: 20)'
    )
    p.add_argument(
        '--iterations', type=int, default=200,
        help='Timed iterations (default: 200)'
    )
    p.add_argument(
        '--power-interval', type=int, default=100,
        help='tegrastats sampling interval in ms (default: 100 = 10 Hz)'
    )
    p.add_argument(
        '--out', default='/workspace/output/benchmark.json',
        help='Output JSON path'
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f'[bench] Loading engine: {args.engine}')
    eng = Engine(args.engine)

    input_name = list(eng.inputs.keys())[0]
    print(f'[bench] Input: {input_name}  '
          f'shape={eng.inputs[input_name]["shape"]}')

    if args.calib_bev:
        bev = _load_bev_sample(args.calib_bev)
        print(f'[bench] Using BEV sample from {args.calib_bev}')
    else:
        bev = np.random.rand(*BEV_SHAPE).astype(np.float32)
        print('[bench] Using synthetic random BEV input')

    eng.feed(input_name, bev)

    # Warmup
    print(f'[bench] Warming up ({args.warmup} iters)...')
    for _ in range(args.warmup):
        eng.run()
    eng.sync()

    # Start power monitoring
    monitor = TegrastatsMonitor(interval_ms=args.power_interval)
    monitor.start()

    # Timed iterations using CUDA events
    print(f'[bench] Timing {args.iterations} iterations...')
    latencies_ms = []
    t_start = cuda.Event()
    t_end = cuda.Event()

    for i in range(args.iterations):
        t_start.record(eng.stream)
        eng.run()
        t_end.record(eng.stream)
        eng.sync()
        latencies_ms.append(t_start.time_till(t_end))

        if (i + 1) % 50 == 0:
            print(f'[bench] {i + 1}/{args.iterations}  '
                  f'last={latencies_ms[-1]:.2f}ms')

    power = monitor.stop()
    latencies_ms = np.array(latencies_ms)

    results = {
        'engine': str(args.engine),
        'variant': Path(args.engine).parent.name,
        'iterations': args.iterations,
        'latency_ms': {
            'mean': float(np.mean(latencies_ms)),
            'p50': float(np.percentile(latencies_ms, 50)),
            'p90': float(np.percentile(latencies_ms, 90)),
            'p99': float(np.percentile(latencies_ms, 99)),
            'min': float(np.min(latencies_ms)),
            'max': float(np.max(latencies_ms)),
        },
        'fps': float(1000.0 / np.mean(latencies_ms)),
        'power': power,
    }

    # mJ/frame uses VDD_CPU_GPU_CV (GPU+CPU rail) if available
    if power.get('available') and power['vdd_cpu_gpu_cv_w']['mean']:
        results['mj_per_frame'] = float(
            power['vdd_cpu_gpu_cv_w']['mean'] * np.mean(latencies_ms)
        )
    else:
        results['mj_per_frame'] = None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n[bench] Results for {results["variant"]}:')
    print(f'  Latency  mean={results["latency_ms"]["mean"]:.2f}ms  '
          f'p50={results["latency_ms"]["p50"]:.2f}ms  '
          f'p99={results["latency_ms"]["p99"]:.2f}ms')
    print(f'  FPS      {results["fps"]:.1f}')
    if power.get('available'):
        cv = power['vdd_cpu_gpu_cv_w']
        vi = power['vdd_in_w']
        print(f'  Power    {cv["mean"]:.2f}W VDD_CPU_GPU_CV  '
              f'/ {vi["mean"]:.2f}W VDD_IN')
        if results['mj_per_frame']:
            print(f'           {results["mj_per_frame"]:.2f} mJ/frame')
    print(f'[bench] Saved: {args.out}')


if __name__ == '__main__':
    main()

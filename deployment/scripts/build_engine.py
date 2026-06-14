"""
Build TRT engines for CenterPoint: INT8 for backbone+neck+head (the part
that matters for size/latency), FP16 for the voxel encoder (5KB FP32 ONNX —
negligible size, not worth INT8 calibration complexity).

TRT 10.3.0 API (Jetson Orin Nano Super, JetPack R36.4.0):
  - builder.build_serialized_network() — TRT10 removed build_cuda_engine()
    which returned an ICudaEngine directly; now returns serialized bytes,
    written straight to disk.
  - IInt8EntropyCalibrator2 interface unchanged from TRT8.x.

ASSUMPTIONS — verify before running:
  - backbone_neck_head spatial_features: [1, 64, 512, 512], STATIC shape.
    512x512 derived from point_cloud_range=[-51.2,-51.2,51.2,51.2] /
    voxel_size=0.2 ("pillar02" config). Override with --height/--width if
    your config differs.
  - Calibration data (--calib-dir) is a directory of .npy files, each a
    pre-computed spatial_features array [64, 512, 512] (float32) — i.e. BEV
    features AFTER pts_voxel_encoder + pts_middle_encoder, matching the
    backbone_neck_head ONNX's input directly. If jetson_calib contains
    something else (raw point clouds, pre-voxelized pillars), _load_batch
    in BEVFeatureCalibrator needs adjusting.

Usage:
    python3 build_engine.py \
        --onnx-dir /workspace/onnx --calib-dir /workspace/calib \
        --out /workspace/output/engines --variant pruned25
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class BEVFeatureCalibrator(trt.IInt8EntropyCalibrator2):
    """Feeds pre-computed BEV spatial_features [64,512,512] .npy files."""

    def __init__(self, calib_dir: str, shape: tuple, cache_file: str) -> None:
        super().__init__()
        self.cache_file = cache_file
        self.files = sorted(Path(calib_dir).glob('*.npy'))
        if not self.files:
            raise FileNotFoundError(f'No .npy files found in {calib_dir}')
        print(f'[calib] {len(self.files)} calibration samples in {calib_dir}')

        self.shape = shape  # (1, 64, 512, 512)
        self.batch_size = shape[0]
        nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
        self.device_input = cuda.mem_alloc(nbytes)
        self.index = 0

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list) -> list:
        if self.index >= len(self.files):
            return None
        arr = np.load(self.files[self.index]).astype(np.float32)
        arr = np.ascontiguousarray(arr.reshape(self.shape))
        cuda.memcpy_htod(self.device_input, arr)
        self.index += 1
        if self.index % 50 == 0 or self.index == len(self.files):
            print(f'[calib] {self.index}/{len(self.files)}')
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f'[calib] Using cached calibration: {self.cache_file}')
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache) -> None:
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f'[calib] Wrote calibration cache: {self.cache_file}')


def _new_network(builder: 'trt.Builder') -> 'trt.INetworkDefinition':
    return builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )


def _parse_onnx(network: 'trt.INetworkDefinition', onnx_path: str) -> None:
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f'Failed to parse {onnx_path}')


def _save_engine(serialized, out_path: str) -> None:
    if serialized is None:
        raise RuntimeError('Engine build failed — see parser/config errors above')
    # TRT 10.x build_serialized_network() returns IHostMemory, not bytes.
    # Use memoryview() to write and .nbytes for the size.
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(memoryview(serialized))
    print(f'[build] Saved: {out_path} ({serialized.nbytes / 1e6:.2f} MB)')


def build_backbone_neck_head_int8(
    onnx_path: str, calib_dir: str, out_path: str, height: int, width: int
) -> None:
    builder = trt.Builder(TRT_LOGGER)
    network = _new_network(builder)
    _parse_onnx(network, onnx_path)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    # INT8 with FP16 fallback for any layer TRT can't quantize.
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    input_name = network.get_input(0).name
    shape = (1, 64, height, width)

    # Static shape — spatial_features is always [1,64,H,W] for a single
    # nuScenes frame at this config's voxel_size/point_cloud_range.
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, shape, shape, shape)
    config.add_optimization_profile(profile)

    cache_file = str(Path(out_path).with_suffix('.calib_cache'))
    config.int8_calibrator = BEVFeatureCalibrator(calib_dir, shape, cache_file)

    print(f'[build] Building INT8 engine: {onnx_path} -> {out_path}')
    serialized = builder.build_serialized_network(network, config)
    _save_engine(serialized, out_path)


def build_encoder_fp32(onnx_path: str, out_path: str, max_voxels: int) -> None:
    """Build encoder as FP32 with a dynamic-shape profile.

    The encoder is tiny (5KB ONNX → ~0.11 MB engine) — FP16 is not worth the
    risk here. FP16 tensor-core tactics for the ReduceMax (max-pool over points
    per pillar) request 217MB+ of workspace during TRT optimization, which can
    exceed Jetson's available memory budget at build time. FP32 uses simpler
    tactics with much lower memory requirements and has negligible latency
    difference for a model this size.
    """
    builder = trt.Builder(TRT_LOGGER)
    network = _new_network(builder)
    _parse_onnx(network, onnx_path)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    # No precision flags — FP32 default, avoids tensor-core memory pressure.

    inp = network.get_input(0)
    # input_features: [num_voxels, num_max_points, 11] — dims 0,1 dynamic.
    feat_dim = inp.shape[2]
    max_pts = inp.shape[1] if inp.shape[1] > 0 else 32
    min_shape = (1, max_pts, feat_dim)
    opt_shape = (max_voxels // 2, max_pts, feat_dim)
    max_shape = (max_voxels, max_pts, feat_dim)

    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    print(f'[build] Building FP32 engine: {onnx_path} -> {out_path}')
    serialized = builder.build_serialized_network(network, config)
    _save_engine(serialized, out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build TRT engines for CenterPoint')
    p.add_argument(
        '--onnx-dir', required=True,
        help='compression/results/onnx_export/ (contains {variant}/ subdirs)'
    )
    p.add_argument(
        '--calib-dir', required=True,
        help='Directory of pre-computed BEV .npy calibration samples'
    )
    p.add_argument('--out', required=True, help='Output directory for .engine files')
    p.add_argument('--variant', required=True, help='e.g. fp32, pruned25')
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--max-voxels', type=int, default=40000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    onnx_dir = Path(args.onnx_dir) / args.variant
    out_dir = Path(args.out) / args.variant

    encoder_onnx = onnx_dir / f'pts_voxel_encoder_centerpoint_{args.variant}.onnx'
    bnk_onnx = onnx_dir / f'pts_backbone_neck_head_centerpoint_{args.variant}.onnx'

    build_encoder_fp32(
        str(encoder_onnx), str(out_dir / 'pts_voxel_encoder.engine'), args.max_voxels
    )
    build_backbone_neck_head_int8(
        str(bnk_onnx), args.calib_dir, str(out_dir / 'pts_backbone_neck_head.engine'),
        args.height, args.width
    )


if __name__ == '__main__':
    main()

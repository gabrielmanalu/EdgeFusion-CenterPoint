# ============================================================================
# EdgeFusion-CenterPoint — Jetson TRT INT8 Benchmark Container
# ============================================================================
#
# Purpose: build TRT INT8 engines from the exported ONNX files, benchmark
#          latency/power, and evaluate on-device mAP/NDS via a standalone
#          CenterPoint decode + nuscenes-devkit (no mmdet3d on ARM).
#
# Base: nvcr.io/nvidia/l4t-jetpack:r36.4.0
#   Same base as EdgeDrive-Perception → TensorRT 10.3.0, CUDA 12.6, cuDNN,
#   OpenCV 4.8.0 already present. TRT 10.x API: build_serialized_network()
#   (not the older build_cuda_engine()).
#
# Build (on Jetson only — ARM64, TRT engines are device-specific):
#   docker build -t edgefusion-centerpoint:latest -f deployment/docker/Dockerfile .
#
# Run:
#   docker compose -f deployment/docker/docker-compose.yml run --rm dev
# ============================================================================

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

LABEL maintainer="EdgeFusion-CenterPoint"
LABEL description="CenterPoint TRT INT8 benchmark — Jetson Orin Nano, TRT 10.3.0"
LABEL version="1.0"

ENV DEBIAN_FRONTEND=noninteractive

# l4t-jetpack's python3 (with TensorRT bindings pre-configured) doesn't
# include pip. Install via apt rather than symlinking python3 paths —
# the ros2 Dockerfile's symlink fix solved a different (ROS2 tooling)
# problem and is not needed here; applying it overwrites l4t's python3
# with the bare system python3, which lacks both pip AND the tensorrt
# module.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# nuscenes-devkit pulls in opencv-python, Shapely, scikit-learn, matplotlib —
# opencv-python likely conflicts/duplicates l4t's prebuilt CUDA-enabled
# OpenCV 4.8.0, and Shapely/scikit-learn may lack ARM64 wheels (build from
# source on Jetson = slow). Install with --no-deps and add only what
# DetectionEval/NuScenes actually import at runtime; expect to iterate on
# this list once errors surface on-device.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        nuscenes-devkit --no-deps && \
    python3 -m pip install --no-cache-dir \
        pyquaternion \
        cachetools \
        descartes \
        fire \
        pandas \
        tqdm \
        scipy

# pycuda: GPU memory alloc/copy for the TRT INT8 calibrator (build_engine.py).
# Needs CUDA toolkit headers (nvcc, cuda.h) — present in l4t-jetpack at
# /usr/local/cuda. Build from source on ARM can take several minutes.
ENV PATH="/usr/local/cuda/bin:${PATH}"
RUN python3 -m pip install --no-cache-dir pycuda

WORKDIR /workspace

CMD ["bash"]
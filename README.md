# EdgeFusion-CenterPoint

> Compression and edge deployment of CenterPoint LiDAR 3D detection for the Autoware autonomous-driving stack on Jetson Orin Nano.

[![CI](https://github.com/gabrielmanalu/EdgeFusion-CenterPoint/actions/workflows/ci.yml/badge.svg)](https://github.com/gabrielmanalu/EdgeFusion-CenterPoint/actions)

---

## Overview

This project compresses [CenterPoint](https://arxiv.org/abs/2006.11275) — a pillar-based 3D LiDAR object detector — for deployment within [Autoware](https://github.com/autowarefoundation/autoware) on a Jetson Orin Nano (8 GB, 15 W). The pipeline covers the full arc from an evaluated FP32 baseline through quantization, pruning, and distillation to a production-format TensorRT engine running in a ROS 2 Humble node emitting `autoware_perception_msgs::DetectedObjects`.

```
nuScenes LiDAR ──► Pillar Encoder ──► 2D Backbone + SecFPN ──► Center Head ──► 3D Boxes
                       (ONNX)                  (ONNX)             (custom CUDA)
```

---

## Stack

| Component | Version |
|---|---|
| PyTorch | 2.1.0 + cu118 |
| mmdetection3d | 1.3.0 (autowarefoundation fork) |
| TensorRT | 8.x |
| ROS 2 | Humble |
| CUDA | 12.8 |
| Hardware — cloud | NVIDIA A40 48 GB |
| Hardware — edge | Jetson Orin Nano Super 8 GB |

---

## Results

### Accuracy — nuScenes val

| Variant | mAP | NDS |
|---|---|---|
| FP32 baseline | **48.15** | **59.22** |
| PTQ INT8 | **48.20** | **59.18** |
| QAT INT8 | — | — |
| Pruned + QAT | — | — |
| Distilled | — | — |

### Edge performance — Jetson Orin Nano (15 W)

| Variant | FPS | p99 (ms) | Power (W) | mJ/frame |
|---|---|---|---|---|
| FP32 | — | — | — | — |
| QAT INT8 | — | — | — | — |
| Operating point | — | — | — | — |

---

## Repository layout

```
baseline/           FP32 eval harness + Autoware-ONNX export
compression/        PTQ · sensitivity · QAT · pruning · distillation · Pareto
deployment/         TensorRT engines · custom CUDA center-head · Jetson benchmarks
ros2_autoware/      ROS 2 node → autoware_perception_msgs
docs/               design decisions · jd_coverage
configs/            project-level config wrappers
```

---

## Quick start

### Environment

```bash
# First-time setup on a new pod (installs miniconda, conda env, all packages)
bash script/setup_env.sh

# Subsequent sessions — activate the existing env
source /workspace/activate_env.sh

# Verify
python -c "import torch, mmdet3d; print(torch.__version__, mmdet3d.__version__)"
```

### Eval FP32 baseline

```bash
cd /workspace/mmdetection3d
python tools/test.py \
    configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
    /workspace/data/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth \
    --task lidar_det
# mAP: 0.4815   NDS: 0.5922
```

### Compression sweep

```bash
python compression/ptq.py        --config ... --checkpoint ...
python compression/sensitivity.py --config ... --checkpoint ...
python compression/qat.py         --config ... --checkpoint ... --sensitivity ...
```

---

## Acknowledgements

- [CenterPoint](https://github.com/tianweiy/CenterPoint) — Yin et al., 2021
- [autowarefoundation/mmdetection3d](https://github.com/autowarefoundation/mmdetection3d)
- [Autoware](https://github.com/autowarefoundation/autoware)
- [EdgeDrive-Perception](https://github.com/gabrielmanalu/EdgeDrive-Perception) — prior project; pillar CUDA kernels reused
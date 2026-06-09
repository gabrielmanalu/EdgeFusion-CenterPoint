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
| CUDA | 11.8 |
| Hardware — cloud | NVIDIA A40 48 GB |
| Hardware — edge | Jetson Orin Nano Super 8 GB |

---

## Results

### Accuracy — nuScenes val

| Variant | mAP | NDS |
|---|---|---|
| FP32 baseline | — | — |
| PTQ INT8 | — | — |
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

```bash
# 1. Activate the conda environment (A40 cloud pod)
source /workspace/activate_env.sh

# 2. FP32 baseline evaluation
python baseline/eval.py \
    --config configs/centerpoint_pillar02_circlenms_nus.py \
    --checkpoint /workspace/data/centerpoint/centerpoint_nuscenes.pth

# 3. Compression sweep
python compression/ptq.py --config ... --checkpoint ...
python compression/sensitivity.py --config ... --checkpoint ...
python compression/qat.py --config ... --checkpoint ... --sensitivity ...
```

---

## Acknowledgements

- [CenterPoint](https://github.com/tianweiy/CenterPoint) — Yin et al., 2021
- [autowarefoundation/mmdetection3d](https://github.com/autowarefoundation/mmdetection3d)
- [Autoware](https://github.com/autowarefoundation/autoware)
- [EdgeDrive-Perception](https://github.com/gabrielmanalu/EdgeDrive-Perception) — prior project; pillar CUDA kernels reused

# EdgeFusion-CenterPoint

> Compression and edge deployment of CenterPoint LiDAR 3D detection for the Autoware autonomous-driving stack on Jetson Orin Nano.

[![CI](https://github.com/gabrielmanalu/EdgeFusion-CenterPoint/actions/workflows/ci.yml/badge.svg)](https://github.com/gabrielmanalu/EdgeFusion-CenterPoint/actions)

---

INT8-compressed CenterPoint LiDAR 3D detector for real-time autonomous driving inference
on Jetson Orin Nano 8GB (15W).

Covers the full pipeline from FP32 baseline evaluation through INT8 quantization,
structured pruning, knowledge distillation, TensorRT deployment, and Autoware integration.
Target: `autoware_perception_msgs::DetectedObjects` from a 15W edge device.

---

## Why this project

Modern production autonomous driving stacks run 3D LiDAR detection on data center GPUs.
Deploying the same quality perception on a 15W edge device requires careful compression
without destroying detection reliability.

This project answers: how much of a state-of-the-art LiDAR detector can be preserved
at INT8 precision and reduced channel count on Jetson Orin Nano, and what does the
accuracy / latency / power trade-off look like across the full Pareto front?

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EdgeFusion-CenterPoint                           │
├─────────────────────────────────────────────────────────────────────┤
│  Input: PointCloud2 (nuScenes LIDAR_TOP, 32-beam)                   │
│                                                                     │
│  Voxelization → PointPillars encoder → Pillar scatter (512×512 BEV) │
│       ↓                                                             │
│  SECOND backbone  [3 blocks, stride 1/2/2]                          │
│  SECONDFPN neck   [upsample → 512×512 concat]                       │
│       ↓                                                             │
│  CenterPoint head [10 classes × 6 regression tasks]                 │
│  heatmap / reg / height / dim / rot / vel                           │
│       ↓                                                             │
│  Circle-NMS → DetectedObjects                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Compression stack (cloud A40)                                      │
│  PTQ INT8 → Sensitivity → QAT → Pruning sweep → Distillation        │
├─────────────────────────────────────────────────────────────────────┤
│  Deployment (Jetson Orin Nano 8GB / 15W)                            │
│  ONNX → TRT INT8 engine → ROS2 node → Autoware                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Results

### Quantization

| Variant       | mAP    | NDS    | Δ mAP  | Δ NDS  |
| ------------- | ------ | ------ | ------ | ------ |
| FP32 baseline | 0.4815 | 0.5922 | —      | —      |
| PTQ INT8      | 0.4812 | 0.5903 | −0.03% | −0.19% |
| QAT INT8      | TBD    | TBD    | —      | —      |

### Per-class AP (FP32 baseline)

| Class      | AP    | Class                | AP    |
| ---------- | ----- | -------------------- | ----- |
| Car        | 0.836 | Motorcycle           | 0.416 |
| Pedestrian | 0.761 | Barrier              | 0.596 |
| Bus        | 0.605 | Traffic cone         | 0.533 |
| Truck      | 0.483 | Bicycle              | 0.154 |
| Trailer    | 0.326 | Construction vehicle | 0.107 |

### Pruning + Distillation (in progress)

| Variant         | Prune ratio | mAP | NDS | Latency (Jetson) | Power |
| --------------- | ----------- | --- | --- | ---------------- | ----- |
| QAT INT8        | 0%          | TBD | TBD | TBD              | TBD   |
| Pruned + QAT    | 25%         | TBD | TBD | TBD              | TBD   |
| Pruned + QAT    | 40%         | TBD | TBD | TBD              | TBD   |
| Pruned + QAT    | 55%         | TBD | TBD | TBD              | TBD   |
| Distilled + QAT | —           | TBD | TBD | TBD              | TBD   |

---

## Repository Structure

```
EdgeFusion-CenterPoint/
├── baseline/              ← FP32 evaluation + ONNX export
│   ├── eval.py
│   ├── export_onnx.py
│   └── README.md
├── compression/           ← PTQ, sensitivity, QAT, pruning, distillation, pareto
│   ├── ptq.py
│   ├── sensitivity.py
│   ├── qat.py
│   ├── pruning.py
│   ├── distillation.py
│   ├── pareto.py
│   ├── check_fakequant.py
│   ├── results/
│   └── README.md
├── docs/
│   └── design_decisions.md
├── scripts/
│   └── setup_env.sh       ← new pod environment setup (~60-90 min)
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Quick Start

### Environment setup (new pod)

```bash
git clone https://github.com/gabrielmanalu/EdgeFusion-CenterPoint.git
cd /workspace/mmdetection3d
bash EdgeFusion-CenterPoint/scripts/setup_env.sh
source /workspace/activate_env.sh

# Verify
python -c "import torch, mmdet3d; print(torch.__version__, mmdet3d.__version__)"
```

Setup installs mmcv 2.1.0 from source (~60-90 min), mmdet3d from the
autowarefoundation fork, and all dependencies.

### Set variables

```bash
CFG=configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py
CKPT=/workspace/data/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth
PTQ=EdgeFusion-CenterPoint/compression/results/ptq/ptq_calibrated.pth
```

### 1. Evaluate FP32 baseline

```bash
python EdgeFusion-CenterPoint/baseline/eval.py \
    --config $CFG --checkpoint $CKPT
```

### 2. Export ONNX

```bash
python EdgeFusion-CenterPoint/baseline/export_onnx.py \
    --config $CFG --checkpoint $CKPT \
    --out-dir /workspace/data/centerpoint/onnx_multitask
```

### 3. PTQ calibration + INT8 eval (~35 min)

```bash
python EdgeFusion-CenterPoint/compression/ptq.py \
    --config $CFG --checkpoint $CKPT --calib-size 512
```

### 4. Sensitivity analysis (~65 min)

```bash
python EdgeFusion-CenterPoint/compression/sensitivity.py \
    --config $CFG --fp32-ckpt $CKPT --ptq-ckpt $PTQ
```

### 5. QAT fine-tuning (~5-6 hrs)

```bash
python EdgeFusion-CenterPoint/compression/qat.py \
    --config $CFG \
    --fp32-ckpt $CKPT \
    --ptq-ckpt $PTQ \
    --sensitivity EdgeFusion-CenterPoint/compression/results/sensitivity/sensitivity.json \
    --ptq-map 0.4812 --epochs 5 --batch-size 4
```

---

## Documentation

| Document                                               | Description                                                                      |
| ------------------------------------------------------ | -------------------------------------------------------------------------------- |
| [`baseline/README.md`](baseline/README.md)             | FP32 eval, ONNX export, Autoware param mapping                                   |
| [`compression/README.md`](compression/README.md)       | PTQ, sensitivity, QAT — implementation details, toolkit choice, design decisions |
| [`docs/design_decisions.md`](docs/design_decisions.md) | Architecture choices, checkpoint sourcing, quantization toolkit, ONNX export     |

---

## Key Design Decisions

**open-mmlab checkpoint over Autoware's**
Autoware does not release `.pth` weights — their model is trained on proprietary TIER IV
data. We use the open-mmlab nuScenes checkpoint and reproduce the published baseline
within ±0.6%, giving a fully reproducible starting point.

**Custom ONNX exporter**
Autoware's provided converter targets their 5-class model. Our 10-class multi-task head
cannot be exported with their script without modification. `export_onnx.py` directly
traces each head task and assembles the two-model structure Autoware expects.

**torch.ao over pytorch-quantization**
NVIDIA's `pytorch-quantization` has a C++ ABI mismatch against PyTorch 2.1.0+cu118 in
this environment, and source build fails due to CUDA 12.8 vs PyTorch's 11.8 requirement.
`torch.ao.quantization` is built into PyTorch and requires no separate compilation.

**Backbone and neck quantized, head kept FP32**
`pts_bbox_head` is ~4% of FLOPs. Its output layers (heatmap sigmoid, regression heads)
are sensitive to precision loss, and keeping them FP32 is standard practice in TRT AMP
for detection networks. Quantization effort concentrates on the SECOND backbone and
SECONDFPN neck where the gains are.

**PTQ = FP32 is correct, not a bug**
CenterPoint's BatchNorm architecture normalizes activations at every conv layer,
producing bounded symmetric distributions — exactly what INT8 quantization requires.
The result (−0.03% mAP) is genuine, not a measurement artifact.

**FP16 default in Autoware, INT8 here**
Autoware's production `autoware_lidar_centerpoint` defaults to `trt_precision: fp16`.
Our INT8 target is more aggressive and produces a 2× memory reduction and hardware
INT8 throughput gains on Jetson's 1024-core Ampere GPU.

---

## Hardware

| Component              | Spec                                             |
| ---------------------- | ------------------------------------------------ |
| Training / compression | RunPod A40 (48GB VRAM, CUDA 12.8)                |
| Deployment target      | Jetson Orin Nano 8GB (1024-core Ampere, 67 TOPS) |
| TDP target             | 15W                                              |

---

## Dataset

nuScenes v1.0 — 700 training scenes, 150 validation scenes, 6019 val keyframes.

```
data/nuscenes/
├── samples/LIDAR_TOP/     ← point cloud sweeps (.pcd.bin)
├── sweeps/LIDAR_TOP/
├── maps/
└── v1.0-trainval/         ← annotations, calibration, scene metadata
```

pkl files (`nuscenes_infos_train.pkl`, `nuscenes_infos_val.pkl`,
`nuscenes_dbinfos_train.pkl`) are generated by mmdet3d's `create_data.py`
and must be placed at `data/nuscenes/` or the paths specified in the config.

---

## Acknowledgements

- [CenterPoint](https://github.com/tianweiy/CenterPoint) — Yin et al., 2021
- [autowarefoundation/mmdetection3d](https://github.com/autowarefoundation/mmdetection3d)
- [Autoware](https://github.com/autowarefoundation/autoware)
- [EdgeDrive-Perception](https://github.com/gabrielmanalu/EdgeDrive-Perception) — prior project; pillar CUDA kernels reused

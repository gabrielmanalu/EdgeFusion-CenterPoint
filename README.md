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

### Compression Sweep

| Variant              | mAP    | NDS    | Params | Δ mAP  |
| -------------------- | ------ | ------ | ------ | ------ |
| FP32 baseline        | 0.4815 | 0.5922 | 100%   | —      |
| PTQ INT8             | 0.4812 | 0.5903 | 100%   | −0.03% |
| QAT INT8             | 0.4814 | 0.5910 | 100%   | −0.01% |
| Pruned 25%           | 0.4081 | 0.5382 | 56.4%  | −15.2% |
| Pruned 40%           | 0.2838 | 0.3902 | 36.0%  | −41.0% |
| Pruned 55%           | 0.2149 | 0.3136 | 20.3%  | −55.4% |
| Distilled (25% arch) | 0.4094 | 0.5344 | 56.4%  | −15.0% |

INT8 quantization is essentially free on this architecture (BatchNorm normalizes
activations before every conv, producing INT8-friendly distributions). One-shot
magnitude pruning shows a steep, accelerating accuracy cost past 25% — see
[`compression/README.md`](compression/README.md) for the full pruning sweep analysis,
per-class breakdown, and the EMA/BatchNorm buffer bug + recalibration recovery.

Knowledge distillation (same architecture/init/budget as Pruned 25%, with added teacher
guidance) produced essentially the same mAP (+0.32%, within noise) but worse NDS
(−0.71% — mAOE and mAVE both degraded). **Pruned 25% (task-loss-only) remains the
practical choice** for this ratio; see `compression/README.md` for the full root-cause
analysis (background-dominated heatmap distillation loss, teacher/student spatial
misalignment in regression distillation).

### Per-class AP (FP32 baseline)

| Class      | AP    | Class                | AP    |
| ---------- | ----- | -------------------- | ----- |
| Car        | 0.836 | Motorcycle           | 0.416 |
| Pedestrian | 0.761 | Barrier              | 0.596 |
| Bus        | 0.605 | Traffic cone         | 0.533 |
| Truck      | 0.483 | Bicycle              | 0.154 |
| Trailer    | 0.326 | Construction vehicle | 0.107 |

### Per-class AP — Pruning Sweep

| Class                | FP32  | Pruned 25% | Distilled 25% | Pruned 40% | Pruned 55% |
| -------------------- | ----- | ---------- | ------------- | ---------- | ---------- |
| car                  | 0.836 | 0.806      | 0.804         | 0.714      | 0.634      |
| pedestrian           | 0.761 | 0.717      | 0.712         | 0.601      | 0.507      |
| bus                  | 0.605 | 0.570      | 0.571         | 0.424      | 0.326      |
| barrier              | 0.596 | 0.533      | 0.534         | 0.319      | 0.221      |
| traffic_cone         | 0.533 | 0.429      | 0.427         | 0.252      | 0.167      |
| truck                | 0.483 | 0.415      | 0.425         | 0.248      | 0.137      |
| motorcycle           | 0.416 | 0.270      | 0.263         | 0.144      | 0.096      |
| trailer              | 0.326 | 0.249      | 0.265         | 0.109      | 0.054      |
| bicycle              | 0.154 | 0.034      | 0.035         | 0.003      | 0.000      |
| construction_vehicle | 0.107 | 0.059      | 0.059         | 0.024      | 0.006      |

Distillation's per-class deltas vs Pruned 25% are mostly noise-level (trailer +0.016,
truck +0.010, motorcycle −0.007). The two classes `L_heatmap_distill` specifically
targeted — bicycle and construction_vehicle — show **no change**.

### Pareto Candidates (Jetson deployment — pending)

Two views are assembled by `compression/pareto.py`: an **architecture Pareto**
(measured params vs mAP/NDS) and a **projected deployment Pareto** (size under TRT
INT8, assuming the validated near-free INT8 factor — 0.25× — extends to pruned
architectures, pending Jetson validation).

| Variant                    | mAP             | Projected size (TRT INT8) | Status                      | Latency | Power |
| -------------------------- | --------------- | ------------------------- | --------------------------- | ------- | ----- |
| QAT INT8 (full arch)       | 0.4814          | 25.0%                     | measured (A40 FakeQuantize) | TBD     | TBD   |
| Pruned 25% / Distilled 25% | 0.4081 / 0.4094 | 14.1%                     | projected                   | TBD     | TBD   |
| Pruned 40%                 | 0.2838          | 9.0%                      | projected                   | TBD     | TBD   |
| Pruned 55%                 | 0.2149          | 5.1%                      | projected                   | TBD     | TBD   |

Pruned 25% and Distilled land at effectively the same point (14.1% size, ~0.408-0.409
mAP) — distillation didn't shift the Pareto front (see Compression Sweep note above).

INT8 quantization shown near-free for the unpruned architecture (PTQ 0.4812 / QAT
0.4814 vs FP32 0.4815). Pruned architectures retain BatchNorm after every conv (the
property responsible for this), so the same factor is _projected_ for pruned variants —
this is exactly what TRT INT8 benchmarking on Jetson Orin Nano validates next. TRT
calibration uses the 512-sample `jetson_calib` set.

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
│   ├── results/            ← checkpoints not in repo, see Checkpoints below
│   │   └── pareto/
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
git clone https://github.com/gabrielmanalu/EdgeFusion-CenterPoint.git \
    /workspace/mmdetection3d/EdgeFusion-CenterPoint
cd /workspace/mmdetection3d/EdgeFusion-CenterPoint
bash script/setup_env.sh 2>&1 | tee /workspace/setup_env.log
source /workspace/activate_env.sh
```

Setup installs PyTorch 2.1.0+cu118, mmcv 2.1.0, mmdet3d from the
autowarefoundation fork, and all dependencies (~60-90 min).

### Set variables

```bash
CFG=configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py
CKPT=/workspace/data/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth
PTQ=EdgeFusion-CenterPoint/compression/results/ptq/ptq_calibrated.pth
QAT=EdgeFusion-CenterPoint/compression/results/qat/qat_best.pth
```

### 1. Evaluate FP32 baseline (~30 min)

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

### 5. QAT fine-tuning (~12 hrs)

```bash
python EdgeFusion-CenterPoint/compression/qat.py \
    --config $CFG \
    --fp32-ckpt $CKPT \
    --ptq-ckpt $PTQ \
    --sensitivity EdgeFusion-CenterPoint/compression/results/sensitivity/sensitivity.json \
    --ptq-map 0.4812 --epochs 5 --batch-size 4
```

### 6. Structured pruning sweep (~8.5 hrs per ratio)

```bash
for RATIO in 0.25 0.40 0.55; do
    python EdgeFusion-CenterPoint/compression/pruning.py \
        --config $CFG --checkpoint $CKPT \
        --ratio $RATIO --epochs 5 \
        --batch-size 16 --lr 4e-4 --num-workers 16
done
```

### 7. Knowledge distillation (~8.5 hrs)

```bash
python EdgeFusion-CenterPoint/compression/distillation.py \
    --config $CFG --checkpoint $CKPT \
    --ratio 0.25 --epochs 5 \
    --batch-size 16 --lr 4e-4 --num-workers 16 \
    --alpha 1.0 --beta 1.0
```

---

## Checkpoints

Model weights are not stored in this repository.

**Download from Google Drive:** _(link to be added)_

| File                                                  | Description                                         | Size    |
| ----------------------------------------------------- | --------------------------------------------------- | ------- |
| `centerpoint_02pillar_...pth`                         | FP32 baseline (open-mmlab)                          | 24 MB   |
| `ptq_calibrated.pth`                                  | PTQ INT8 calibrated checkpoint                      | 24 MB   |
| `qat_best.pth`                                        | QAT INT8 fine-tuned checkpoint                      | 24 MB   |
| `sensitivity.json`                                    | Per-layer sensitivity results                       | 3 KB    |
| `pruned_25_recalib.pth`, `pruned_model_25_recalib.pt` | Pruned 25% (recalibrated)                           | ~14 MB  |
| `pruned_40.pth`, `pruned_model_40.pt`                 | Pruned 40%                                          | ~9 MB   |
| `pruned_55.pth`, `pruned_model_55.pt`                 | Pruned 55%                                          | ~5 MB   |
| `distilled_25.pth`, `distilled_model_25.pt`           | Distilled (25% arch)                                | ~14 MB  |
| `onnx_multitask/`                                     | Exported ONNX files (encoder + backbone-neck-head)  | 40 MB   |
| `jetson_calib/`                                       | 512 point clouds for TRT INT8 calibration on Jetson | ~200 MB |

The FP32 baseline checkpoint is also available from the
[open-mmlab model zoo](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/centerpoint).

After downloading, place files at:

```bash
mkdir -p EdgeFusion-CenterPoint/compression/results/{ptq,qat,sensitivity,pruning/ratio_25,pruning/ratio_40,pruning/ratio_55,distillation/ratio_25}

cp ptq_calibrated.pth EdgeFusion-CenterPoint/compression/results/ptq/
cp qat_best.pth EdgeFusion-CenterPoint/compression/results/qat/
cp sensitivity.json EdgeFusion-CenterPoint/compression/results/sensitivity/
cp pruned_*25*.p* EdgeFusion-CenterPoint/compression/results/pruning/ratio_25/
cp pruned_*40*.p* EdgeFusion-CenterPoint/compression/results/pruning/ratio_40/
cp pruned_*55*.p* EdgeFusion-CenterPoint/compression/results/pruning/ratio_55/
cp distilled_*25*.p* EdgeFusion-CenterPoint/compression/results/distillation/ratio_25/
```

---

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

**One-shot pruning over iterative pruning**
Channel pruning was applied in a single pass (L1-magnitude, deterministic given FP32
weights + ratio) followed by 5-epoch fine-tuning, rather than iterative prune-and-retrain
cycles. One-shot is simpler and ~3-5x cheaper, at the cost of a steeper accuracy falloff
past 25% (see pruning sweep results). Iterative pruning is the standard mitigation and
noted as future work if a sub-25% operating point is needed.

**Raw dataset for pruning/distillation fine-tuning, not CBGS**
CBGS class-balanced sampling inflates the dataset ~4.4x (1,759 → 7,724 steps/epoch),
making the 3-ratio pruning sweep cost ~112 hrs instead of ~25 hrs. Since fine-tuning
starts from FP32 weights that already encode CBGS-trained knowledge (validated by QAT:
raw dataset, 0.4814 mAP matching published numbers), the time savings were taken as a
documented trade-off — rare-class recovery is somewhat weaker without CBGS, visible in
the pruning sweep's bicycle/construction_vehicle results.

**Distillation as a controlled comparison, not a new architecture**
Rather than designing a novel "tiny" student, the distillation student uses the exact
same architecture/init/budget as Pruned 25% (same L1 channel selection from FP32). The
only difference is the loss function (task loss vs task + teacher distillation). This
isolates whether teacher guidance recovers pruning-induced capacity loss better than
fine-tuning alone — a directly comparable A/B result.

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

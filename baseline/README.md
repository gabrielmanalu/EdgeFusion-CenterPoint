# Baseline

FP32 evaluation and ONNX export for the open-mmlab CenterPoint model on nuScenes.

This establishes the accuracy baseline that all compressed variants are measured against,
and produces the ONNX files used for TensorRT deployment in Autoware.

---

## Results

| Metric | Value      | Published (open-mmlab) | Δ       |
| ------ | ---------- | ---------------------- | ------- |
| mAP    | **0.4815** | 0.4816                 | −0.0001 |
| NDS    | **0.5922** | 0.5936                 | −0.0014 |
| mATE   | 0.3256     | —                      | —       |
| mASE   | 0.2634     | —                      | —       |
| mAOE   | 0.3794     | —                      | —       |
| mAVE   | 0.3500     | —                      | —       |
| mAAE   | 0.1979     | —                      | —       |

Evaluated on nuScenes val split (6019 samples).
Result matches the published checkpoint within ±0.6%, confirming correct
dataset preparation and evaluation setup.

---

## Files

```
baseline/
├── eval.py             ← FP32 evaluation on nuScenes val
└── export_onnx.py      ← Multi-task ONNX export (encoder + backbone-neck-head)
```

---

## How to Run

Set environment variables first (from `/workspace/mmdetection3d`):

```bash
source /workspace/activate_env.sh

CFG=configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py
CKPT=/workspace/data/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth
```

### Evaluate FP32 baseline

```bash
python EdgeFusion-CenterPoint/baseline/eval.py \
    --config $CFG \
    --checkpoint $CKPT
```

Runs inference on all 6019 val samples (~30 min on A40) and prints per-class
AP alongside overall mAP and NDS. Results are saved to
`baseline/results/fp32_baseline.json`.

### Export to ONNX

```bash
python EdgeFusion-CenterPoint/baseline/export_onnx.py \
    --config $CFG \
    --checkpoint $CKPT \
    --out-dir /workspace/data/centerpoint/onnx_multitask
```

Produces two ONNX files:

```
onnx_multitask/
├── pts_voxel_encoder.onnx          ← [N, max_pts, 11] → [N, 1, 64]
└── pts_backbone_neck_head.onnx     ← [B, 64, H, W] →
                                         heatmap [B, 10, H, W]
                                         reg     [B, 12, H, W]
                                         height  [B,  6, H, W]
                                         dim     [B, 18, H, W]
                                         rot     [B, 12, H, W]
                                         vel     [B, 12, H, W]
```

---

## ONNX Architecture

Autoware's `autoware_lidar_centerpoint` node expects two separate ONNX models:
a voxel feature encoder and a backbone-neck-head network. The node handles
all preprocessing (voxelization, pillar scatter) and postprocessing (NMS,
box decoding) externally in C++.

### Why a custom exporter instead of Autoware's converter

Autoware's provided ONNX converter (`projects/AutowareCenterPoint/centerpoint_onnx_converter.py`)
targets the `centerpoint_custom` config — a 5-class model
(CAR, TRUCK, BUS, BICYCLE, PEDESTRIAN) trained on a combination of nuScenes and
TIER IV's internal dataset. It produces single-task heads designed for that
specific class set.

Our model uses the standard open-mmlab config with a 10-class multi-task head
structure. Autoware's converter cannot parse this head without modification.
`export_onnx.py` implements a direct export that traces each head task
separately and assembles them into the two-model structure Autoware expects,
preserving full 10-class output.

### Model parameters for Autoware deployment

When creating the Autoware param.yaml for this model, use the values from the
training config — not Autoware's published defaults, which target their
5-class model:

```yaml
/**:
  ros__parameters:
    class_names:
      [
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
      ]
    point_feature_size: 5
    max_voxel_size: 40000
    point_cloud_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size: [0.2, 0.2, 8.0]
    downsample_factor: 1
    encoder_in_feature_size: 11
```

Note the z-range difference from Autoware's deployed model (`-3.0/+5.0` vs
`-5.0/+3.0`) — this reflects different vehicle sensor mounting heights
between the nuScenes dataset vehicle and TIER IV's test vehicle. The z-range
must match the training config exactly.

---

## Checkpoint

The FP32 checkpoint used throughout this project:

```
centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth
```

Available at the open-mmlab mmdetection3d model zoo. This is the standard
nuScenes CenterPoint checkpoint trained with the pillar02 config for 20 epochs
using 4×8 GPU cyclic training.

# Design Decisions

---

## Model — CenterPoint (pillar variant)

Pillar encoding avoids 3D sparse convolution, making quantization predictable:
the backbone consists entirely of standard 2D `Conv2d` layers on a dense BEV
feature map, with no sparse-tensor bookkeeping that complicates fake-quant insertion.
Peak-finding postprocessing (max-pool NMS) replaces anchor-based NMS,
enabling a cleaner custom CUDA postprocessing kernel with fewer edge cases.

## Starting checkpoint

**open-mmlab CenterPoint nuScenes baseline**
(`centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus`)

Autoware does not release the `.pth` checkpoint for their production model;
their training included proprietary TIER IV sensor data and the resulting
weights are kept internal. The open-mmlab checkpoint uses the same
architecture and the same public nuScenes dataset.

Autoware's exported ONNX files are retained as:
1. A reference for validating the ONNX export format.
2. A comparison baseline: Autoware's published mAP/NDS numbers serve as
   the "production target" against which compression is evaluated.
3. A deployment comparison: the Autoware ONNX is also quantized via
   TensorRT INT8 calibration in the Jetson benchmark for a direct side-by-side.

## Dataset — nuScenes v1.0 full trainval

Matches Autoware's training regime (10 classes, 850 scenes, 10 Hz LiDAR).
Industry-standard benchmark for automotive LiDAR 3D detection.

## Quantization strategy — PTQ first, then QAT

PTQ establishes the naive INT8 baseline (fast, no training loop). Per-layer
sensitivity analysis identifies which layers drive most of the accuracy drop;
those layers remain in FP16 for the mixed-precision run. QAT then fine-tunes
the remaining INT8 layers with fake-quant nodes to recover the accuracy gap.

Contrast with EdgeDrive-Perception v1: QAT was applied without sensitivity
analysis and failed to converge. Sensitivity-guided mixed precision is the
specific fix introduced here.

## Experiment tracking — MLflow

All compression runs are tracked in a local MLflow server for reproducibility
and Pareto construction. Run naming convention is documented in `mlflow/README.md`.

---

*Updated as decisions are made.*

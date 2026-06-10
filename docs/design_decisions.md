# Design Decisions

---

## Model — CenterPoint (pillar variant)

Pillar encoding avoids 3D sparse convolution, making quantization predictable:
the backbone consists entirely of standard 2D `Conv2d` layers on a dense BEV
feature map, with no sparse-tensor bookkeeping that complicates fake-quant
insertion. Peak-finding postprocessing (max-pool NMS) replaces anchor-based
NMS, enabling a cleaner custom CUDA postprocessing kernel with fewer edge cases.

## Starting checkpoint

**open-mmlab CenterPoint nuScenes baseline**
(`centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth`)

Autoware does not release the `.pth` checkpoint for their production model;
their training included proprietary TIER IV sensor data. The open-mmlab
checkpoint uses the same architecture and the same public nuScenes dataset.

Autoware's exported ONNX files are retained as:
1. A reference for validating the ONNX export format.
2. A comparison baseline against Autoware's published mAP/NDS numbers.

## Dataset — nuScenes v1.0 full trainval

Matches Autoware's training regime (10 classes, 850 scenes, 10 Hz LiDAR).
Industry-standard benchmark for automotive LiDAR 3D detection.

## ONNX export — custom multi-task script

Autoware's `centerpoint_onnx_converter.py` cannot be used on the open-mmlab
checkpoint because:
1. It overrides the encoder type to `PillarFeatureNetAutoware`, which is
   architecturally incompatible with our standard `PillarFeatureNet`.
2. `CenterHeadONNX.forward()` only processes `task_heads[0]`, exporting
   only the first task head (car, 1 class) and silently dropping the other 9.

`baseline/export_onnx.py` reproduces the same two-ONNX split using:
- A forward hook on `pfn_layers[0]` to capture pre-PFN features for the
  encoder export (avoids reimplementing the relative feature computation).
- A `BackboneNeckHeadONNX` wrapper that concatenates all 6 task head outputs
  channel-wise: heatmap `[B, 10, H, W]`, reg/height/dim/rot/vel accordingly.

## Quantization toolkit — torch.ao.quantization (not pytorch-quantization)

NVIDIA pytorch-quantization was the original plan. It failed for two reasons:
1. The pre-built wheel has a C++ ABI mismatch against PyTorch 2.1.0+cu118.
2. Building from source requires CUDA 11.8 (`nvcc`), but the pod has only
   CUDA 12.8. There is no CUDA 11.8 toolkit available on the image.

`torch.ao.quantization` (built into PyTorch 2.1) was used instead:
- `prepare_fx` inserts `FakeQuantize` nodes that simulate INT8 on GPU via
  Straight-Through Estimator — no CUDA compilation required.
- The same prepared model serves both PTQ (calibration) and QAT (fine-tuning).
- Calibrated scales transfer to TRT via Q/DQ ops in the exported ONNX.

## Quantized submodules — backbone + neck only

| Module | FLOPs share | Decision |
|---|---|---|
| `pts_backbone` (SECOND) | ~75% | INT8 — Conv2d+BN, FX-traceable |
| `pts_neck` (SECONDFPN) | ~20% | INT8 — ConvTranspose2d, FX-traceable |
| `pts_bbox_head` (CenterHead) | ~4% | FP32 — output layers, sensitive |
| `pts_voxel_encoder` (PFN) | <1% | FP32 — negligible compute |

The head's final heatmap conv outputs feed directly into peak-finding
(the NMS replacement). INT8 rounding on these values shifts detection scores
and changes which peaks survive the threshold — high sensitivity, negligible
compute gain. TensorRT's automatic mixed precision applies the same heuristic.

## SECONDFPN FX tracing patch

`SECONDFPN.forward()` calls `len(x)` on the backbone output tuple. PyTorch's
FX symbolic tracer cannot evaluate `len()` on symbolic tensors. The fix:
temporarily patch `SECONDFPN.forward` to use `len(self.deblocks)` instead —
a Python constant visible to FX at trace time. The original method is restored
in a `finally` block so no other code is affected.

## PTQ accuracy — identical to FP32

PTQ INT8 achieved mAP 48.20 / NDS 59.18 vs FP32 baseline 48.15 / 59.22.
The change is within measurement noise (< 0.1%).

Root cause: CenterPoint's SECOND backbone applies BatchNorm after every Conv2d.
BatchNorm normalises activations to a consistent, outlier-free range — exactly
the condition under which INT8 histogram calibration is near-perfect. This is a
known property of BatchNorm-heavy architectures.

Consequence for QAT: QAT on the unpruned model recovers no accuracy (there is
nothing to recover). QAT is still demonstrated on the baseline to validate the
pipeline. Its practical value is on pruned models, where fewer channels reduce
redundancy and PTQ causes measurable accuracy drop — making QAT recovery
meaningful and the Pareto chart interesting.

## Experiment tracking — MLflow

All compression runs are tracked in a local MLflow server for reproducibility
and Pareto construction. Run naming convention is in `mlflow/README.md`.

---

*Updated as decisions are made.*
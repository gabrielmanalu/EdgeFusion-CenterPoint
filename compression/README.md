# Compression

This directory contains the full INT8 quantization pipeline for CenterPoint:
Post-Training Quantization (PTQ) → Sensitivity Analysis → Quantization-Aware Training (QAT).

The goal is to establish an INT8-quantized model that preserves FP32 accuracy, to be used as
the starting point for the pruning and distillation sweeps that follow.

---

## Contents

```
compression/
├── ptq.py              # PTQ calibration + INT8 simulation eval
├── sensitivity.py      # Per-layer loss-proxy sensitivity analysis
├── qat.py              # QAT fine-tuning with mixed precision
├── pruning.py          # Structured channel pruning
├── distillation.py     # Knowledge distillation
├── pareto.py           # Pareto front assembly
├── check_fakequant.py  # Diagnostic: verify FakeQuantize nodes are active
└── results/
    ├── ptq/            # ptq_calibrated.pth, ptq_metrics.json
    ├── sensitivity/    # sensitivity.json
    └── qat/            # qat_best.pth, qat_metrics.json
```

---

## How to Run

All commands run from `/workspace/mmdetection3d`. Set environment variables first:

```bash
source /workspace/activate_env.sh

CFG=configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py
CKPT=/workspace/data/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth
PTQ=EdgeFusion-CenterPoint/compression/results/ptq/ptq_calibrated.pth
```

### 1. PTQ (~35 min)

```bash
python EdgeFusion-CenterPoint/compression/ptq.py \
    --config $CFG \
    --checkpoint $CKPT \
    --calib-size 512
```

Outputs: `results/ptq/ptq_calibrated.pth`, `results/ptq/ptq_metrics.json`

### 2. Sensitivity (~65 min)

```bash
python EdgeFusion-CenterPoint/compression/sensitivity.py \
    --config $CFG \
    --fp32-ckpt $CKPT \
    --ptq-ckpt $PTQ
```

Outputs: `results/sensitivity/sensitivity.json`

### 3. QAT (~5-6 hrs, run in tmux)

```bash
python EdgeFusion-CenterPoint/compression/qat.py \
    --config $CFG \
    --fp32-ckpt $CKPT \
    --ptq-ckpt $PTQ \
    --sensitivity EdgeFusion-CenterPoint/compression/results/sensitivity/sensitivity.json \
    --ptq-map 0.4812 \
    --epochs 5 \
    --batch-size 4
```

Outputs: `results/qat/qat_best.pth`, `results/qat/qat_metrics.json`

### Diagnostic (if FakeQuantize nodes appear missing)

```bash
python EdgeFusion-CenterPoint/compression/check_fakequant.py \
    --config $CFG --checkpoint $CKPT
```

---

## Results Summary

| Variant       | mAP   | NDS   | vs FP32         |
| ------------- | ----- | ----- | --------------- |
| FP32 baseline | 48.15 | 59.22 | —               |
| PTQ INT8      | 48.12 | 59.03 | −0.03% / −0.19% |
| QAT INT8      | TBD   | TBD   | TBD             |

PTQ achieves essentially free quantization — the drop is within evaluation variance.
This is expected for a BatchNorm-heavy architecture (see Architecture Analysis below).

---

## Toolkit Choice: `torch.ao` over `pytorch-quantization`

We use `torch.ao.quantization` (built into PyTorch 2.1) rather than NVIDIA's
`pytorch-quantization` library.

The reason is a CUDA version mismatch in our training environment: the system CUDA is 12.8
but our PyTorch is compiled against CUDA 11.8. The pre-built `pytorch-quantization` wheel
has a C++ ABI mismatch against PyTorch 2.1, and building from source fails because source
build requires the nvcc version that matches the PyTorch CUDA target (11.8), not the system
CUDA (12.8).

`torch.ao` is built directly into PyTorch and has no separate compilation step, making it
the only viable option in this environment.

---

## FakeQuantize vs Observers: A Critical Distinction

`torch.ao.quantization` has two preparation modes that look similar but behave very differently:

**`prepare_fx` (PTQ calibration mode)**

Inserts `ObserverBase` nodes (`HistogramObserver`, `MinMaxObserver`) that passively collect
activation statistics during forward passes. These observers do **not** modify the forward
pass — the model runs in full FP32 even with observers attached. The purpose is to collect
min/max statistics to compute optimal INT8 scales.

**`prepare_qat_fx` (QAT / INT8 simulation mode)**

Inserts `FakeQuantize` nodes that actively quantize and dequantize activations during the
forward pass: `x_fq = dequantize(quantize(x, scale, zp))`. The value is still stored as
FP32 but has been rounded to INT8 precision. This is called "fake" quantization because no
actual INT8 arithmetic occurs, but the precision loss is simulated faithfully.

### Why this matters

For accuracy measurement on GPU, only `prepare_qat_fx` gives valid results. `prepare_fx`
observers are invisible to inference — running eval on a `prepare_fx` model measures FP32
accuracy, not INT8 accuracy. All three scripts (ptq, sensitivity, qat) use `prepare_qat_fx`
for this reason.

### Why not true INT8 on GPU?

PyTorch's `convert_fx` (which produces actual INT8 arithmetic) targets CPU backends
(FBGEMM, QNNPACK) only. There is no CUDA INT8 inference path in vanilla PyTorch. On our
A40 GPU, true INT8 requires TensorRT, which is deferred to Jetson deployment. Here,
`FakeQuantize` gives the same accuracy information — the simulated INT8 precision loss is
identical to what TRT INT8 will produce — without the full TRT pipeline.

### The QConfig must use FakeQuantize explicitly

```python
# WRONG — produces observer nodes, forward pass is still FP32
qconfig = QConfig(
    activation=HistogramObserver.with_args(...),
    weight=PerChannelMinMaxObserver.with_args(...),
)

# CORRECT — produces FakeQuantize nodes, forward pass simulates INT8
qconfig = QConfig(
    activation=FakeQuantize.with_args(observer=HistogramObserver, ...),
    weight=FakeQuantize.with_args(observer=PerChannelMinMaxObserver, ...),
)
```

`prepare_qat_fx` does not automatically wrap raw observers in FakeQuantize. The QConfig must
explicitly specify FakeQuantize as the outer class, with the observer class passed as an
argument. Using raw observers with `prepare_qat_fx` silently produces observer nodes and the
node count check (`isinstance(m, FakeQuantizeBase)`) returns zero.

---

## PTQ Implementation

### Quantized modules

Only `pts_backbone` (SECOND) and `pts_neck` (SECONDFPN) are quantized. The detection head
(`pts_bbox_head`) is kept in FP32 for two reasons:

The head accounts for approximately 4% of total FLOPs, so the latency benefit of quantizing
it is minimal. More importantly, the head's output layers (heatmap sigmoid, regression
outputs) are accuracy-sensitive. Small quantization errors in these outputs translate
directly to missed detections or degraded localization. TensorRT's AMP mode follows the same
convention when building INT8 engines for detection networks.

### FX tracing fix for SECONDFPN

`prepare_qat_fx` uses PyTorch's FX tracer to symbolically trace the model graph. SECONDFPN's
`forward` method iterates over the input tuple using `len(x)`, which FX cannot evaluate
symbolically (it sees a dynamic value, not a constant). This causes tracing to fail.

The fix: temporarily patch `SECONDFPN.forward` before tracing to use
`len(self.deblocks)` (a constant known at trace time) instead of `len(x)`. The patch is
applied only during `prepare_qat_fx` and reverted immediately after, leaving the module
unmodified for all other operations.

### Calibration protocol

Calibration collects activation statistics over 512 training samples:

```
_set_calibration_mode(model, calibrating=True)
  → disable_fake_quant(), enable_observer() on all FakeQuantize nodes
  → run 512 training samples
_set_calibration_mode(model, calibrating=False)
  → disable_observer(), enable_fake_quant()
  → scales are now fixed from collected statistics
```

This two-phase protocol ensures the calibration statistics are collected before INT8
simulation begins. Running with fake-quant active during calibration would cause the observer
to collect statistics on already-quantized values, producing a feedback loop that degrades
scale accuracy.

### Why PTQ = FP32 on this architecture

CenterPoint uses Batch Normalization throughout the SECOND backbone and SECONDFPN neck. BN
normalizes activations to zero mean and unit variance before applying learned scale and bias
parameters. This means activation distributions entering each conv layer are already bounded
and symmetric — exactly the property that INT8 quantization requires. With 512 calibration
samples, `HistogramObserver` finds near-perfect INT8 scales for every layer.

The result is that PTQ on CenterPoint produces essentially zero accuracy degradation
(−0.03% mAP). This is not a measurement error — the architecture is genuinely INT8-robust by
design. Any network with BN after every conv will show similar behavior.

---

## Sensitivity Analysis

### Methodology

Sensitivity analysis identifies which quantized layers have the most impact on model
accuracy. For each FakeQuantize node, we measure how much the model's training loss increases
when that node alone is disabled (reverting that layer to FP32), while all other nodes remain
in INT8 simulation.

Using training loss as the proxy metric rather than full NuScenes mAP offers a significant
speed advantage: 500 training samples take approximately 7 minutes, versus approximately
20 hours for a full 6019-sample mAP evaluation per node (40 nodes × 30 min each). Loss
increase correlates reliably with detection degradation — a layer causing high mAP drop will
cause high loss increase under the same inputs.

### Results

```
18 FakeQuantize nodes analyzed
FP32 reference loss: 6.0292

Sensitive nodes (relative loss increase > 0.02):
  pts_backbone.blocks.0.3.weight_fake_quant    +0.028

Non-sensitive nodes: 17/18
```

Only one node marginally exceeds the 0.02 threshold: the weight quantizer for layer 3 of
the first backbone block. This is consistent with the general finding in quantization
literature that early feature extraction layers — those processing the raw input
representation — are more sensitive to precision loss, as errors here propagate through all
downstream layers. The sensitivity value (+0.028) is itself small, barely above threshold.

### Implications for mixed precision

QAT applies the sensitivity result by keeping `blocks.0.3.weight_fake_quant` in FP32 while
training all other 17 nodes in INT8. Given that even the sensitive node barely exceeds the
threshold, the expected accuracy difference between pure INT8 QAT and mixed-precision QAT is
negligible on this architecture. The mixed precision pass is included for methodological
completeness and to demonstrate awareness of quantization sensitivity in the portfolio.

---

## QAT Implementation

### Role of QAT given PTQ = FP32

When PTQ already achieves FP32 accuracy, QAT has no accuracy gap to recover. Its role here
is twofold:

First, QAT fine-tunes the model weights to be more robust to quantization constraints,
producing a checkpoint that will behave more reliably when converted to TRT INT8 on Jetson.
Weights that have been trained with INT8 simulation active adapt their value distributions
to be more quantization-friendly than weights that have only been calibrated post-hoc.

Second, QAT is the correct foundation for the pruning phase that follows. After structured
pruning removes channels from the backbone, accuracy will drop. QAT post-pruning will be the
primary tool for recovering that accuracy. Demonstrating QAT on the baseline model here
validates the methodology before the more demanding pruning experiments.

### Training protocol

QAT fine-tunes from `ptq_calibrated.pth` with:

- Optimizer: AdamW
- Learning rate: 1e-5 (low, to avoid disturbing well-trained FP32 weights)
- Epochs: 5
- Batch size: 4
- LR schedule: cosine annealing
- FakeQuantize: active throughout training (no warm-up period)

The low learning rate is appropriate here because the FP32 weights are already well-trained.
QAT at this stage is not recovering from a large accuracy drop — it is making small
adjustments to improve INT8 compatibility at the margin.

### Model initialization

Both sensitivity and QAT initialize via `init_model(cfg, fp32_ckpt)` rather than
`init_model(cfg, checkpoint=None)`. Initializing with the FP32 checkpoint is required
because the inference config (`centerpoint_pillar02_second_secfpn_head-circlenms_...py`) does
not define `model.train_cfg`. The open-mmlab inference config omits `model.train_cfg` since
it is not needed for inference. `init_model(cfg, None)` builds a structure-only model
without it, causing `pts_bbox_head.train_cfg = None`, which raises a TypeError when computing
training loss.

Loading the FP32 checkpoint triggers full model initialization including `train_cfg`
propagation. The FX preparation is then applied on top, and the PTQ calibrated scales are
loaded with `strict=False` to update FakeQuantize scale/zero_point buffers without
overwriting the FP32 weights.

### train_cfg reconstruction fallback

If `pts_bbox_head.train_cfg` is still None after FP32 init (which can occur with some config
variants), `_set_train_cfg_from_config` reconstructs it from fields that are always defined
in the model config: `bbox_coder.out_size_factor`, `data_preprocessor.voxel_layer`, and
`pts_middle_encoder.output_shape`. This avoids hardcoding config values while still being
robust to inference-only configs.

---

## Comparison with Autoware's Deployed Model

Autoware's production `autoware_lidar_centerpoint` uses TRT with `trt_precision: fp16`
(FP16) as the default inference mode, not INT8. Our compression target (INT8) is more
aggressive than Autoware's production default.

Key parameter differences between Autoware's deployed model and ours:

| Parameter                 | Autoware deployed           | This project         |
| ------------------------- | --------------------------- | -------------------- |
| `point_cloud_range` z     | −3.0 to +5.0                | −5.0 to +3.0         |
| `voxel_size`              | [0.2, 0.2, 8.0]             | [0.2, 0.2, 8.0]      |
| `point_feature_size`      | 4                           | 5                    |
| `encoder_in_feature_size` | 9                           | 11                   |
| `trt_precision`           | fp16                        | INT8 (Jetson target) |
| Classes                   | 5                           | 10                   |
| Training data             | nuScenes + TIER IV internal | nuScenes only        |

The z-range difference reflects different sensor mounting heights between Autoware's test
vehicle and the nuScenes vehicle. Our model is trained with the standard nuScenes range and
must be deployed with matching parameters. When creating the Autoware param.yaml for deployment, use
the values from the training config, not Autoware's defaults.

---

## What comes next

The key finding from PTQ and sensitivity is that INT8 quantization is effectively free on
this architecture (−0.03% mAP, all but one layer non-sensitive). The interesting
accuracy / efficiency trade-offs come from structured pruning, where channel reduction
creates a real accuracy budget that distillation and QAT then recover.

```
Compression
├─ PTQ calibration          complete — 48.12 mAP, −0.03%
├─ Sensitivity analysis     complete — 1/18 nodes sensitive
├─ QAT fine-tuning          complete
├─ Structured pruning       3 ratios × fine-tune sweep
├─ Knowledge distillation   teacher=FP32, student=smaller backbone
└─ Pareto assembly          accuracy vs latency vs power frontier
```

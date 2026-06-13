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

- `prepare_qat_fx` (NOT `prepare_fx`) inserts `FakeQuantize` nodes that actively
  simulate INT8 on GPU during the forward pass. `prepare_fx` inserts passive
  `Observer` nodes that collect statistics but don't affect inference — using it
  for accuracy measurement silently evaluates FP32 and was an early mistake
  (see "PTQ accuracy" below).
- The QConfig must specify `FakeQuantize.with_args(observer=...)` explicitly —
  `prepare_qat_fx` does not auto-wrap raw observer classes in FakeQuantize.
- The same prepared model serves both PTQ (calibration) and QAT (fine-tuning).
- Calibrated scales transfer to TRT via Q/DQ ops in the exported ONNX.

## Quantized submodules — backbone + neck only

| Module                       | FLOPs share | Decision                             |
| ---------------------------- | ----------- | ------------------------------------ |
| `pts_backbone` (SECOND)      | ~75%        | INT8 — Conv2d+BN, FX-traceable       |
| `pts_neck` (SECONDFPN)       | ~20%        | INT8 — ConvTranspose2d, FX-traceable |
| `pts_bbox_head` (CenterHead) | ~4%         | FP32 — output layers, sensitive      |
| `pts_voxel_encoder` (PFN)    | <1%         | FP32 — negligible compute            |

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

## PTQ accuracy — near-identical to FP32

PTQ INT8 achieved mAP 48.12 / NDS 59.03 vs FP32 baseline 48.15 / 59.22
(−0.03% / −0.19%) — within measurement noise.

**Measurement history:** an earlier run reported 48.20/59.18 (slightly _above_
FP32, which is impossible for real INT8). This was `prepare_fx` inserting
passive observers instead of active `FakeQuantize` nodes — the model was
evaluated in FP32 the whole time. After switching to `prepare_qat_fx` with an
explicit `FakeQuantize` QConfig (see "Quantization toolkit" above), the
calibration step also needed an explicit two-phase mode switch
(`_set_calibration_mode`: observers ON / fake-quant OFF during calibration,
then observers OFF / fake-quant ON for evaluation) — without this, the
"calibrated" model was still running FP32. The corrected 48.12/59.03 result is
the first run where the sign of the accuracy delta is consistently negative,
confirming FakeQuantize was genuinely active.

Root cause for the near-zero drop: CenterPoint's SECOND backbone applies
BatchNorm after every Conv2d. BatchNorm normalises activations to a consistent,
outlier-free range — exactly the condition under which INT8 histogram
calibration is near-perfect. This is a known property of BatchNorm-heavy
architectures, and the same property extends to the pruned backbones (BN
remains after every conv regardless of channel count) — so TRT INT8 on the
pruned models is also expected to be near-free, without re-running PTQ
separately for each ratio.

Consequence for QAT: QAT on the unpruned model recovers +0.0002 mAP / +0.0007
NDS over PTQ (48.14/59.10) — there is almost nothing to recover. QAT is still
demonstrated on the baseline to validate the pipeline and produce
quantization-friendly fine-tuned weights, with the one marginally-sensitive
layer (`pts_backbone.blocks.0.3.weight_fake_quant`, identified via per-layer
sensitivity analysis) kept in FP16.

## Structured pruning — L1-magnitude, one-shot

`torch-pruning` removes whole output channels from `pts_backbone` (SECOND) and
`pts_neck` (SECONDFPN), selected by L1 weight magnitude. One-shot (single prune

- 5-epoch fine-tune) was chosen over iterative prune-retrain cycles:

| Approach          | Cost (3 ratios) | Expected accuracy                       |
| ----------------- | --------------- | --------------------------------------- |
| One-shot (chosen) | ~25 hrs         | steep falloff past 25%                  |
| Iterative         | ~75-125 hrs     | typically a few points higher per ratio |

Channel pruning compounds across layer pairs — total params scale as
`(1 − ratio)²`: 25% → 56.4%, 40% → 36.0%, 55% → 20.3% of original backbone+neck
params. Iterative pruning is noted as future work if a sub-25% operating point
becomes necessary.

### shared_conv: the pruning/head boundary

`pts_bbox_head.shared_conv` sits between the (prunable) neck output and the
(frozen) task heads, which expect a fixed 64-channel input. torch-pruning
cannot trace through this boundary cleanly:

- Including `shared_conv` in the pruning graph and constraining only its
  output → torch-pruning still propagated pruning into its input shape
  incorrectly (crash: expected 384, got 288).
- Including it without output constraints → over-pruned its output and broke
  the frozen task heads (crash: expected 64, got 48).

**Fix:** prune `pts_backbone` + `pts_neck` only (same SECONDFPN FX patch as
PTQ — `len(self.deblocks)` instead of `len(x)`), then run one forward pass to
detect the pruned neck's actual output width and reinitialize
`shared_conv.conv` with the correct `in_channels` (kaiming-normal). This single
3×3 conv learns quickly during fine-tuning — by end of epoch 1 it's no longer
distinguishable from the rest of the network's contribution to loss.

### EMA + BatchNorm buffer bug

The ratio-25% pruning run completed with a healthy loss curve (10.87 → 7.78 →
7.20 → 6.94 → ~6.4) but evaluated at mAP 0.0026 — a 180x gap between training
signal and eval result.

**Root cause:** `model.load_state_dict(ema_model.state_dict())` was used to
load EMA-smoothed weights before evaluation. `state_dict()` includes both
parameters AND buffers, but the EMA loop only updated `.parameters()` —
BatchNorm's `running_mean`/`running_var` are buffers, frozen in `ema_model` at
`copy.deepcopy()` time (right after pruning, before training). Loading this
state dict overwrote `model`'s properly-trained BN statistics with these stale,
pre-training values. `model.eval()` uses `running_mean`/`running_var` for
normalization — with these wrong, every activation is mis-normalized
(symptom: `mATE`/`mASE`/`mAOE` saturate at their 1.0 clip values, AP≈0 across
all classes).

**Fix:** `_merged_ema_state_dict` takes parameters from `ema_model` (EMA,
correct) and buffers from `model` (live, properly-trained). Applied to all
subsequent runs (ratios 40%, 55%, and distillation) with no recurrence.

**Recovery for ratio 25% — BN recalibration, not retrain:** rebuild the pruned
architecture (deterministic L1 selection given the same FP32 weights + ratio),
load the _trained_ EMA parameters from the broken checkpoint (these were
correct — only buffers were stale), reset all BatchNorm running stats, and run
~200 forward-only batches in `train()` mode with cumulative averaging
(`momentum=None`) to recompute `running_mean`/`running_var` against the trained
weights. Standard "BN re-estimation" — ~20 min vs an 8.5 hr retrain. Recovered
mAP 0.4081.

### Dataset — raw, not CBGS, for fine-tuning

CBGS class-balanced sampling inflates the dataset ~4.4x (1,759 → 7,724
steps/epoch at batch=16), making the 3-ratio sweep cost ~112 hrs instead of
~25 hrs. Fine-tuning starts from FP32 weights that already encode 20 epochs of
CBGS-trained knowledge (validated independently by QAT: raw dataset, 0.4814
mAP matching published numbers within noise) — so the 4.4x time saving was
taken as a documented trade-off. Cost: rare-class recovery (bicycle,
construction_vehicle) is weaker than it would be with CBGS, visible in the
pruning sweep's per-class results — bicycle AP drops from 0.154 (FP32) to
0.000 at 55% pruning.

### Pruning sweep results

| Ratio     | Params | mAP   | NDS   | Δ mAP  |
| --------- | ------ | ----- | ----- | ------ |
| 0% (FP32) | 100%   | 48.15 | 59.22 | —      |
| 25%       | 56.4%  | 40.81 | 53.82 | −15.2% |
| 40%       | 36.0%  | 28.38 | 39.02 | −41.0% |
| 55%       | 20.3%  | 21.49 | 31.36 | −55.4% |

The accuracy cost accelerates sharply past 25% — 40% pruning loses more than
2.5x the relative mAP of 25% for ~20pp fewer params. Velocity estimation
(mAVE) degrades worst: 0.350 (FP32) → 1.053 (55%), with bus/motorcycle
exceeding 2.0 — reduced backbone capacity struggles most with motion cues.
25% pruning is the only ratio with a defensible accuracy/size tradeoff; 40%
and 55% establish the curve shape and the point where one-shot pruning hits
its capacity ceiling.

## Knowledge distillation — controlled comparison, not a new architecture

Rather than designing a novel "tiny" student (which would need ~20 epochs from
random init to converge), the distillation student uses the **identical
architecture, init, epoch budget, and dataset as Pruned 25%** — same L1
channel selection from FP32, 5 epochs, raw dataset, batch=16. The only
variable is the loss function:

```
Pruned 25%:  L_task only                              → 0.4081 mAP, 0.5382 NDS
Distilled:   L_task + alpha·L_heatmap + beta·L_reg    → 0.4094 mAP, 0.5344 NDS
```

This isolates whether teacher guidance recovers pruning-induced capacity loss
better than fine-tuning alone — a direct A/B, not confounded by architecture
or compute differences.

Loss design:

- `L_heatmap_distill`: dense MSE between teacher/student sigmoid heatmaps —
  "dark knowledge" on object confidence including rare classes, targeting the
  bicycle/construction_vehicle collapse seen under pruning.
- `L_reg_distill`: L1 on reg/height/dim/rot/vel, weighted by the teacher's
  heatmap confidence (max over classes) as a soft attention mask — focuses
  regression distillation where the teacher believes an object exists, without
  needing GT-based positive/negative masks. Targets the mAVE collapse.

No feature adapter is needed: pruning only touches backbone/neck, and
`shared_conv` is rebuilt to the same 64-channel output feeding the (untouched)
task heads — teacher and student produce identically-shaped outputs.

**`extract_feat` signature and return value:** `MVXTwoStageDetector.extract_feat`
requires `batch_input_metas` as a second positional argument
(`extract_feat(batch_inputs_dict, batch_input_metas)`), constructed as
`[ds.metainfo for ds in data_samples]`. Calling it with only
`batch_inputs_dict` raises `TypeError: missing 1 required positional argument`.

It returns a 2-tuple `(img_feats, pts_feats)` — even for a LiDAR-only model,
where `img_feats=None`. Passing the raw return value directly to
`pts_bbox_head(...)` makes `multi_apply` iterate over `(None, pts_feats)`,
calling `forward_single(None)` first → `shared_conv(None)` →
`TypeError: conv2d() received ... NoneType`. Unpack and use only `pts_feats`:
`_, pts_feats = model.extract_feat(inputs, batch_input_metas)`.

### Loss weight calibration

At `alpha=beta=1.0` (initial attempt), `hm=0.0011` and `reg=0.030` against
`task≈25-29` — distillation contributed <0.2% of total loss, effectively a
no-op. Raised to `alpha=2000, beta=50`, giving `alpha*hm≈1.8` (7% of task) and
`beta*reg≈1.25` (5% of task) — a meaningful but non-dominant auxiliary signal
(~12% combined), the range run reported below.

### Results: no improvement over task-loss fine-tuning

| Metric | Pruned 25% (task-only) | Distilled (task+distill) | Δ                |
| ------ | ---------------------- | ------------------------ | ---------------- |
| mAP    | 0.4081                 | 0.4094                   | +0.0013 (+0.32%) |
| NDS    | 0.5382                 | 0.5344                   | −0.0038 (−0.71%) |
| mATE   | 0.3270                 | 0.3551                   | worse            |
| mASE   | 0.2631                 | 0.2705                   | worse            |
| mAOE   | 0.3664                 | 0.4541                   | worse (+0.088)   |
| mAVE   | 0.3442                 | 0.4344                   | worse (+0.090)   |
| mAAE   | 0.1961                 | 0.1884                   | better           |

Per-class deltas mostly noise-level (trailer +0.016, truck +0.010, motorcycle
−0.007). The two classes `L_heatmap_distill` specifically targeted — bicycle,
construction_vehicle — show **no change** (0.034→0.035, 0.059→0.059).

### Root cause: loss formulation, not loss magnitude

**`L_heatmap_distill` is background-dominated.** Dense sigmoid-MSE over a
512×512 heatmap is overwhelmingly background, where teacher and student
already agree (both near-0). Rare-class foreground pixels are a tiny fraction
of the total MSE. Scaling `alpha` amplifies the whole (background-dominated)
loss uniformly — it doesn't selectively target rare-class signal. This is the
same imbalance problem focal loss solves for the task loss; our distillation
loss has no equivalent. A foreground-weighted or per-class heatmap distillation
term is the fix — not a larger `alpha`.

**`L_reg_distill` likely suffers teacher/student spatial misalignment.** The
confidence weighting uses the _teacher's_ heatmap peaks, but the student's
`shared_conv` was reinitialized (random kaiming init) — its spatial confidence
pattern can differ from the teacher's even post-training. If
teacher-confident locations don't align with GT-assigned locations,
`L_reg_distill` and `L_task`'s regression terms pull in different directions
for the same cells, plausibly explaining why mAVE/mAOE got _worse_ rather than
moving toward the teacher's better values (mAVE=0.350). A GT-location-based
regression distillation (matching teacher outputs at GT-assigned cells) would
avoid this.

Given both root causes are about _formulation_, further alpha/beta tuning is
unlikely to produce a clear win — noted as "distillation v2" future work
(foreground-weighted heatmap term, GT-aligned regression term).

### Conclusion

At these settings, output-level distillation does not outperform task-loss-only
fine-tuning for 25% pruning — roughly equivalent on mAP, worse on NDS/box
quality. **Pruned 25% (task-loss-only) remains the practical choice.**
Distilled technically edges Pruned 25% on mAP alone (+0.0013) on the
architecture Pareto chart, but given NDS is worse, the two should be read as
the same operating point reached by different recipes — not a real
improvement.

Checked the epoch3 checkpoint (mAP 0.3758, NDS 0.5086) for an earlier-epoch
sweet spot before reg-distillation's effect compounded — substantially worse
than epoch5 (−0.0336 mAP, −0.0258 NDS). Epoch5 is the representative best
checkpoint; no early-stopping benefit found.

All compression runs are tracked in a local MLflow server for reproducibility
and Pareto construction. Run naming convention is in `mlflow/README.md`.

---

_Updated as decisions are made._

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

**Measurement history:** an earlier run reported 48.20/59.18 (slightly *above*
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
+ 5-epoch fine-tune) was chosen over iterative prune-retrain cycles:

| Approach | Cost (3 ratios) | Expected accuracy |
|---|---|---|
| One-shot (chosen) | ~25 hrs | steep falloff past 25% |
| Iterative | ~75-125 hrs | typically a few points higher per ratio |

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
load the *trained* EMA parameters from the broken checkpoint (these were
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

| Ratio | Params | mAP | NDS | Δ mAP |
|---|---|---|---|---|
| 0% (FP32) | 100% | 48.15 | 59.22 | — |
| 25% | 56.4% | 40.81 | 53.82 | −15.2% |
| 40% | 36.0% | 28.38 | 39.02 | −41.0% |
| 55% | 20.3% | 21.49 | 31.36 | −55.4% |

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

| Metric | Pruned 25% (task-only) | Distilled (task+distill) | Δ |
|---|---|---|---|
| mAP | 0.4081 | 0.4094 | +0.0013 (+0.32%) |
| NDS | 0.5382 | 0.5344 | −0.0038 (−0.71%) |
| mATE | 0.3270 | 0.3551 | worse |
| mASE | 0.2631 | 0.2705 | worse |
| mAOE | 0.3664 | 0.4541 | worse (+0.088) |
| mAVE | 0.3442 | 0.4344 | worse (+0.090) |
| mAAE | 0.1961 | 0.1884 | better |

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
confidence weighting uses the *teacher's* heatmap peaks, but the student's
`shared_conv` was reinitialized (random kaiming init) — its spatial confidence
pattern can differ from the teacher's even post-training. If
teacher-confident locations don't align with GT-assigned locations,
`L_reg_distill` and `L_task`'s regression terms pull in different directions
for the same cells, plausibly explaining why mAVE/mAOE got *worse* rather than
moving toward the teacher's better values (mAVE=0.350). A GT-location-based
regression distillation (matching teacher outputs at GT-assigned cells) would
avoid this.

Given both root causes are about *formulation*, further alpha/beta tuning is
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

## Deployment decision: QAT INT8 (full architecture) on 25W

**Final recommendation: deploy the full FP32 architecture, INT8-quantized via QAT +
explicit quantization, on Jetson Orin Nano's 25W power mode (13.29 MB engine,
25.70ms, 19.87W VDD_IN, on-device mAP 0.4265).** Two independent conclusions combine
into this: (a) *pruning is the wrong compression axis* (this section), and (b) *of the
INT8 production methods, only QAT + explicit quantization recovers FP32-level accuracy
on TensorRT* (see "The INT8 deployment finding" below). Latency/power/size are real
Jetson measurements on the 25W mode with `jetson_clocks` locked; on-device mAP/NDS are
on the 512-sample subset.

Jetson benchmarks (TRT 10.3.0, 25W) showed structured pruning does not translate
to proportional inference speedup on this hardware. Latency/power are PTQ-INT8 engines
(the architecture-comparison axis); on-device mAP reflects PTQ on plain fine-tuned
weights and so understates the pruned variants' true capability (none were QAT-trained):

| Variant | params% | on-device mAP | p50 latency | FPS | VDD_IN |
|---|---|---|---|---|---|
| FP32 → TRT INT8 (PTQ) | 100% | 0.3612 | 8.09ms | 123.5 | 16.60W |
| Pruned 25% → TRT INT8 (PTQ) | 56.4% | 0.2637 | 7.46ms | 134.1 | 16.41W |
| Pruned 40% → TRT INT8 (PTQ) | 36.0% | 0.1176 | 8.17ms | 122.3 | 15.81W |
| Pruned 55% → TRT INT8 (PTQ) | 20.3% | 0.1556 | 6.93ms | 144.3 | 15.61W |
| **QAT INT8 (deployed)** | 100% | **0.4265** | 25.70ms | 38.9 | 19.87W |

### Why pruning doesn't help on Jetson Orin Nano

**Tensor core alignment.** L1 magnitude pruning removes a fixed ratio of channels
per layer, yielding non-standard counts (e.g. 75% of 128 = 96 for Pruned 25%,
60% of 128 = ~77 for Pruned 40%). TRT pads channel dimensions to the nearest
multiple of 16 for INT8 tensor core dispatch. Pruned 40% falls into a particularly
bad alignment zone — it is empirically *slower* than the FP32 INT8 engine (8.17ms vs
8.09ms) despite having 64% fewer parameters.

**Memory bandwidth bound.** The 512×512 BEV spatial dimensions are the true latency
bottleneck on Jetson's unified memory, not parameter count. These dimensions don't
change with channel pruning. Halving channel depth reduces memory traffic for those
layers, but fixed-overhead layers (first conv, SECONDFPN upsample + concat, task
heads) impose a constant floor.

**Unpruned task heads.** All 6 CenterHead task heads (~1.1–1.5M params) are
excluded from pruning scope (pruning was applied to backbone+neck+shared_conv only,
which is the safe boundary for L1 magnitude pruning — task heads use shared_conv
features and reinitializing them would lose all regression training). They contribute
a fixed latency regardless of backbone size.

### Why the FP32 architecture is the right choice

INT8 quantization is near-free for this architecture (A40 PTQ −0.03% / QAT −0.01% mAP)
because BatchNorm normalises activations before every conv, producing
INT8-friendly distributions throughout. Crucially, *realizing* that near-free property
on the deployed TensorRT engine required QAT-trained weights with explicit quantization
— post-training calibration of plain weights left a real gap (see the INT8 deployment
finding below). There is no budget pressure that would justify the large mAP cost of
pruning for a latency change of under a millisecond. The compression story for this
project is: **quantization, not pruning — and QAT-trained quantization specifically.**

### What pruning research established

The pruning sweep was not wasted work — it established *why* pruning doesn't help
here (tensor core alignment, memory-bandwidth bottleneck) rather than simply
asserting it. This is the correct outcome of a controlled experiment. The Pareto
chart (architecture and deployment views) documents the full accuracy–efficiency
frontier, making the quantization-over-pruning recommendation defensible rather than
assumed.

*Updated: Final Jetson benchmark results; superseded headline by the QAT
deployment finding below.*

## On-device mAP validation — methodology, debugging saga, and findings

The latency/power/size numbers above were measured directly on Jetson hardware
and required no further validation. On-device **accuracy** (mAP/NDS of the
deployed TRT INT8 engine) was a separate effort, and by far the most
debugging-intensive part of the whole project — multiple days, 20+ build/eval
cycles, across a chain of distinct bugs that each looked plausible until ruled
out. The full chain, in the order discovered:

1. **Coordinate frame.** The standalone decode emitted predictions in LiDAR/BEV
   frame; nuScenes' official eval requires global (UTM-like) frame. Fixed via
   per-sample `lidar2ego` + `ego2global` 4×4 transforms from the val pkl.
2. **Quaternion orthogonality.** pkl float drift on rotation matrices tripped
   `pyquaternion`'s strict orthogonality check. Fixed via SVD re-orthogonalization
   (`R = U·Vᵗ`, the nearest true orthogonal matrix to a drifted one) — verified
   numerically to introduce ~3e-8 rad of angular error on realistic drift.
3. **Box size convention.** The decode emitted `[length, width, height]`;
   nuScenes requires `[width, length, height]`. A silent transpose that
   corrupted every box's footprint while leaving box *centers* correct —
   which is why early debugging (checking center-distance matches) didn't
   catch it.
4. **Mini-NuScenes GT loading.** To keep host-side evaluation memory-efficient
   (~100-200MB vs 2-4GB for the full annotation set), a filtered "mini"
   NuScenes is built containing only the 512 tokens under evaluation.
   `load_gt()` follows `sample_annotation` `prev`/`next` links (for velocity
   computation) that can point to annotations outside the 512-token subset,
   raising `KeyError` mid-load and silently producing zero ground truth.
   Fixed by severing any `prev`/`next` link pointing outside the subset before
   loading (`box_velocity` then returns `[nan, nan]`, its documented behavior
   at sequence boundaries — handled gracefully by the evaluator).
5. **The precision floor.** With frame, rotation, and box-size bugs fixed, mAP
   was still exactly 0.0000. Direct use of nuScenes' own `accumulate()`
   function (rather than the wrapped `DetectionEval.evaluate()`) revealed why:
   **max_recall = 1.000, max_precision = 0.018** — every ground-truth object
   was being detected, but each was accompanied by ~50 duplicate/false-positive
   boxes. `calc_ap` clips the precision-recall curve below `min_precision=0.1`
   and returns exactly 0 when max precision falls under that floor — which is
   why the result was identically zero rather than merely low. This reframed
   the problem from "nothing matches" to "too many false positives," which
   pointed at missing peak suppression: CenterPoint requires keeping only
   local-maximum cells in the heatmap before decoding (a 3×3 max-pool NMS);
   the standalone decode had been thresholding every cell above 0.1
   independently, decoding entire heatmap blobs as separate boxes.
6. **Peak suppression alone wasn't enough — heatmap confidence was uniformly
   weak.** Even after adding 3×3 max-pool peak extraction, predicted box
   counts only dropped modestly (190k → 177k across 512 samples) and mAP
   stayed at 0. Inspecting raw heatmap activations showed car peaking at
   sigmoid ≈0.31 and bicycle at ≈0.79 — backwards from a realistic scene, and
   far below the 0.8-0.99 a healthy CenterPoint head produces for confident
   detections.
7. **INT8 quantization was suspected and ruled out.** Running the FP32 ONNX
   (CPU, zero quantization) through the identical decode on the identical
   input produced the *same* flat heatmap profile as the INT8 engine — proof
   the problem was upstream of TRT entirely.
8. **Root cause: missing multi-sweep LiDAR aggregation.** The model's config
   pipeline specifies `LoadPointsFromMultiSweeps(sweeps_num=9)` — CenterPoint
   expects ~270-290k aggregated points per frame (current frame + 9
   motion-compensated past sweeps), not a single raw LiDAR scan (~25-35k
   points). Every BEV calibration tensor generated up to this point had used
   single-sweep point clouds — the detector was never broken; it was starved
   of input density. Confirmed by regenerating calibration BEV via a forward
   hook on `pts_middle_encoder` during the model's own real forward pass (fed
   by the actual multi-sweep test pipeline built straight from the model
   config, rather than a hand-reconstructed voxelize→encode→scatter path):
   point counts jumped to ~278k/sample and heatmap peaks to 0.85-0.93.

**The meta-lesson**: the symptom (mAP = 0) pointed at the evaluation harness,
but the cause was three layers further upstream, in calibration-data
preprocessing fidelity. The turning point was using nuScenes' own `accumulate()`
directly rather than treating the eval framework as a black box — that one
diagnostic separated "is detection working?" (recall) from "is detection
usable?" (precision), which made the false-positive flood, and from there the
missing peak suppression and the input density bug, traceable in sequence.

### The INT8 deployment finding: PTQ calibration vs QAT + explicit quantization

Once correct multi-sweep BEV calibration data was confirmed, a real (and separate)
finding emerged about *how* INT8 is produced for TensorRT. The A40 compression sweep
had shown INT8 to be near-free under PyTorch's FakeQuantize/observer-based calibration
(PTQ −0.03%, QAT −0.01% mAP). But the **deployed** TensorRT engine, calibrated from
scratch on the plain FP32 weights, did not reproduce that — and the gap was traced
through three distinct calibration approaches, all evaluated on the same 512-sample
subset with the same standalone decode (so only the quantization path differs):

```
ONNX FP32 (zero quantization, reference ceiling):  mAP 0.432    NDS 0.485
TRT INT8, IInt8EntropyCalibrator2 (plain weights): mAP 0.30     NDS 0.39
TRT INT8, IInt8MinMaxCalibrator   (plain weights): mAP 0.3612   NDS 0.4304
TRT INT8, QAT weights + explicit Q/DQ:             mAP 0.4265   NDS 0.4804
```

The progression and what each step established:

1. **Entropy calibration (TRT default) was the worst** — 0.30 mAP, a ~30% relative
   drop from the FP32 ceiling. TRT's `IInt8EntropyCalibrator2` picks per-tensor
   quantization ranges by minimizing histogram KL-divergence, which can clip rare,
   high-magnitude activations (a classification head needs logits up to +2..+5 for a
   confident peak — exactly the low-probability-density tail entropy calibration is
   willing to sacrifice for a tighter range on the bulk distribution).

2. **MinMax calibration recovered about half the gap** — 0.3612 mAP. It uses the literal
   observed per-tensor min/max with no distributional assumption, so it doesn't clip
   those rare extremes. This confirmed the calibration *algorithm* was part of the
   problem, but a real gap remained.

3. **Forcing the classification head to FP16 did essentially nothing** (0.3612 →
   0.3613). This was an informative negative result: it ruled out the hypothesis that
   the loss was concentrated in the head, and showed the residual PTQ loss is
   distributed across the INT8-quantized backbone/neck features themselves (which feed
   every head — box position, size, orientation, not just classification confidence).
   Note this FP16-head marking required `BuilderFlag.OBEY_PRECISION_CONSTRAINTS` to
   take effect at all — without it, TRT's optimizer silently ignored the per-layer
   precision hint (identical predictions until the flag was added).

4. **A controlled overlap check ruled out a same-data artifact.** Because the 512
   calibration samples doubled as the eval set, a natural worry was that this inflated
   INT8's measured accuracy. Splitting into a disjoint 256-calibration / 256-held-out
   eval gave the same per-class pattern (and the theory says same-data is the
   *favorable* case for INT8, so a real held-out eval can only be equal or worse) —
   confirming the gap is architectural, not a measurement artifact.

5. **QAT-trained weights + explicit quantization recovered the gap** — 0.4265 mAP,
   sitting essentially at the FP32 ceiling of 0.432 (−1.3% relative). This is the A40
   "near-free INT8" result finally reproduced in a deployed TensorRT engine. The
   mechanism: QAT weights were trained with simulated quantization noise in the loop,
   so they tolerate the precision reduction; exporting them via `torch.ao.quantization`
   produces ONNX with explicit `QuantizeLinear`/`DequantizeLinear` nodes carrying the
   learned scales, and TRT's *explicit-quantization* path uses those directly rather
   than calibrating its own.

**The catch — QAT's explicit quantization is 3× slower on TensorRT.** The same Q/DQ
nodes that carry the learned scales also **block TRT's Conv+BN+ReLU layer fusion**: in
explicit-quantization mode TRT honors the Q/DQ graph structure, so each quantized layer
becomes a separate kernel launch with dtype conversions instead of fusing into one INT8
kernel. Measured on-device: QAT INT8 runs at 25.70ms / 19.87W VDD_IN versus PTQ INT8's
8.09ms / 16.60W. The build log confirms the path switch (`Calibrator won't be used in
explicit quantization mode`). So the final tradeoff is explicit:

| | PTQ INT8 (minmax) | QAT INT8 (explicit Q/DQ) |
| --- | --- | --- |
| on-device mAP | 0.3612 | **0.4265** |
| latency | 8.09ms | 25.70ms |
| VDD_IN | 16.60W | 19.87W |
| engine | 6.82 MB | 13.29 MB |

**Decision: deploy QAT INT8 on the 25W power mode.** The accuracy recovery
(0.3612 → 0.4265) is worth the latency cost because 25.70ms still fits the 10–20Hz
LiDAR real-time budget (50–100ms/frame) with wide margin, and 19.87W fits the 25W
envelope — the same power mode used for the ROS2 camera/LiDAR/fusion visualization in
the companion EdgeDrive-Perception project.

**Future work to make QAT INT8 *also* fast**: `torch.ao.quantization` is PyTorch's
general-purpose QAT framework and is not TensorRT-fusion-aware. NVIDIA's
`pytorch-quantization` toolkit places Q/DQ nodes specifically where TRT's compiler can
fuse through them, which should retain QAT accuracy while restoring PTQ-level latency.
Re-running QAT with that toolkit is the clear next step for a production deployment.

### Evaluation methodology: standalone decode, not mmdet3d's PyTorch postprocessing

One deliberate decision made during this investigation: the ONNX-FP32 control
above uses the project's own standalone numpy decode (`deployment/scripts/eval.py
decode_outputs`), not mmdet3d's official PyTorch `CenterHead.predict_by_feat`.
This was considered and rejected for *evaluating the deployed system specifically*
(it remains useful as an internal diagnostic — see step 7 above). The reasoning:

- The deployed Jetson pipeline is TensorRT + pycuda + numpy by design — no
  PyTorch, no mmdet3d. That's the point of compiling to a TRT engine: a
  minimal-footprint embedded inference path, which is also what the latency/
  power numbers above measure.
- Routing the TRT engine's raw head outputs through mmdet3d's PyTorch decode
  (on the pod, where mmdet3d is available) would answer "what could this
  engine's outputs achieve under more optimal postprocessing" — a valid but
  different question from "what does the actually-deployed system achieve."
  For a deployment project, the second question is the relevant one: it's the
  number that reflects what ships.
- Practical effect: the standalone decode's own imprecision (see ONNX-FP32 vs
  A40 full-val gap below) is *part of* the deployed system's accuracy, not a
  measurement artifact to be optimized away by switching evaluation backends.

### Known limitations of the 512-sample on-device evaluation

The on-device mAP/NDS figures are not directly comparable to the A40's
full-val (6019-sample) numbers, for two compounding, both-expected reasons,
isolated via the ONNX-FP32 control above:

1. **Sample size.** `jetson_calib`'s 512 tokens were sized for TRT INT8
   calibration (typically 100-500 samples is sufficient), not statistically
   rigorous evaluation. nuScenes mAP averages equally across 10 classes
   regardless of instance count; rare classes in this subset have very few
   ground-truth boxes (bus: 173, construction_vehicle: 139, motorcycle: 86),
   making their per-class AP — and therefore the mean — noisy.
2. **Decode implementation gap.** The standalone numpy decode is a faithful
   but not byte-identical reimplementation of mmdet3d's tuned official
   postprocessing (NMS radius choices, score handling, top-K selection).
3. **TRT INT8 calibration choice** — covered above, and isolated separately
   from (1) and (2) via the ONNX-FP32 vs TRT-INT8 comparison on identical data.

Combined, (1) and (2) account for the ONNX-FP32-vs-A40 gap (0.4815 → 0.432,
~10% relative on mAP) and are not evidence of any deployment defect. Scaling
to a larger on-device evaluation subset (1000-2000 samples, using the full
nuScenes trainval data now available from the multi-sweep BEV regeneration
effort) would reduce (1) without further engineering work, if a tighter
number is wanted later; this was deprioritized once the deployed QAT INT8
engine was shown to sit at the FP32 ceiling (0.4265 vs 0.432), which is the
result that actually mattered.

*Updated: Multi-sweep BEV root cause found and fixed;
INT8 deployment resolved (entropy → minmax → QAT + explicit quantization);
QAT INT8 on 25W selected as the deployable configuration.*

## CUDA center-head postprocessing and end-to-end pipeline

The engine-only deployment left the postprocessing in Python/numpy. The
TRT engine produces raw tensors; the original eval pipeline copied them to CPU and ran
sigmoid, 3×3 max-pool NMS, box decode, and circle NMS in numpy. We wrote the
hot path as a custom CUDA stage and measured the complete end-to-end pipeline.

### Design: post-engine CUDA stage, not a TRT plugin

Two options were considered: wrapping the postprocessing as a TRT plugin (`IPluginV2DynamicExt`)
that runs inside the engine, or as a post-engine CUDA stage that queues kernels on the
same stream after `execute_async_v3`. The standalone stage was chosen because:

- Simpler implementation (no plugin serialization/registration machinery)
- Same functional outcome: all GPU work on one stream, no host↔device copies between
  engine and postproc
- Equal zero-copy benefit (the output buffers from the engine feed directly into the
  CUDA kernels as pointers on the same stream)
- Easier to unit-test independently (the parity test suite runs the kernels without
  any TRT engine)

### The managed-memory segfault (implementation finding)

`pycuda.driver.managed_zeros` (unified memory, CPU+GPU accessible) was the initial
approach for postprocessor output buffers, targeting zero-copy result reads on Jetson's
UMA. It worked in the parity test suite (no TRT engine present) but segfaulted at
`_counts_buf[0] = 0` when combined with a live TRT engine.

Root cause: TRT's `deserialize_cuda_engine` pushes/pops the CUDA context stack. Managed
memory allocated *after* this lands on a different context state than the one pycuda's
autoinit established, causing the CPU access to be invalid. The parity tests avoided
this because the postprocessor was created at module import time, before any engine
existed.

Fix: use plain `cuda.mem_alloc` (device memory) for all GPU output buffers. After
`stream.synchronize()`, copy results with `cuda.memcpy_dtoh` — the total copy is
~170 KB (peaks + scores + counts + boxes at 3000 max detections), which takes
microseconds and is negligible. All buffers remain pre-allocated from `__init__`; the
hot loop does zero Python-side allocations.

### The NMS bottleneck (optimization finding)

First end-to-end benchmark showed a **12.61ms CPU tail** (dtoh + circle NMS) out of a
42.33ms total pipeline, despite the decode work having moved to GPU. The GPU portion
was 29.37ms; the CPU was 30% of the total latency and the dominant source of jitter.

Root cause: circle NMS runs on all *pre-NMS* peak candidates, not just the final
post-NMS detections. The QAT engine produces many heatmap cells above the 0.1
score threshold (the 512×512 grid with a 0.1 sigmoid floor allows many to survive).
The Python NMS loop with a O(N²) per-class structure ran on ~100-200 boxes per class
× 10 classes. Additionally, `pyquaternion.Quaternion()` was being instantiated for each
of the final ~248 detections per frame (~0.03ms/object, ~7ms total).

Fix 1 — **vectorized NMS**: precompute the full N×N squared-distance matrix once
(one numpy broadcast operation: `(xy[:, None, :] - xy[None, :, :]) ** 2).sum(2)`),
then the suppression loop does only boolean indexing on the precomputed matrix. For
N~150 per class, this eliminates the per-pair sqrt and the O(N²) numpy-per-pair
overhead.

Fix 2 — **direct quaternion**: replaced `Quaternion(axis=[0,0,1], angle=yaw)` with
direct `w = cos(yaw/2)`, `z = sin(yaw/2)` numpy calls. pyquaternion's constructor
performs matrix operations; 248 instantiations per frame is measurable overhead.

Result: CPU tail **12.61ms → 4.88ms (-61%)**.

### Final end-to-end pipeline measurements (200 iterations, 25W MAXN)

```
Stage                              p50       p99      jitter
────────────────────────────────────────────────────────────
TRT engine (engine-only)          25.70ms   28.08ms   2.38ms
+ BEV htod + CUDA kernels         29.37ms   38.76ms   9.39ms
+ dtoh + circle NMS (CPU)         +4.88ms    —         —
────────────────────────────────────────────────────────────
Total pipeline                    34.12ms   43.93ms   9.82ms

FPS: 28.7    VDD_IN: 16.29W    mJ/frame: 566.78
```

Real-time assessment: p50=34.12ms gives 15.9ms headroom to the 20Hz (50ms) budget.
p99=43.93ms also clears 20Hz. At 10Hz (100ms) there is 65ms of headroom.

The remaining 9.82ms jitter is almost entirely GPU-side (9.39ms). Its source is
TRT's explicit-quantization path producing variable-length kernel execution sequences
per frame — PTQ INT8 on the same engine architecture shows ~0.02ms GPU jitter. The fix
(future work) is `pytorch-quantization`'s TRT-fusion-aware Q/DQ placement, which would
let TRT fuse through Q/DQ nodes and restore the fast, low-jitter INT8 path.

*Updated: Jetson deployment complete including CUDA postprocessing,
NMS optimization, and end-to-end pipeline measurement.*
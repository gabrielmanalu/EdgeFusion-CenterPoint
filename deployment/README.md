# Deployment — CenterPoint on Jetson Orin Nano (TensorRT)

End-to-end deployment of the compressed CenterPoint LiDAR detector to a Jetson Orin
Nano Super (8GB), running TensorRT 10.3.0 on JetPack R36.4.0. This directory contains
the Docker environment, engine-build / benchmark / eval scripts, and the record of
what actually had to be solved to get a correct, fast, accurate engine on-device.

The short version of the outcome: the deployable configuration is **QAT INT8, exported
with explicit quantize/dequantize nodes, run on the 25W power mode** — on-device
mAP 0.4265 (essentially the FP32 ceiling for this evaluation setup), 25.70ms/frame,
19.87W. The longer version — why it isn't the obvious "PTQ INT8 on 15W" — is the point
of this document.

---

## TL;DR — final result

| | Value |
|---|---|
| Deployed config | QAT INT8 + explicit Q/DQ, 25W |
| On-device mAP / NDS (512-sample) | 0.4265 / 0.4804 |
| Latency (p50 / p99) | 25.70ms / 28.08ms |
| Power (VDD_IN total module) | 19.87W |
| Engine size (backbone+neck+head) | 13.29 MB |
| Real-time headroom | fits 10–20Hz LiDAR (50–100ms budget) |

Full per-variant numbers and the accuracy methodology are in
[`../docs/design_decisions.md`](../docs/design_decisions.md) → "On-device mAP
validation". This README focuses on the deployment *process*.

---

## Directory layout

```
deployment/
├── docker/
│   ├── Dockerfile            # l4t-jetpack:r36.4.0 base, TRT 10.3.0 already present
│   └── docker-compose.yml    # build-engines / benchmark / infer / eval / dev services
├── scripts/
│   ├── build_engine.py       # ONNX → TRT engine (PTQ entropy/minmax, QAT explicit-Q, FP16)
│   ├── benchmark.py          # CUDA-event latency + tegrastats power
│   ├── eval.py               # in-container: engine → decode → submission JSON (--no-eval)
│   ├── eval_metrics.py       # on-host: submission JSON → nuscenes-devkit mAP/NDS
│   └── eval_all.sh           # batch all variants
└── README.md                 # this file
```

The split between `eval.py` (in-container, produces a submission JSON) and
`eval_metrics.py` (on-host, computes mAP/NDS) exists because the lean deployment
container deliberately has no PyTorch/mmdet3d — see "Why the deployment container is
lean" below.

---

## Quick start

Prerequisites: the engine inputs must be present on the Jetson — the exported ONNX
(`compression/results/onnx_export/{variant}/`) and the multi-sweep BEV calibration
tensors (`jetson_calib_bev/`, 512 × `[64,512,512]` `.npy`). See "The calibration data
is not raw LiDAR" below for what these are and why they're large.

```bash
# 0. Lock GPU clocks — REQUIRED before any benchmarking, on the HOST.
#    Without this the GPU idles at a low clock and latency is not reproducible.
sudo jetson_clocks

# 1. Build the engine. For the deployed QAT config (ONNX already carries Q/DQ):
docker compose -f deployment/docker/docker-compose.yml run --rm \
    --entrypoint python3 build-engines \
    scripts/build_engine.py \
    --onnx-dir /workspace/onnx --calib-dir /workspace/calib_bev \
    --out /workspace/output/engines --variant qat_best

#    For a PTQ INT8 variant (calibrated from plain FP32 weights):
docker compose -f deployment/docker/docker-compose.yml run --rm \
    --entrypoint python3 build-engines \
    scripts/build_engine.py \
    --onnx-dir /workspace/onnx --calib-dir /workspace/calib_bev \
    --out /workspace/output/engines --variant fp32 --calibrator minmax

# 2. Benchmark latency + power (uses the 25W mode if set on the host).
VARIANT=qat_best docker compose -f deployment/docker/docker-compose.yml run --rm benchmark

# 3. Generate a submission JSON (in-container, no mAP computed here):
VARIANT=qat_best docker compose -f deployment/docker/docker-compose.yml run --rm infer

# 4. Compute mAP/NDS on the HOST (needs nuScenes annotation JSONs + val pkl):
python3 deployment/scripts/eval_metrics.py \
    --nuscenes ~/Downloads/v1.0-trainval_meta \
    --val-pkl ~/Downloads/runpod/nuscenes_infos_val.pkl \
    --submissions deployment/output \
    --variants qat_best \
    --out deployment/output/eval_qat_best.json
```

> **`--entrypoint python3` on builds is deliberate.** The image's default entrypoint
> intercepts a plain `... run build-engines python3 scripts/...` and can silently
> no-op or double-wrap the command. Overriding the entrypoint is the pattern that
> reliably runs one-off scripts against this image. If a rebuild "succeeds" but the
> engine timestamp doesn't change, this is the first thing to check.

> **Clear the calibration cache before rebuilding a PTQ engine** or TRT will reuse a
> stale one: `rm -f deployment/output/engines/<variant>/pts_backbone_neck_head.calib_cache`.

---

## The deployment pipeline

CenterPoint is split into two engines, mirroring the ONNX export:

```
raw LiDAR (multi-sweep, ~278k points)
   → [host/preproc] voxelize + pillar feature encoding
   → pts_voxel_encoder.engine          (FP32, ~0.09 MB — tiny, not worth quantizing)
   → [scatter to dense 512×512 BEV grid]
   → pts_backbone_neck_head.engine     (the engine that matters: INT8/QAT/FP16)
   → 6 head outputs (heatmap/reg/height/dim/rot/vel)
   → [standalone numpy decode + circle-NMS]
   → nuScenes submission → mAP/NDS
```

`build_engine.py` builds both. The backbone+neck+head engine is where all the
size/latency/accuracy questions live; the encoder is negligible.

### Why the deployment container is lean (TensorRT + pycuda + numpy, no PyTorch)

The whole point of compiling to a TensorRT engine is a minimal-footprint embedded
inference path. The container has TensorRT, pycuda, numpy, and nuscenes-devkit — but
**not** PyTorch or mmdet3d. This has one important consequence for evaluation: the
detection decode (heatmap sigmoid → peak extraction → box regression → circle-NMS) is
a **standalone numpy reimplementation** (`scripts/eval.py`), not mmdet3d's PyTorch
`CenterHead.predict_by_feat`. That decode's own imprecision is therefore *part of* the
deployed system's measured accuracy — which is the honest thing to measure for a
deployment, and is why we did not route engine outputs back through mmdet3d on the pod
just to get a prettier number (that would answer "what could the engine achieve under
ideal postprocessing", a different question from "what ships").

---

## What we struggled with, and what we found

This is the part worth reading. Deployment was not "export ONNX, build engine, done" —
it was a multi-week debugging effort across two largely independent problems: getting
the on-device accuracy to be **non-zero and correct**, and then getting INT8 to be
**accurate without being slow**.

### Problem 1 — on-device mAP was exactly 0.0000 for days

The submission ran, produced boxes, and scored a clean `mAP = 0.0000`. The cause turned
out to be a chain of separate bugs, each of which had to be removed before the next was
visible. In discovery order:

1. **Coordinate frame.** The decode emitted boxes in the LiDAR/BEV frame; nuScenes eval
   requires the global frame. Fixed with per-sample `lidar2ego` + `ego2global`
   transforms from the val pkl.
2. **Quaternion orthogonality.** pkl float drift tripped `pyquaternion`'s strict
   orthogonality check; fixed by re-orthonormalizing rotation matrices via SVD
   (`R = U·Vᵀ`).
3. **Box size convention.** Decode emitted `[l, w, h]`; nuScenes wants `[w, l, h]`. A
   silent transpose that corrupted every box footprint while leaving centers correct
   (which is why center-distance debugging missed it).
4. **Mini-nuScenes GT loading.** To evaluate memory-efficiently, a filtered nuScenes is
   built from only the 512 evaluated tokens. `sample_annotation` `prev`/`next` links can
   point outside that subset, raising `KeyError` mid-load and silently yielding zero GT.
   Fixed by severing any link pointing outside the subset (velocity then falls back to
   `[nan, nan]`, the documented boundary behavior).
5. **The precision floor.** With the above fixed, mAP was *still* exactly 0. Calling
   nuScenes' own `accumulate()` directly (instead of the wrapped evaluator) revealed
   why: **max_recall = 1.000, max_precision = 0.018** — every object was detected but
   buried under ~50 false positives each, and `calc_ap` clips precision below
   `min_precision = 0.1` and returns *exactly* 0 below that floor. This reframed the
   problem from "nothing matches" to "false-positive flood" → pointed at missing peak
   suppression (CenterPoint needs a 3×3 max-pool NMS keeping only local-maximum cells;
   the decode had been thresholding every cell).
6. **Peak suppression wasn't enough — the heatmap was uniformly weak.** Even after
   adding peak extraction, predicted box counts barely dropped and mAP stayed at 0. Raw
   heatmap activations showed car peaking at sigmoid ≈0.31 (a healthy head produces
   0.8–0.99) — backwards and far too low.
7. **INT8 was suspected, and ruled out.** Running the FP32 ONNX (zero quantization)
   through the *identical* decode produced the *same* flat heatmap — proof the problem
   was upstream of TensorRT entirely.
8. **Root cause: missing multi-sweep LiDAR aggregation.** The model config specifies
   `LoadPointsFromMultiSweeps(sweeps_num=9)` — CenterPoint expects ~278k aggregated
   points per frame (current + 9 motion-compensated past sweeps), not a single ~30k
   scan. Every BEV calibration tensor up to that point had been built from
   **single-sweep** point clouds. The detector was never broken; it was starved of
   input density. Regenerating calibration BEV via a forward hook on the model's real
   multi-sweep forward pass took point counts to ~278k and heatmap peaks to 0.85–0.93,
   and mAP became non-zero.

**Meta-lesson:** the symptom (mAP = 0) pointed at the eval harness, but the cause was
three layers upstream in calibration-data preprocessing. The breakthrough was
instrumenting nuScenes' own `accumulate()` to separate recall ("is detection working?")
from precision ("is it usable?") rather than treating the evaluator as a black box.

> The corrected calibration generator lives in `compression/generate_calib_bev.py`. It
> registers a forward hook on `pts_middle_encoder` and runs the model's real
> `extract_feat()` over samples from `nuscenes_infos_val.pkl` (which supplies the
> multi-sweep metadata), so the saved BEV tensors are exactly what the deployed engine
> will see — not a hand-reconstructed approximation.

### Problem 2 — INT8 was accurate on the A40, but lost accuracy on TensorRT

With correct multi-sweep data, the first TRT INT8 engine still scored far below FP32.
This was a second, independent finding about *how* INT8 is produced. We isolated it
with a controlled experiment — same 512 samples, same decode, only the quantization
path differs:

```
ONNX FP32 (zero quantization, the realistic ceiling):  mAP 0.432    NDS 0.485
TRT INT8 — entropy calibrator (plain FP32 weights):    mAP 0.30     NDS 0.39
TRT INT8 — minmax calibrator  (plain FP32 weights):    mAP 0.3612   NDS 0.4304
TRT INT8 — QAT weights + explicit Q/DQ:                mAP 0.4265   NDS 0.4804
```

What each step established:

- **Entropy calibration (TRT default) was worst.** It minimizes histogram
  KL-divergence, which clips the rare high-magnitude classification activations a
  CenterPoint head needs for confident peaks.
- **MinMax recovered about half the gap** by using literal observed min/max (no
  distributional clipping). Added to `build_engine.py` as `--calibrator minmax`.
- **Forcing the head to FP16 did essentially nothing** (0.3612 → 0.3613). An
  informative *negative* result: it ruled out "the loss is in the head" and showed the
  PTQ loss is distributed across the INT8 backbone/neck features themselves. (This also
  surfaced a TRT gotcha: per-layer `precision = fp16` is only a *hint* — it's ignored
  unless you also set `BuilderFlag.OBEY_PRECISION_CONSTRAINTS`. Without that flag,
  predictions were bit-identical with and without `--fp16-head`.)
- **A disjoint calib/eval split ruled out a same-data artifact.** Because the 512
  calibration samples doubled as the eval set, we checked whether that inflated INT8's
  numbers by calibrating on 256 and evaluating on the held-out 256. Same per-class
  pattern — and theory says same-data is the *favorable* case for INT8, so a real
  held-out eval can only be equal or worse. The gap is architectural, not an artifact.
- **QAT weights + explicit quantization recovered the gap** (0.4265, at the 0.432
  ceiling). This is the A40 "near-free INT8" result finally reproduced on-device.

#### How the QAT engine was actually produced (and a dead end we avoided)

The QAT checkpoint (`qat_best.pth`) was trained with `torch.ao.quantization`, so its
state_dict mixes real weights with FakeQuantize bookkeeping and has BatchNorm folded
into the conv modules (FX fusion). A tempting-but-wrong approach is to strip the
FakeQuantize modules and load the weights into a plain FP32 model — but that **discards
the QAT tuning** and gets you back to plain weights (i.e. the 0.36 PTQ result).

What actually worked: apply QAT preparation in the **export** step (`export_onnx.py`
handles quantization detection on the backbone+neck and exports ONNX *with* explicit
`QuantizeLinear`/`DequantizeLinear` nodes carrying the learned scales). TensorRT then
runs its explicit-quantization path — confirmed by the build log:

```
[TRT] Calibrator won't be used in explicit quantization mode.
      Please insert Quantize/Dequantize layers ...
```

That message is the *signal it worked*: the calibrator becomes a no-op because the
scales come from the QAT graph, not from calibration.

#### The catch: QAT explicit quantization is 3× slower on TensorRT

| | PTQ INT8 (minmax) | QAT INT8 (explicit Q/DQ) |
|---|---|---|
| on-device mAP (512) | 0.3612 | **0.4265** |
| latency (p50) | 8.09ms | 25.70ms | 18.87ms |
| VDD_IN | 16.60W | 19.87W |
| engine | 6.82 MB | 13.29 MB |

The same Q/DQ nodes that carry the learned scales also **block TensorRT's Conv+BN+ReLU
layer fusion**: in explicit-quantization mode TRT honors the Q/DQ graph structure, so
each quantized layer becomes a separate kernel launch with dtype conversions instead of
fusing into one INT8 kernel. Hence 25.70ms (vs PTQ's 8.09ms) and the higher power.

This is a property of `torch.ao.quantization`'s general-purpose Q/DQ placement, not an
inherent limit. **Future work:** NVIDIA's `pytorch-quantization` toolkit places Q/DQ
nodes where TRT's compiler can fuse through them, which should keep QAT accuracy while
restoring PTQ-level latency.

---

## Why QAT INT8 on the 25W mode is the right deployment choice

Three things make this the call, despite QAT being the slowest engine:

1. **Accuracy is the project's purpose.** QAT recovers 0.3612 → 0.4265 mAP (to the
   FP32 ceiling). PTQ INT8 is faster but leaves ~16% relative accuracy on the table;
   for a perception model that's the wrong trade.
2. **The latency still fits real-time.** nuScenes LiDAR is 20Hz; automotive LiDAR is
   10–20Hz; Autoware's standard target is 10Hz (100ms/frame). At 25.70ms the QAT engine
   clears the 10Hz budget with ~74ms to spare for voxelization, decode, and ROS2
   messaging. The deployment constraint is *latency vs frame period*, not raw FPS — the
   FPS column in the benchmark tables is only useful for comparing variants.
3. **The 25W envelope is already the operating point.** The companion
   EdgeDrive-Perception project runs its ROS2 camera/LiDAR/fusion visualization on the
   Jetson's 25W mode. QAT INT8 draws 19.87W VDD_IN, comfortably within 25W (it
   exceeds the 15W mode, which is why 15W is not the target here).

---

## Why pruning is absent from the deployment recommendation

The pruned/distilled variants were benchmarked (see tables in
[`../docs/design_decisions.md`](../docs/design_decisions.md)) but are not deployment
candidates. Removing 43.6% of backbone+neck parameters changes latency by under ~1ms,
and Pruned 40% is actually *slower* than the unpruned INT8 engine, because:

- **Tensor-core alignment:** L1 pruning yields non-power-of-2 channel counts that TRT
  pads to multiples of 16 for INT8 dispatch, erasing much of the nominal FLOP saving.
- **Memory-bandwidth bound:** the 512×512 BEV spatial dimensions dominate latency and
  don't change with channel pruning.
- **Fixed head overhead:** the 6 task heads are unpruned and impose a constant floor.

So this architecture is bandwidth-bound, not compute-bound, on Orin Nano — pruning is
the wrong compression axis, and quantization (done via QAT) is the right one.

> The on-device pruned/distilled mAP numbers are additionally depressed because those
> variants used **PTQ on plain fine-tuned weights** (none were QAT-trained), incurring
> the same ~16% PTQ gap measured above. Their A40 reference mAP is the fair
> variant-to-variant comparison.

---

## Environment notes (Jetson-specific gotchas)

- **`sudo jetson_clocks` before every benchmark**, on the host. It sets hardware
  clocks, not container state; without it the GPU idles and latency is meaningless.
- **Power mode:** set the board to 25W (`sudo nvpmodel -m <id>`; the id is
  board-specific) to reproduce the deployed numbers.
- **Docker daemon after a force-reboot:** if the bridge/iptables kernel modules
  (`xt_addrtype`) failed to load, `/etc/docker/daemon.json` with
  `{"iptables":false,"bridge":"none", ...}` plus the nvidia default-runtime lets the
  `network_mode: host` services run without the bridge.
- **Power telemetry:** `benchmark.py` reads tegrastats and reports both
  `VDD_CPU_GPU_CV` (CPU+GPU+CV rail, inference-relevant) and `VDD_IN` (total module
  input). The 25W envelope is measured against `VDD_IN`.
- **Engine portability warning:** TRT prints "Using an engine plan file across
  different models of devices is not recommended" if an engine built on one device is
  run on another — engines are device-specific; rebuild on the target.

---

## Reproducing the full per-variant sweep

```bash
sudo jetson_clocks

# Build every variant (PTQ minmax for the FP32/pruned/distilled engines; qat_best
# carries its own Q/DQ so the calibrator is a no-op there):
for V in fp32 pruned25 pruned40 pruned55 distilled25; do
  rm -f deployment/output/engines/$V/pts_backbone_neck_head.calib_cache
  docker compose -f deployment/docker/docker-compose.yml run --rm \
      --entrypoint python3 build-engines scripts/build_engine.py \
      --onnx-dir /workspace/onnx --calib-dir /workspace/calib_bev \
      --out /workspace/output/engines --variant $V --calibrator minmax
done
docker compose -f deployment/docker/docker-compose.yml run --rm \
    --entrypoint python3 build-engines scripts/build_engine.py \
    --onnx-dir /workspace/onnx --calib-dir /workspace/calib_bev \
    --out /workspace/output/engines --variant qat_best

# Benchmark + submission for all:
for V in fp32 pruned25 pruned40 pruned55 distilled25 qat_best; do
  VARIANT=$V docker compose -f deployment/docker/docker-compose.yml run --rm benchmark
  VARIANT=$V docker compose -f deployment/docker/docker-compose.yml run --rm infer
done

# mAP/NDS for all (on host):
python3 deployment/scripts/eval_metrics.py \
    --nuscenes ~/Downloads/v1.0-trainval_meta \
    --val-pkl ~/Downloads/runpod/nuscenes_infos_val.pkl \
    --submissions deployment/output \
    --variants fp32 pruned25 pruned40 pruned55 distilled25 qat_best \
    --out deployment/output/eval_all_final.json
```

---

## Status

P3 (Jetson deployment) is complete: the engine pipeline builds, benchmarks, and
evaluates on-device; the multi-sweep accuracy bug is fixed; the INT8 deployment method
is resolved (QAT + explicit quantization); and QAT INT8 on 25W is the selected
configuration. Remaining future work is the `pytorch-quantization`-based QAT export to
remove the layer-fusion latency penalty, and ROS2 / Autoware node integration.
"""
Pareto front assembly for the EdgeFusion-CenterPoint compression sweep.

Produces three views:

1. ARCHITECTURE PARETO (measured)
   x = backbone+neck params, % of FP32 baseline — the metric pruning directly
       controls: (1-ratio)^2 -> 25%->56.4%, 40%->36.0%, 55%->20.3%.
   y = mAP / NDS, as measured.

   FP32/PTQ/QAT all sit at x=100% (identical architecture, only precision
   differs) — INT8's benefit isn't visible on this axis. This view answers
   "how much does removing channels cost, independent of quantization?"

2. DEPLOYMENT PARETO (real TRT engine sizes)
   x = real TRT engine size, % of the FP32 PTQ engine (6.82 MB = 100%),
       measured on Jetson Orin Nano.
   y = mAP. FP32/PTQ/QAT carry their A40 mAP for the architecture comparison;
       note QAT's engine is LARGER (194.9%, 13.29 MB) not smaller, because
       explicit-quantization engines carry Q/DQ scale tensors and keep more of
       the graph unfused. Pruned/Distilled points use real Jetson engine sizes
       with mAP carried from their A40 FP32 eval (marked distinctly).

3. ON-DEVICE PARETO (full end-to-end pipeline)
   x = full pipeline p50 latency (ms): BEV htod + TRT engine + CUDA postproc
       + dtoh + circle NMS. QAT INT8 is MEASURED via benchmark_e2e.py (200
       iterations, 25W MAXN). PTQ/pruned/distilled variants are ESTIMATED as
       engine_p50 + CUDA_PP_OVERHEAD (3.67ms, measured on QAT) + NMS_OVERHEAD.
   y = on-device mAP (512-sample subset, standalone CUDA decode).

4. OPERATING-POINT PARETO (the interview slide)
   x = full pipeline p99 latency (conservative real-time metric).
   y = on-device mAP.
   Budget constraints plotted as dashed lines:
     mAP ≥ 0.40 (production-viable accuracy threshold)
     p99 ≤ 50ms (20Hz LiDAR frame budget)
   The green feasibility region (upper-left) contains exactly one variant:
   QAT INT8. This is the operating-point decision.

Usage:
    python EdgeFusion-CenterPoint/compression/pareto.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
PARETO_DIR = RESULTS_DIR / "pareto"
PARETO_DIR.mkdir(parents=True, exist_ok=True)

# Real TRT engine backbone+neck+head sizes, measured on Jetson Orin Nano
# (JetPack R36.4.0, TRT 10.3.0, 25W). Stored as % of fp32 PTQ engine.
TRT_ENGINE_MB = {
    "FP32 baseline": 6.82,
    "PTQ INT8": 6.82,
    "QAT INT8": 13.29,
    "Pruned 25%": 4.90,
    "Pruned 40%": 4.35,
    "Pruned 55%": 3.44,
    "Distilled (25% arch)": 4.92,
}
FP32_ENGINE_MB = TRT_ENGINE_MB["FP32 baseline"]

# ── End-to-end pipeline overhead constants (from benchmark_e2e.py, qat_best) ──
# QAT INT8 was the only variant benchmarked end-to-end. The overhead components
# are applied as estimates to the PTQ variants:
#   CUDA_PP_OVERHEAD: BEV htod + peak_finding + box_decode kernels = 3.67ms
#     (QAT gpu_portion 29.37ms − engine-only 25.70ms = 3.67ms)
#     Same for all variants — same heatmap shape [1,10,128,128], same kernels.
#   NMS_OVERHEAD_FULL: dtoh + circle NMS for full-arch variants = 4.88ms
#     (QAT CPU tail; PTQ has similar heatmap activation density at score_thr=0.1)
#   NMS_OVERHEAD_PRUNED: estimated 3.5ms for pruned/distilled — sparser heatmaps
#     produce fewer pre-NMS candidates → faster vectorised NMS.
#   E2E_P99_JITTER: conservative p99 inflation for PTQ variants (engine p99 jitter
#     is ~0.02ms; CUDA PP and NMS add ~1.5ms jitter budget). QAT p99 is measured.
CUDA_PP_OVERHEAD = 3.67  # ms — measured
NMS_OVERHEAD_FULL = 4.88  # ms — measured (QAT); applied to full-arch PTQ
NMS_OVERHEAD_PRUNED = 3.50  # ms — estimated for pruned/distilled
E2E_P99_JITTER = 1.50  # ms — conservative estimate for PTQ p99 inflation

# Operating-point budget
BUDGET_MAP_MIN = 0.40  # mAP ≥ this (on-device 512-sample)
BUDGET_P99_MS = 50.0  # p99 ≤ this ms (20Hz LiDAR frame budget)
BUDGET_POWER_W = 25.0  # VDD_IN ≤ this W (25W MAXN envelope)

# Measured results — see compression/README.md for full derivation of each.
#
# mAP/NDS: A40 PyTorch full-val (6019-sample) accuracy (architecture Pareto).
# trt_engine_pct: real TRT engine size as % of FP32 PTQ engine (deployment Pareto).
# jetson_latency_ms: engine-only p50 (from benchmark.py — kept for reference).
# jetson_map512: on-device 512-sample mAP via CUDA decode (on-device Pareto).
# e2e_p50_ms: full pipeline p50 — MEASURED for QAT, ESTIMATED for others.
# e2e_p99_ms: full pipeline p99 — MEASURED for QAT, ESTIMATED for others.
# e2e_vddin_w: VDD_IN during full pipeline — MEASURED for QAT, engine-only for rest.
VARIANTS = [
    {
        "name": "FP32 baseline",
        "mAP": 0.4815,
        "NDS": 0.5922,
        "params_pct": 100.0,
        "full_model_pct": 100.0,
        "precision": "FP32",
        "trt_engine_pct": 100.0,
        "jetson_latency_ms": 8.09,
        "jetson_vddin_w": 16.60,
        "jetson_map512": 0.3612,
        "e2e_p50_ms": 8.09 + CUDA_PP_OVERHEAD + NMS_OVERHEAD_FULL,
        "e2e_p99_ms": 8.09 + CUDA_PP_OVERHEAD + NMS_OVERHEAD_FULL + E2E_P99_JITTER,
        "e2e_vddin_w": 16.60,
        "e2e_measured": False,
    },
    {
        "name": "PTQ INT8",
        "mAP": 0.4812,
        "NDS": 0.5903,
        "params_pct": 100.0,
        "full_model_pct": 100.0,
        "precision": "INT8",
        "trt_engine_pct": 100.0,
        "jetson_latency_ms": 8.09,
        "jetson_vddin_w": 16.60,
        "jetson_map512": 0.3612,
        "e2e_p50_ms": 8.09 + CUDA_PP_OVERHEAD + NMS_OVERHEAD_FULL,
        "e2e_p99_ms": 8.09 + CUDA_PP_OVERHEAD + NMS_OVERHEAD_FULL + E2E_P99_JITTER,
        "e2e_vddin_w": 16.60,
        "e2e_measured": False,
    },
    {
        "name": "QAT INT8",
        "mAP": 0.4814,
        "NDS": 0.5910,
        "params_pct": 100.0,
        "full_model_pct": 100.0,
        "precision": "INT8",
        "trt_engine_pct": 194.9,
        "jetson_latency_ms": 25.70,
        "jetson_vddin_w": 19.87,
        "jetson_map512": 0.4265,
        "e2e_p50_ms": 34.12,  # MEASURED — benchmark_e2e.py, 200 iters, 25W
        "e2e_p99_ms": 43.93,  # MEASURED
        "e2e_vddin_w": 16.29,  # MEASURED (lower than engine-only: GPU idles during NMS)
        "e2e_measured": True,
    },
    {
        "name": "Pruned 25%",
        "mAP": 0.4081,
        "NDS": 0.5382,
        "params_pct": 56.4,
        "full_model_pct": 67.15,
        "precision": "FP32",
        "trt_engine_pct": 71.85,
        "jetson_latency_ms": 7.46,
        "jetson_vddin_w": 16.41,
        "jetson_map512": 0.2637,
        "e2e_p50_ms": 7.46 + CUDA_PP_OVERHEAD + NMS_OVERHEAD_PRUNED,
        "e2e_p99_ms": 7.46 + CUDA_PP_OVERHEAD + NMS_OVERHEAD_PRUNED + E2E_P99_JITTER,
        "e2e_vddin_w": 16.41,
        "e2e_measured": False,
    },
    {
        "name": "Pruned 40%",
        "mAP": 0.2838,
        "NDS": 0.3902,
        "params_pct": 36.0,
        "full_model_pct": 51.48,
        "precision": "FP32",
        "trt_engine_pct": 63.78,
        "jetson_latency_ms": 8.17,
        "jetson_vddin_w": 15.81,
        "jetson_map512": 0.1176,
        "e2e_p50_ms": 8.17 + CUDA_PP_OVERHEAD + NMS_OVERHEAD_PRUNED,
        "e2e_p99_ms": 8.17 + CUDA_PP_OVERHEAD + NMS_OVERHEAD_PRUNED + E2E_P99_JITTER,
        "e2e_vddin_w": 15.81,
        "e2e_measured": False,
    },
    {
        "name": "Pruned 55%",
        "mAP": 0.2149,
        "NDS": 0.3136,
        "params_pct": 20.3,
        "full_model_pct": 39.55,
        "precision": "FP32",
        "trt_engine_pct": 50.44,
        "jetson_latency_ms": 6.93,
        "jetson_vddin_w": 15.61,
        "jetson_map512": 0.1556,
        "e2e_p50_ms": 6.93 + CUDA_PP_OVERHEAD + NMS_OVERHEAD_PRUNED,
        "e2e_p99_ms": 6.93 + CUDA_PP_OVERHEAD + NMS_OVERHEAD_PRUNED + E2E_P99_JITTER,
        "e2e_vddin_w": 15.61,
        "e2e_measured": False,
    },
]


def load_distillation_result() -> None:
    """Append distillation result if available — runs independently of this
    script's other (final) values, since it's the only pending stage."""
    path = RESULTS_DIR / "distillation" / "ratio_25" / "distilled_25_metrics.json"
    if not path.exists():
        print(f"[pareto] {path} not found — distillation result omitted.")
        return
    with open(path) as f:
        d = json.load(f)
    VARIANTS.append(
        {
            "name": "Distilled (25% arch)",
            "mAP": d["mAP"],
            "NDS": d["NDS"],
            "params_pct": 56.4,
            "full_model_pct": 67.15,
            "precision": "FP32",
            "trt_engine_pct": 72.14,
            "jetson_latency_ms": 7.46,
            "jetson_vddin_w": 16.38,
            "jetson_map512": 0.2599,
            "e2e_p50_ms": 7.46 + CUDA_PP_OVERHEAD + NMS_OVERHEAD_PRUNED,
            "e2e_p99_ms": 7.46 + CUDA_PP_OVERHEAD + NMS_OVERHEAD_PRUNED + E2E_P99_JITTER,
            "e2e_vddin_w": 16.38,
            "e2e_measured": False,
        }
    )
    print(f'[pareto] Distillation result loaded: mAP {d["mAP"]:.4f}  NDS {d["NDS"]:.4f}')


def pareto_front(points: list, x_key: str, y_key: str) -> list:
    """Return the subset of points not dominated by any other point.

    A point is dominated if another point has x <= this.x AND y >= this.y,
    with at least one strict inequality (lower size is better, higher
    accuracy is better).
    """
    front = []
    for p in points:
        dominated = any(
            q is not p
            and q[x_key] <= p[x_key]
            and q[y_key] >= p[y_key]
            and (q[x_key] < p[x_key] or q[y_key] > p[y_key])
            for q in points
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p[x_key])


def plot_architecture_pareto(variants: list) -> list:
    """Chart 1: measured params_pct vs mAP/NDS."""
    front = pareto_front(variants, "params_pct", "mAP")
    front_names = {p["name"] for p in front}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes, ["mAP", "NDS"], ["mAP", "NDS"]):
        # Variants sharing the same params_pct (FP32/PTQ/QAT all at 100%)
        # get vertically-stacked label offsets to avoid overlap.
        seen_x = {}
        for v in variants:
            marker = "o" if v["precision"] == "FP32" else "s"
            color = "tab:blue" if v["name"] in front_names else "lightgray"
            ax.scatter(
                v["params_pct"],
                v[metric],
                marker=marker,
                s=80,
                color=color,
                edgecolors="black",
                zorder=3,
            )
            stack = seen_x.get(v["params_pct"], 0)
            seen_x[v["params_pct"]] = stack + 1
            ax.annotate(
                v["name"],
                (v["params_pct"], v[metric]),
                textcoords="offset points",
                xytext=(8, 6 - stack * 12),
                fontsize=8,
            )
        fx = [p["params_pct"] for p in front]
        fy = [p[metric] for p in front]
        ax.plot(fx, fy, "--", color="tab:blue", alpha=0.5, zorder=2, label="Pareto front")
        ax.set_xlabel("Backbone+Neck Params (% of FP32)")
        ax.set_ylabel(title)
        ax.set_title(f"Architecture Pareto — {title} vs Params")
        ax.set_xlim(108, 12)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")

    fig.suptitle(
        "Measured: precision held fixed per variant " "(FP32 \u25cb circle / INT8 \u25a1 square)"
    )
    fig.tight_layout()
    out = PARETO_DIR / "architecture_pareto.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[pareto] Saved {out}")
    return front


def plot_deployment_pareto(variants: list) -> list:
    """Chart 2: real TRT engine size (% of FP32 PTQ engine) vs mAP.

    QAT is plotted at its real measured engine size (194.9% — larger than the
    PTQ engine, since explicit-quantization engines carry Q/DQ scale tensors
    and keep more of the graph unfused). Pruned/Distilled variants use real
    Jetson engine sizes (trt_engine_pct) with mAP carried from their A40 FP32
    eval — marked with hollow markers. FP32 baseline and PTQ are excluded
    (FP32 isn't INT8; PTQ is dominated by QAT on accuracy).
    """
    points = []
    for v in variants:
        if v["name"] == "QAT INT8":
            points.append(
                {
                    **v,
                    "size_pct": v["trt_engine_pct"],
                    "projected": False,
                }
            )
        elif v["precision"] == "FP32" and v["name"] != "FP32 baseline":
            points.append(
                {
                    **v,
                    "size_pct": v["trt_engine_pct"],
                    "projected": True,
                }
            )
    # FP32 baseline and PTQ excluded: FP32 isn't INT8, and PTQ is dominated
    # by QAT at the same size with higher accuracy.

    front = pareto_front(points, "size_pct", "mAP")
    front_names = {p["name"] for p in front}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    # Points within this (size_pct, mAP) distance get vertically-stacked
    # label offsets to avoid overlap (e.g. Pruned 25% and Distilled sit at
    # nearly the same point ~72% / ~0.408-0.409).
    placed = []
    for p in points:
        marker = "D" if not p["projected"] else "o"
        facecolor = "tab:blue" if p["name"] in front_names else "lightgray"
        if p["projected"]:
            ax.scatter(
                p["size_pct"],
                p["mAP"],
                marker=marker,
                s=80,
                facecolors="none",
                edgecolors=facecolor,
                linewidths=2,
                zorder=3,
            )
            label = f"{p['name']} (Jetson TRT)"
        else:
            ax.scatter(
                p["size_pct"],
                p["mAP"],
                marker=marker,
                s=90,
                color=facecolor,
                edgecolors="black",
                zorder=3,
            )
            label = f"{p['name']} (measured)"

        stack = sum(
            1 for (px, py) in placed if abs(px - p["size_pct"]) < 1.5 and abs(py - p["mAP"]) < 0.01
        )
        placed.append((p["size_pct"], p["mAP"]))
        ax.annotate(
            label,
            (p["size_pct"], p["mAP"]),
            textcoords="offset points",
            xytext=(8, 6 - stack * 14),
            fontsize=8,
        )

    fx = [p["size_pct"] for p in front]
    fy = [p["mAP"] for p in front]
    ax.plot(fx, fy, "--", color="tab:blue", alpha=0.5, zorder=2, label="Pareto front")

    ax.set_xlabel("TRT engine size (% of FP32 PTQ engine, 6.82 MB = 100%)")
    ax.set_ylabel("mAP (A40 full-val reference)")
    ax.set_title(
        "Deployment Pareto — Real TRT Engine Sizes (Jetson Orin Nano)\n"
        "\u25c6 QAT INT8 measured (13.29 MB)  \u00b7  "
        "\u25cb Pruned/Distilled Jetson engine (mAP from A40 eval)",
        fontsize=10,
    )
    # x descending (smaller-is-better to the right) but must span QAT at 194.9%.
    ax.set_xlim(205, 44)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = PARETO_DIR / "deployment_pareto.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[pareto] Saved {out}")
    return front


def plot_ondevice_pareto(variants: list) -> list:
    """Chart 3: full e2e pipeline p50 vs on-device 512-sample mAP.

    x = e2e pipeline p50 (ms): engine + CUDA postproc + dtoh + NMS.
        MEASURED for QAT INT8 (benchmark_e2e.py, 200 iters).
        ESTIMATED for PTQ variants: engine_p50 + CUDA_PP_OVERHEAD + NMS_OVERHEAD.
    y = on-device mAP (512-sample subset, CUDA decode).
    """
    points = [v for v in variants if "e2e_p50_ms" in v]
    front = pareto_front(points, "e2e_p50_ms", "jetson_map512")
    front_names = {p["name"] for p in front}

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for x, label, col in [
        (50.0, "20Hz (50ms)", "tab:orange"),
        (100.0, "10Hz (100ms)", "tab:green"),
    ]:
        ax.axvline(x, color=col, linestyle=":", alpha=0.65, zorder=1)

        ax.text(
            x,
            0.5,
            label,
            color=col,
            fontsize=8,
            rotation=90,
            transform=ax.get_xaxis_transform(),  # x=data, y=axes (0~1)
            va="center",
            ha="right",
        )

    placed = []
    for p in points:
        is_deployed = p["name"] == "QAT INT8"
        on_front = p["name"] in front_names
        color = "tab:red" if is_deployed else ("tab:blue" if on_front else "lightgray")
        marker = "*" if is_deployed else ("o" if p["e2e_measured"] else "s")
        size = 320 if is_deployed else 90
        ax.scatter(
            p["e2e_p50_ms"],
            p["jetson_map512"],
            marker=marker,
            s=size,
            color=color,
            edgecolors="black",
            zorder=3,
        )
        suffix = " (deployed)" if is_deployed else ("" if p["e2e_measured"] else " (est.)")
        stack = sum(
            1
            for (px, py) in placed
            if abs(px - p["e2e_p50_ms"]) < 1.5 and abs(py - p["jetson_map512"]) < 0.02
        )
        placed.append((p["e2e_p50_ms"], p["jetson_map512"]))
        ax.annotate(
            f"{p['name']}{suffix}",
            (p["e2e_p50_ms"], p["jetson_map512"]),
            textcoords="offset points",
            xytext=(8, 6 - stack * 13),
            fontsize=8,
        )

    fx = [p["e2e_p50_ms"] for p in front]
    fy = [p["jetson_map512"] for p in front]
    ax.plot(fx, fy, "--", color="tab:blue", alpha=0.5, zorder=2, label="Pareto front")

    legend_elems = [
        plt.scatter(
            [],
            [],
            marker="*",
            s=140,
            color="tab:red",
            edgecolors="black",
            label="QAT INT8 (deployed, measured)",
        ),
        plt.scatter(
            [],
            [],
            marker="o",
            s=60,
            color="tab:blue",
            edgecolors="black",
            label="On Pareto front (measured)",
        ),
        plt.scatter(
            [],
            [],
            marker="s",
            s=60,
            color="lightgray",
            edgecolors="black",
            label="Off front (e2e estimated)",
        ),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)
    ax.set_xlabel("Full pipeline p50 latency (ms)  —  \u25cf measured  \u25a1 estimated")
    ax.set_ylabel("On-device mAP (512-sample subset)")
    ax.set_title(
        "On-device Pareto — Full Pipeline Latency vs Accuracy  (25W)\n"
        "QAT INT8: highest on-device accuracy within 10-20Hz LiDAR budget",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PARETO_DIR / "ondevice_pareto.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[pareto] Saved {out}")
    return front


def plot_operating_point(variants: list) -> None:
    """Chart 4: operating-point decision — p99 latency vs on-device mAP.

    The interview slide. Shows all variants against the stated deployment budget:
      mAP >= 0.40  (horizontal threshold — production-viable accuracy)
      p99 <= 50ms  (vertical threshold — 20Hz LiDAR frame budget)
      VDD_IN <= 25W (all variants pass; noted in title, not a chart axis)

    The green feasibility region (upper-left) contains exactly one variant:
    QAT INT8. This makes the operating-point choice visually unambiguous.
    """
    points = [v for v in variants if "e2e_p99_ms" in v]

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Feasibility region shading
    xlim_frac = BUDGET_P99_MS / 115.0
    ax.axhspan(BUDGET_MAP_MIN, 0.50, xmin=0, xmax=xlim_frac, alpha=0.10, color="#2DC653", zorder=0)
    ax.text(
        2,
        BUDGET_MAP_MIN + 0.003,
        "  feasible region",
        color="#1a8a38",
        fontsize=8.5,
        fontstyle="italic",
        va="bottom",
    )

    # Budget constraint lines
    ax.axhline(
        BUDGET_MAP_MIN,
        color="#2DC653",
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        zorder=2,
        label=f"mAP >= {BUDGET_MAP_MIN} (accuracy budget)",
    )
    ax.axvline(
        BUDGET_P99_MS,
        color="#E63946",
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        zorder=2,
        label=f"p99 <= {BUDGET_P99_MS:.0f}ms (20Hz budget)",
    )
    ax.axvline(
        100.0,
        color="#888",
        linestyle=":",
        linewidth=1.2,
        alpha=0.6,
        zorder=1,
        label="p99 <= 100ms (10Hz budget)",
    )

    placed = []
    for p in points:
        is_deployed = p["name"] == "QAT INT8"
        in_budget = p["jetson_map512"] >= BUDGET_MAP_MIN and p["e2e_p99_ms"] <= BUDGET_P99_MS
        color = "#2DC653" if in_budget else "#BBBBBB"
        ec = "#155724" if in_budget else "#666"
        marker = "*" if is_deployed else ("o" if p["e2e_measured"] else "s")
        size = 400 if is_deployed else 100
        zo = 5 if is_deployed else 3
        ax.scatter(
            p["e2e_p99_ms"],
            p["jetson_map512"],
            marker=marker,
            s=size,
            color=color,
            edgecolors=ec,
            linewidths=1.5,
            zorder=zo,
        )
        suffix = "\n(deployed)" if is_deployed else (" (est.)" if not p["e2e_measured"] else "")
        stack = sum(
            1
            for (px, py) in placed
            if abs(px - p["e2e_p99_ms"]) < 2.5 and abs(py - p["jetson_map512"]) < 0.025
        )
        placed.append((p["e2e_p99_ms"], p["jetson_map512"]))
        ha = "left" if p["e2e_p99_ms"] < 80 else "right"
        ox = 4 if ha == "left" else -4
        ax.annotate(
            f"{p['name']}{suffix}",
            (p["e2e_p99_ms"], p["jetson_map512"]),
            textcoords="offset points",
            xytext=(ox, 6 - stack * 15),
            fontsize=8.5,
            ha=ha,
            color="#155724" if in_budget else "#444",
        )

    ax.set_xlim(0, 115)
    ax.set_ylim(0.05, 0.50)
    ax.set_xlabel(
        "Full pipeline p99 latency (ms, Jetson Orin Nano 25W)\n"
        "\u2605 measured  \u25a1 estimated (engine + CUDA postproc + NMS overhead)",
        fontsize=9.5,
    )
    ax.set_ylabel("On-device mAP (512-sample, CUDA decode)", fontsize=9.5)
    ax.set_title(
        "Operating-Point Decision  (Accuracy x Latency Pareto)\n"
        f"Budget: mAP >= {BUDGET_MAP_MIN}  p99 <= {BUDGET_P99_MS:.0f}ms  "
        f"VDD_IN <= {BUDGET_POWER_W:.0f}W  ->  QAT INT8 is the only feasible point",
        fontsize=10,
    )
    ax.grid(alpha=0.25, zorder=0)

    legend_elems = [
        mpatches.Patch(color="#2DC653", alpha=0.25, label="Feasible region"),
        plt.Line2D(
            [0], [0], color="#2DC653", linestyle="--", lw=1.5, label=f"mAP >= {BUDGET_MAP_MIN}"
        ),
        plt.Line2D([0], [0], color="#E63946", linestyle="--", lw=1.5, label="p99 <= 50ms (20Hz)"),
        plt.Line2D([0], [0], color="#888", linestyle=":", lw=1.2, label="p99 <= 100ms (10Hz)"),
        plt.scatter(
            [],
            [],
            marker="*",
            s=140,
            color="#2DC653",
            edgecolors="#155724",
            lw=1.5,
            label="QAT INT8 — feasible (measured)",
        ),
        plt.scatter(
            [],
            [],
            marker="o",
            s=60,
            color="#BBB",
            edgecolors="#666",
            label="Out of budget (measured)",
        ),
        plt.scatter(
            [],
            [],
            marker="s",
            s=60,
            color="#BBB",
            edgecolors="#666",
            label="Out of budget (estimated)",
        ),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="upper right", framealpha=0.92, frameon=True)
    fig.tight_layout()
    out = PARETO_DIR / "operating_point.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"[pareto] Saved {out}")


def main() -> None:
    load_distillation_result()

    print("\n[pareto] All variants:")
    for v in VARIANTS:
        print(
            f"  {v['name']:24s}  mAP {v['mAP']:.4f}  NDS {v['NDS']:.4f}  "
            f"params {v['params_pct']:5.1f}%  ({v['precision']})"
        )

    arch_front = plot_architecture_pareto(VARIANTS)
    print("\n[pareto] Architecture Pareto front (params_pct vs mAP):")
    for p in arch_front:
        print(f"  {p['name']:24s}  {p['params_pct']:5.1f}% params  " f"mAP {p['mAP']:.4f}")

    deploy_front = plot_deployment_pareto(VARIANTS)
    print("\n[pareto] Deployment Pareto front (engine size_pct vs A40 mAP):")
    for p in deploy_front:
        tag = "measured" if not p["projected"] else "mAP from A40 eval"
        print(f"  {p['name']:24s}  {p['size_pct']:5.1f}% size  " f"mAP {p['mAP']:.4f}  ({tag})")

    ondevice_front = plot_ondevice_pareto(VARIANTS)
    print("\n[pareto] On-device Pareto front (e2e p50 vs 512-sample mAP):")
    for p in ondevice_front:
        tag = "measured" if p["e2e_measured"] else "estimated"
        print(
            f"  {p['name']:24s}  e2e_p50={p['e2e_p50_ms']:5.2f}ms  "
            f"on-device mAP {p['jetson_map512']:.4f}  ({tag})"
        )

    plot_operating_point(VARIANTS)
    print("\n[pareto] Operating-point budget:")
    print(
        f"  mAP >= {BUDGET_MAP_MIN}  |  p99 <= {BUDGET_P99_MS:.0f}ms  "
        f"|  VDD_IN <= {BUDGET_POWER_W:.0f}W"
    )
    for p in VARIANTS:
        if "e2e_p99_ms" not in p:
            continue
        feasible = p["jetson_map512"] >= BUDGET_MAP_MIN and p["e2e_p99_ms"] <= BUDGET_P99_MS
        status = "FEASIBLE" if feasible else "out of budget"
        print(
            f"  {p['name']:24s}  mAP={p['jetson_map512']:.4f}  "
            f"p99={p['e2e_p99_ms']:.1f}ms  -> {status}"
        )

    summary = {
        "variants": VARIANTS,
        "architecture_pareto_front": [p["name"] for p in arch_front],
        "deployment_pareto_front": [
            {
                "name": p["name"],
                "size_pct": p["size_pct"],
                "mAP": p["mAP"],
                "projected": p["projected"],
            }
            for p in deploy_front
        ],
        "ondevice_pareto_front": [
            {
                "name": p["name"],
                "e2e_p50_ms": p["e2e_p50_ms"],
                "map512": p["jetson_map512"],
                "e2e_measured": p["e2e_measured"],
            }
            for p in ondevice_front
        ],
        "budget_constraints": {
            "map_min": BUDGET_MAP_MIN,
            "p99_max_ms": BUDGET_P99_MS,
            "power_max_w": BUDGET_POWER_W,
        },
        "operating_point": "QAT INT8",
        "fp32_engine_mb": FP32_ENGINE_MB,
    }
    summary_path = PARETO_DIR / "pareto_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[pareto] Summary saved to {summary_path}")

    print("\n[pareto] Operating-point decision:")
    print("  QAT INT8 — only variant satisfying all budget constraints.")
    print(
        f"  on-device mAP 0.4265  |  e2e p50={34.12:.2f}ms / p99={43.93:.2f}ms" "  |  16.29W VDD_IN"
    )


if __name__ == "__main__":
    main()

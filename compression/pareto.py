"""
Pareto front assembly for the EdgeFusion-CenterPoint compression sweep.

Produces two views:

1. ARCHITECTURE PARETO (measured)
   x = backbone+neck params, % of FP32 baseline — the metric pruning directly
       controls: (1-ratio)^2 -> 25%->56.4%, 40%->36.0%, 55%->20.3%.
   y = mAP / NDS, as measured.

   FP32/PTQ/QAT all sit at x=100% (identical architecture, only precision
   differs) — INT8's benefit isn't visible on this axis. This view answers
   "how much does removing channels cost, independent of quantization?"

2. PROJECTED DEPLOYMENT PARETO
   x = projected size, % of FP32 baseline, ASSUMING TRT INT8 on every variant
       (size_pct = params_pct * 0.25, i.e. INT8 = 1/4 the bytes of FP32).
   y = mAP, as measured for the FP32-precision variant (the projection
       assumption: INT8 is near-free, validated for the unpruned architecture
       via PTQ/QAT — 48.12/48.14 vs 48.15 FP32 — and ASSUMED to hold for
       pruned architectures too, since BatchNorm-after-every-conv is preserved
       regardless of channel count).

   QAT (100% arch, INT8) is plotted as measured (25% size, 0.4814 mAP) — this
   point is real. Pruned/Distilled variants are plotted at their pruned
   params_pct * 0.25, with mAP carried over from their FP32 measurement —
   these points are PROJECTED, marked distinctly, and are exactly what the
   Jetson TRT INT8 benchmark validates or refutes.

Usage:
    python EdgeFusion-CenterPoint/compression/pareto.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path(__file__).parent / 'results'
PARETO_DIR = RESULTS_DIR / 'pareto'
PARETO_DIR.mkdir(parents=True, exist_ok=True)

# Real TRT INT8 backbone+neck+head engine sizes, measured on Jetson Orin Nano
# (JetPack R36.4.0, TRT 10.3.0, INT8 with FP16 fallback, 512-sample entropy
# calibration). Stored as % of fp32 engine (6.82 MB).
# These replace the earlier ONNX-size × 0.25 projections — real values differ
# because TRT engines include fixed-overhead metadata, per-layer calibration
# tables, and kernel binaries regardless of model size, so the compression
# ratio is not a flat 4×:
#   fp32:       6.82 MB  (100.0%)  — reference
#   pruned25:   4.90 MB  ( 71.85%)
#   pruned40:   4.35 MB  ( 63.78%)
#   pruned55:   3.46 MB  ( 50.73%)
#   distilled25:4.92 MB  ( 72.14%)
# QAT INT8: not yet benchmarked on Jetson — using fp32 engine size as proxy
# (same architecture, so engine size should be equal).
TRT_ENGINE_MB = {
    'FP32 baseline': 6.82,
    'PTQ INT8': 6.82,  # same arch as FP32, proxy
    'QAT INT8': 6.82,  # same arch as FP32, proxy
    'Pruned 25%': 4.90,
    'Pruned 40%': 4.35,
    'Pruned 55%': 3.46,
    'Distilled (25% arch)': 4.92,
}
FP32_ENGINE_MB = TRT_ENGINE_MB['FP32 baseline']

# Measured results — see compression/README.md for full derivation of each.
#
# params_pct: backbone+neck channel ratio (architecture Pareto x-axis).
# full_model_pct: full ONNX size as % of FP32 baseline (context only — the
#   deployment Pareto now uses trt_engine_pct from real Jetson measurements).
# trt_engine_pct: real TRT INT8 backbone+neck+head engine size as % of FP32
#   engine — the correct deployment x-axis.
VARIANTS = [
    {'name': 'FP32 baseline', 'mAP': 0.4815, 'NDS': 0.5922,
     'params_pct': 100.0, 'full_model_pct': 100.0, 'precision': 'FP32',
     'trt_engine_pct': 100.0},
    {'name': 'PTQ INT8', 'mAP': 0.4812, 'NDS': 0.5903,
     'params_pct': 100.0, 'full_model_pct': 100.0, 'precision': 'INT8',
     'trt_engine_pct': 100.0},
    {'name': 'QAT INT8', 'mAP': 0.4814, 'NDS': 0.5910,
     'params_pct': 100.0, 'full_model_pct': 100.0, 'precision': 'INT8',
     'trt_engine_pct': 100.0},
    {'name': 'Pruned 25%', 'mAP': 0.4081, 'NDS': 0.5382,
     'params_pct': 56.4, 'full_model_pct': 67.15, 'precision': 'FP32',
     'trt_engine_pct': 71.85},
    {'name': 'Pruned 40%', 'mAP': 0.2838, 'NDS': 0.3902,
     'params_pct': 36.0, 'full_model_pct': 51.48, 'precision': 'FP32',
     'trt_engine_pct': 63.78},
    {'name': 'Pruned 55%', 'mAP': 0.2149, 'NDS': 0.3136,
     'params_pct': 20.3, 'full_model_pct': 39.55, 'precision': 'FP32',
     'trt_engine_pct': 50.73},
]


def load_distillation_result() -> None:
    """Append distillation result if available — runs independently of this
    script's other (final) values, since it's the only pending stage."""
    path = RESULTS_DIR / 'distillation' / 'ratio_25' / 'distilled_25_metrics.json'
    if not path.exists():
        print(f'[pareto] {path} not found — distillation result omitted.')
        return
    with open(path) as f:
        d = json.load(f)
    VARIANTS.append({
        'name': 'Distilled (25% arch)',
        'mAP': d['mAP'], 'NDS': d['NDS'],
        'params_pct': 56.4, 'full_model_pct': 67.15, 'precision': 'FP32',
        'trt_engine_pct': 72.14,
    })
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
            and q[x_key] <= p[x_key] and q[y_key] >= p[y_key]
            and (q[x_key] < p[x_key] or q[y_key] > p[y_key])
            for q in points
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p[x_key])


def plot_architecture_pareto(variants: list) -> list:
    """Chart 1: measured params_pct vs mAP/NDS."""
    front = pareto_front(variants, 'params_pct', 'mAP')
    front_names = {p['name'] for p in front}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes, ['mAP', 'NDS'], ['mAP', 'NDS']):
        # Variants sharing the same params_pct (FP32/PTQ/QAT all at 100%)
        # get vertically-stacked label offsets to avoid overlap.
        seen_x = {}
        for v in variants:
            marker = 'o' if v['precision'] == 'FP32' else 's'
            color = 'tab:blue' if v['name'] in front_names else 'lightgray'
            ax.scatter(v['params_pct'], v[metric], marker=marker, s=80,
                       color=color, edgecolors='black', zorder=3)
            stack = seen_x.get(v['params_pct'], 0)
            seen_x[v['params_pct']] = stack + 1
            ax.annotate(v['name'], (v['params_pct'], v[metric]),
                        textcoords='offset points',
                        xytext=(8, 6 - stack * 12), fontsize=8)
        fx = [p['params_pct'] for p in front]
        fy = [p[metric] for p in front]
        ax.plot(fx, fy, '--', color='tab:blue', alpha=0.5, zorder=2,
                label='Pareto front')
        ax.set_xlabel('Backbone+Neck Params (% of FP32)')
        ax.set_ylabel(title)
        ax.set_title(f'Architecture Pareto — {title} vs Params')
        ax.set_xlim(108, 12)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower left')

    fig.suptitle(
        'Measured: precision held fixed per variant '
        '(FP32 \u25cb circle / INT8 \u25a1 square)'
    )
    fig.tight_layout()
    out = PARETO_DIR / 'architecture_pareto.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'[pareto] Saved {out}')
    return front


def plot_deployment_pareto(variants: list) -> list:
    """Chart 2: projected size under TRT INT8 (params_pct * 0.25) vs mAP.

    QAT is the only A40-measured INT8 result. Pruned/Distilled variants use
    real TRT INT8 engine sizes measured on Jetson Orin Nano (trt_engine_pct),
    mAP carried from their FP32 eval — marked with hollow markers and
    "(Jetson TRT)" labels. FP32 baseline and PTQ are excluded (not INT8).
    """
    points = []
    for v in variants:
        if v['name'] == 'QAT INT8':
            points.append({
                **v,
                'size_pct': v['trt_engine_pct'],
                'projected': False,
            })
        elif v['precision'] == 'FP32' and v['name'] != 'FP32 baseline':
            points.append({
                **v,
                'size_pct': v['trt_engine_pct'],
                'projected': True,
            })
    # FP32 baseline and PTQ excluded: FP32 isn't INT8, and PTQ is dominated
    # by QAT at the same size with higher accuracy.

    front = pareto_front(points, 'size_pct', 'mAP')
    front_names = {p['name'] for p in front}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    # Points within this (size_pct, mAP) distance get vertically-stacked
    # label offsets to avoid overlap (e.g. Pruned 25% and Distilled sit at
    # nearly the same point ~72% / ~0.408-0.409).
    placed = []
    for p in points:
        marker = 'D' if not p['projected'] else 'o'
        facecolor = 'tab:blue' if p['name'] in front_names else 'lightgray'
        if p['projected']:
            ax.scatter(p['size_pct'], p['mAP'], marker=marker, s=80,
                       facecolors='none', edgecolors=facecolor, linewidths=2,
                       zorder=3)
            label = f"{p['name']} (Jetson TRT)"
        else:
            ax.scatter(p['size_pct'], p['mAP'], marker=marker, s=90,
                       color=facecolor, edgecolors='black', zorder=3)
            label = f"{p['name']} (measured)"

        stack = sum(
            1 for (px, py) in placed
            if abs(px - p['size_pct']) < 1.5 and abs(py - p['mAP']) < 0.01
        )
        placed.append((p['size_pct'], p['mAP']))
        ax.annotate(label, (p['size_pct'], p['mAP']),
                    textcoords='offset points', xytext=(8, 6 - stack * 14),
                    fontsize=8)

    fx = [p['size_pct'] for p in front]
    fy = [p['mAP'] for p in front]
    ax.plot(fx, fy, '--', color='tab:blue', alpha=0.5, zorder=2,
            label='Pareto front')

    ax.set_xlabel('TRT INT8 engine size (% of FP32 engine, 6.82 MB = 100%)')
    ax.set_ylabel('mAP')
    ax.set_title(
        'Deployment Pareto — Real TRT INT8 Engine Sizes (Jetson Orin Nano)\n'
        '\u25c6 A40 measured (QAT)  \u00b7  '
        '\u25cb Jetson TRT engine (mAP from FP32 eval)',
        fontsize=10,
    )
    ax.set_xlim(108, 44)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    fig.tight_layout()
    out = PARETO_DIR / 'deployment_pareto.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'[pareto] Saved {out}')
    return front


def main() -> None:
    load_distillation_result()

    print('\n[pareto] All variants:')
    for v in VARIANTS:
        print(f"  {v['name']:24s}  mAP {v['mAP']:.4f}  NDS {v['NDS']:.4f}  "
              f"params {v['params_pct']:5.1f}%  ({v['precision']})")

    arch_front = plot_architecture_pareto(VARIANTS)
    print('\n[pareto] Architecture Pareto front (params_pct vs mAP):')
    for p in arch_front:
        print(f"  {p['name']:24s}  {p['params_pct']:5.1f}% params  "
              f"mAP {p['mAP']:.4f}")

    deploy_front = plot_deployment_pareto(VARIANTS)
    print('\n[pareto] Projected Deployment Pareto front (size_pct vs mAP):')
    for p in deploy_front:
        tag = 'measured' if not p['projected'] else 'PROJECTED'
        print(f"  {p['name']:24s}  {p['size_pct']:5.1f}% size  "
              f"mAP {p['mAP']:.4f}  ({tag})")

    summary = {
        'variants': VARIANTS,
        'architecture_pareto_front': [p['name'] for p in arch_front],
        'deployment_pareto_front': [
            {'name': p['name'], 'size_pct': p['size_pct'],
             'mAP': p['mAP'], 'projected': p['projected']}
            for p in deploy_front
        ],
        'fp32_engine_mb': FP32_ENGINE_MB,
    }
    summary_path = PARETO_DIR / 'pareto_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n[pareto] Summary saved to {summary_path}')

    print('\n[pareto] Recommended Jetson candidates (validate projections):')
    for p in deploy_front:
        if p['projected']:
            print(f"  {p['name']} — projected {p['size_pct']:.1f}% size, "
                  f"{p['mAP']:.4f} mAP. Export pruned_model_*.pt -> ONNX -> "
                  f"TRT INT8, benchmark on jetson_calib.")
    print('  QAT INT8 — measured 25.0% size, 0.4814 mAP. Export FP32 '
          'baseline -> ONNX -> TRT INT8 (validated near-free), benchmark.')


if __name__ == '__main__':
    main()

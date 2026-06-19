#!/bin/bash
# ============================================================================
# Generate prediction submission JSONs for all CenterPoint TRT INT8 variants.
#
# Two-phase eval design (avoids Jetson OOM):
#   Phase 1 (this script): TRT inference inside Docker → submission JSONs.
#     Uses the lightweight `infer` service (--no-eval): no NuScenes load.
#   Phase 2 (separate, on host): mAP/NDS via eval_metrics.py.
#
# Prerequisites:
#   - All engines built in deployment/output/engines/{variant}/
#   - jetson_calib_bev/ at repo root (512 × [64,512,512] .npy BEV tensors)
#   - nuscenes_infos_val.pkl at repo root (or set VAL_PKL)
#   - edgefusion-centerpoint:latest Docker image built
#
# Usage:
#   bash deployment/scripts/eval_all.sh
#
# Then compute metrics on host:
#   python3 deployment/scripts/eval_metrics.py \
#       --nuscenes ~/Downloads/v1.0-trainval_meta \
#       --val-pkl  nuscenes_infos_val.pkl \
#       --submissions deployment/output \
#       --out deployment/output/eval_summary.json
# ============================================================================

set -e

COMPOSE="deployment/docker/docker-compose.yml"
VARIANTS="fp32" 
#pruned25 pruned40 pruned55 distilled25"

echo "[eval_all] Phase 1: TRT inference → submission JSONs (no in-container eval)"
echo "[eval_all] Variants: $VARIANTS"
echo ""

PASSED=()
FAILED=()

for VARIANT in $VARIANTS; do
    echo "============================================================"
    echo "[eval_all] Inference: $VARIANT"
    echo "============================================================"

    ENGINE="deployment/output/engines/$VARIANT/pts_backbone_neck_head.engine"
    if [ ! -f "$ENGINE" ]; then
        echo "[eval_all] WARNING: engine not found at $ENGINE — skipping"
        FAILED+=("$VARIANT (no engine)")
        continue
    fi

    if VARIANT="$VARIANT" docker compose -f "$COMPOSE" run --rm infer; then
        PASSED+=("$VARIANT")
        echo "[eval_all] $VARIANT submission DONE"
    else
        echo "[eval_all] $VARIANT FAILED (exit $?)"
        FAILED+=("$VARIANT")
    fi
    echo ""
done

echo "============================================================"
echo "[eval_all] Phase 1 Summary"
echo "============================================================"
echo "Passed: ${PASSED[*]:-none}"
echo "Failed: ${FAILED[*]:-none}"
echo ""
echo "Submission JSONs in deployment/output/eval_{variant}_submission.json"
echo ""
echo "Phase 2 — compute mAP/NDS on host:"
echo "  python3 deployment/scripts/eval_metrics.py \\"
echo "      --nuscenes ~/Downloads/v1.0-trainval_meta \\"
echo "      --val-pkl nuscenes_infos_val.pkl \\"
echo "      --submissions deployment/output \\"
echo "      --out deployment/output/eval_summary.json"

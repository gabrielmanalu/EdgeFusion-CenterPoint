/**
 * center_head_postprocess.h
 *
 * Post-engine CUDA stage for CenterPoint center-head postprocessing.
 * Operates entirely on GPU tensors that remain in their TRT output
 * buffers — no host↔device copies on the hot path.
 *
 * Two kernels, called in sequence after engine.execute_async_v3():
 *
 *   1. launch_peak_finding
 *      Applies sigmoid to the raw heatmap, then extracts local-maximum
 *      cells via 3×3 max-pool NMS (equivalent to mmdet3d _nms_heatmap
 *      with kernel=3).  Cells that equal their neighbourhood maximum
 *      AND exceed score_threshold are written atomically into a flat
 *      peaks list.  All on GPU; output is shared/managed memory so the
 *      Python wrapper can read counts without a memcpy.
 *
 *   2. launch_box_decode
 *      For each surviving peak, reads the regression maps (reg, height,
 *      dim, rot, vel) at the peak's (row, col) using the task associated
 *      with that class channel, and decodes into world-space
 *      (cx, cy, cz, d_len, d_wid, d_hgt, yaw, vx, vy).  Math is
 *      identical to decode_outputs() in deployment/scripts/eval.py.
 *
 * After both kernels, only the small decoded-box array (~20 KB for 500
 * detections) needs to be read on CPU — for circle NMS, which is
 * fast and easier to express in Python on a small array.
 *
 * Parity target  : all outputs must match the numpy reference in
 *                  deployment/scripts/eval.py within TOLERANCE = 1e-3.
 * Parity tests   : deployment/plugins/test_parity.py
 * Python wrapper : deployment/plugins/postprocess_wrapper.py
 *
 * Config assumptions (CenterPoint pillar02 nuScenes):
 *   BEV grid    : 512 × 512   (voxel_size = 0.2 m)
 *   Head stride : 4            (head output = 128 × 128)
 *   Classes     : 10  (car / truck / cv / bus / trailer / barrier /
 *                       motorcycle / bicycle / pedestrian / traffic_cone)
 *   Tasks       : 6   (one task → one or two classes, see CLASS_TO_TASK)
 *   Head outputs: heatmap[10,H,W] reg[12,H,W] height[6,H,W]
 *                 dim[18,H,W] rot[12,H,W] vel[12,H,W]
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace edge_fusion {

// ── Constants ─────────────────────────────────────────────────────────────────

// Default nuScenes / CenterPoint pillar02 config values.
// Override via CenterHeadConfig if your config differs.
static constexpr int   BEV_H            = 128;
static constexpr int   BEV_W            = 128;
static constexpr int   NUM_CLASSES      = 10;
static constexpr int   NUM_TASKS        = 6;
static constexpr float VOXEL_X         = 0.2f;
static constexpr float VOXEL_Y         = 0.2f;
static constexpr float X_MIN           = -51.2f;
static constexpr float Y_MIN           = -51.2f;
static constexpr float SCORE_THRESHOLD = 0.1f;
static constexpr float HEAD_STRIDE     = 4.0f;
static constexpr int   POOL_RADIUS     = 1;     // 3×3 window → radius 1
static constexpr int   MAX_DETECTIONS  = 3000;  // per frame across all classes

// Heatmap channel (= global class index) → task index.
// Derived from TASKS in eval.py:
//   task 0 → car                          (channels  0)
//   task 1 → truck, construction_vehicle  (channels  1, 2)
//   task 2 → bus, trailer                 (channels  3, 4)
//   task 3 → barrier                      (channel   5)
//   task 4 → motorcycle, bicycle          (channels  6, 7)
//   task 5 → pedestrian, traffic_cone     (channels  8, 9)
static constexpr int CLASS_TO_TASK_HOST[NUM_CLASSES] = {
    0, 1, 1, 2, 2, 3, 4, 4, 5, 5
};

// ── Configuration struct ──────────────────────────────────────────────────────

struct CenterHeadConfig {
    int   bev_h          = BEV_H;
    int   bev_w          = BEV_W;
    int   num_classes    = NUM_CLASSES;
    int   max_detections = MAX_DETECTIONS;
    float score_threshold = SCORE_THRESHOLD;
    int   pool_radius    = POOL_RADIUS;    // max-pool half-size (3×3 → 1)
    float voxel_x        = VOXEL_X;
    float voxel_y        = VOXEL_Y;
    float x_min          = X_MIN;
    float y_min          = Y_MIN;
    float head_stride    = HEAD_STRIDE;
};

// ── Kernel launchers ──────────────────────────────────────────────────────────

/**
 * Peak finding: sigmoid + 3×3 max-pool NMS on the heatmap.
 *
 * Each cell (b, c, h, w) is kept as a peak iff:
 *   sigmoid(heatmap[b,c,h,w]) > score_threshold  AND
 *   raw heatmap[b,c,h,w] >= max raw value in its 3×3 neighbourhood
 *   (raw comparison is equivalent to sigmoided since sigmoid is monotone)
 *
 * Surviving peaks are written atomically into peaks_out/scores_out.
 * counts[b] tracks how many peaks were found for batch element b.
 *
 * peaks_out  : [B, max_det, 3]  — (row, col, class_id) per peak
 *              MUST be zeroed before the call (cudaMemsetAsync recommended)
 * scores_out : [B, max_det]     — sigmoid score per peak
 * counts     : [B]              — number of valid peaks; may exceed
 *              max_det if more peaks found (caller should clamp)
 *              MUST be zeroed before the call
 *
 * All pointers must be device or managed memory.
 */
void launch_peak_finding(
    const float*           heatmap,       // [B, C, H, W] raw logits on GPU
    int*                   peaks_out,     // [B, max_det, 3] managed
    float*                 scores_out,    // [B, max_det]    managed
    int*                   counts,        // [B]             managed
    const CenterHeadConfig& cfg,
    int                    B,
    cudaStream_t           stream
);

/**
 * Box decode: convert peaks to world-space boxes.
 *
 * For each peak (row, col, class_id) produced by launch_peak_finding,
 * reads the appropriate task's regression channels and decodes:
 *   cx  = (col + reg_x) * stride * voxel_x + x_min
 *   cy  = (row + reg_y) * stride * voxel_y + y_min
 *   cz  = height at peak
 *   l,w,h = exp(clip(dim, -5, 5))
 *   yaw = atan2(sin_rot, cos_rot)
 *   vx, vy = velocity at peak
 *
 * Output per box (10 floats):
 *   [cx, cy, cz, d_len, d_wid, d_hgt, yaw, vx, vy, 0.0]
 *   Note: nuScenes size = [width, length, height]; the Python wrapper
 *   applies the swap (d_wid, d_len, d_hgt) when building the submission.
 *
 * boxes_out  : [B, max_det, 10] managed — only counts[b] entries valid
 *
 * All pointers must be device or managed memory.
 */
void launch_box_decode(
    const float*           reg,           // [B, 12, H, W]
    const float*           height,        // [B,  6, H, W]
    const float*           dim,           // [B, 18, H, W]
    const float*           rot,           // [B, 12, H, W]
    const float*           vel,           // [B, 12, H, W]
    const int*             peaks,         // [B, max_det, 3]  from peak_finding
    const int*             counts,        // [B]
    float*                 boxes_out,     // [B, max_det, 10] managed
    const CenterHeadConfig& cfg,
    int                    B,
    cudaStream_t           stream
);

// ── C interface for Python ctypes ─────────────────────────────────────────────

extern "C" {

void ef_launch_peak_finding(
    const float* heatmap,
    int*         peaks_out,
    float*       scores_out,
    int*         counts,
    int B, int C, int H, int W,
    int max_det, float score_thr, int pool_radius,
    cudaStream_t stream
);

void ef_launch_box_decode(
    const float* reg,
    const float* height,
    const float* dim,
    const float* rot,
    const float* vel,
    const int*   peaks,
    const int*   counts,
    float*       boxes_out,
    int B, int H, int W, int max_det,
    float voxel_x, float voxel_y,
    float x_min, float y_min, float head_stride,
    cudaStream_t stream
);

}  // extern "C"

}  // namespace edge_fusion
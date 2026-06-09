/**
 * center_head_postprocess.h
 *
 * Custom CUDA / TensorRT plugin for CenterPoint center-head postprocessing:
 *   1. Max-pool peak finding on the heatmap (NMS-free peak extraction)
 *   2. Box decoding from regression maps at detected peak locations
 *   3. Score thresholding
 *
 * Operates entirely on GPU tensors to avoid host-device copies on the hot path.
 * Numerical parity against the PyTorch reference is verified in test_parity.py.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace edge_fusion {

// ── Configuration ────────────────────────────────────────────────────────────

struct CenterHeadConfig {
    int   bev_h;            // BEV grid height  (e.g. 512)
    int   bev_w;            // BEV grid width   (e.g. 512)
    int   num_classes;      // 10 for nuScenes
    int   max_detections;   // cap on output boxes per forward pass
    float score_threshold;
    int   pool_radius;      // max-pool window half-size for peak finding
    float voxel_x;          // voxel size (m) along X
    float voxel_y;          // voxel size (m) along Y
    float x_min;            // point-cloud range x_min (m)
    float y_min;            // point-cloud range y_min (m)
};

// ── Kernel launchers ─────────────────────────────────────────────────────────

/**
 * Extract peak locations from the heatmap via max-pool NMS.
 * Writes a flat list of (row, col, class_id) to peaks_out.
 */
void launch_peak_finding(
    const float*           heatmap,     // [B, C, H, W] on GPU
    int*                   peaks_out,   // [B, max_det, 3]
    float*                 scores_out,  // [B, max_det]
    const CenterHeadConfig& cfg,
    cudaStream_t           stream
);

/**
 * Decode box parameters from regression maps at detected peak locations.
 * Output format per box: [x, y, z, log_l, log_w, log_h, sin(θ), cos(θ), vx, vy]
 */
void launch_box_decode(
    const float*           reg_maps,    // [B, 8, H, W]
    const int*             peaks,       // [B, max_det, 3]
    float*                 boxes_out,   // [B, max_det, 10]
    const CenterHeadConfig& cfg,
    cudaStream_t           stream
);

}  // namespace edge_fusion

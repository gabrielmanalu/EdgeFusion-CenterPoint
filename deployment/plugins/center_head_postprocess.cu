/**
 * center_head_postprocess.cu
 *
 * CUDA kernels for CenterPoint center-head postprocessing.
 * See center_head_postprocess.h for the interface.
 *
 * Parity target : outputs must match PyTorch reference within 1e-3 abs tol.
 * Parity test   : deployment/plugins/test_parity.py
 */

#include "center_head_postprocess.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace edge_fusion {

// ── Peak-finding kernel ───────────────────────────────────────────────────────

__global__ void peak_finding_kernel(
    const float* __restrict__ heatmap,
    int*         __restrict__ peaks_out,
    float*       __restrict__ scores_out,
    int B, int C, int H, int W,
    int max_det, float score_thr, int radius
) {
    // TODO: each thread handles one (b, c, h, w) location.
    //       Check if local max in (2*radius+1)^2 window AND score > score_thr.
    //       Atomically append (h, w, c) to peaks_out for that batch element.
}


void launch_peak_finding(
    const float*           heatmap,
    int*                   peaks_out,
    float*                 scores_out,
    const CenterHeadConfig& cfg,
    cudaStream_t           stream
) {
    // TODO: compute grid / block dims → launch peak_finding_kernel
    (void)heatmap; (void)peaks_out; (void)scores_out;
    (void)cfg; (void)stream;
}


// ── Box-decode kernel ─────────────────────────────────────────────────────────

__global__ void box_decode_kernel(
    const float* __restrict__ reg_maps,
    const int*   __restrict__ peaks,
    float*       __restrict__ boxes_out,
    int B, int H, int W, int max_det,
    float voxel_x, float voxel_y,
    float x_min,   float y_min
) {
    // TODO: for each (batch, detection) thread:
    //   row = peaks[b, i, 0];  col = peaks[b, i, 1]
    //   x = voxel_x * (col + reg_maps[b, 0, row, col]) + x_min
    //   y = voxel_y * (row + reg_maps[b, 1, row, col]) + y_min
    //   z = reg_maps[b, 2, row, col]
    //   l, w, h = exp(reg_maps[b, 3..5, ...])
    //   sin_theta, cos_theta = reg_maps[b, 6..7, ...]
    //   vx, vy = reg_maps[b, 8..9, ...]  (if velocity head present)
    //   write [x, y, z, l, w, h, sin, cos, vx, vy] to boxes_out
}


void launch_box_decode(
    const float*           reg_maps,
    const int*             peaks,
    float*                 boxes_out,
    const CenterHeadConfig& cfg,
    cudaStream_t           stream
) {
    // TODO: compute grid / block dims → launch box_decode_kernel
    (void)reg_maps; (void)peaks; (void)boxes_out;
    (void)cfg; (void)stream;
}

}  // namespace edge_fusion

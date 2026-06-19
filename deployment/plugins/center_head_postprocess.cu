/**
 * center_head_postprocess.cu
 *
 * CUDA kernels for CenterPoint center-head postprocessing.
 * See center_head_postprocess.h for the interface and design rationale.
 *
 * Parity target : peak locations and decoded box coordinates must match
 *                 the numpy reference in deployment/scripts/eval.py
 *                 within TOLERANCE = 1e-3 absolute.
 * Parity tests  : deployment/plugins/test_parity.py
 *
 * Build (Jetson Orin Nano, sm_87):
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make -j4
 */

#include "center_head_postprocess.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

namespace edge_fusion {

// ── Device constants ──────────────────────────────────────────────────────────

// Heatmap channel (= global class index) → task index.
// Loaded into L1 constant cache for repeated read-only access.
//   task 0 → car              (channel 0)
//   task 1 → truck, cv        (channels 1, 2)
//   task 2 → bus, trailer     (channels 3, 4)
//   task 3 → barrier          (channel 5)
//   task 4 → motorcycle, bike (channels 6, 7)
//   task 5 → pedestrian, cone (channels 8, 9)
__constant__ int DEV_CLASS_TO_TASK[10] = {0, 1, 1, 2, 2, 3, 4, 4, 5, 5};

// ── Kernel 1: Peak finding ────────────────────────────────────────────────────

/**
 * One thread per (h, w) cell per class channel per batch element.
 *
 * Thread layout:
 *   b   = blockIdx.z
 *   c   = blockIdx.y
 *   hw  = blockIdx.x * blockDim.x + threadIdx.x  (h*W + w)
 *
 * A cell is a peak iff:
 *   (a) sigmoid(raw_score) > score_thr
 *   (b) raw_score >= raw score of every neighbour in the (2*radius+1)^2
 *       window — raw comparison is equivalent to sigmoided since sigmoid
 *       is strictly monotone, and avoids expf() calls for neighbours.
 *
 * Fast-reject shortcut: if raw_score <= logit(score_thr) the sigmoid
 * is guaranteed to be <= score_thr, so we skip those cells cheaply
 * before the neighbourhood scan.
 *
 * Surviving peaks are written atomically; counts[b] is incremented.
 * If counts[b] would exceed max_det the peak is silently dropped.
 */
__global__ void peak_finding_kernel(
    const float* __restrict__ heatmap,   // [B, C, H, W] raw logits on GPU
    int*   __restrict__ peaks_out,       // [B, max_det, 3] (row, col, cls)
    float* __restrict__ scores_out,      // [B, max_det]
    int*   __restrict__ counts,          // [B] atomic counter
    int B, int C, int H, int W,
    int max_det, float score_thr, int radius
) {
    const int b  = blockIdx.z;
    const int c  = blockIdx.y;
    const int hw = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || c >= C || hw >= H * W) return;

    const int h = hw / W;
    const int w = hw % W;

    // Raw score at this cell
    const int   base       = b * C * H * W + c * H * W;
    const float center_raw = heatmap[base + h * W + w];

    // Fast reject using logit(score_thr):  sigmoid(x) > thr iff x > logit(thr)
    const float raw_thr = logf(score_thr / (1.0f - score_thr + 1e-9f));
    if (center_raw <= raw_thr) return;

    // Neighbourhood max — compared in raw space (monotone, no extra expf)
    float local_max = center_raw;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dy == 0 && dx == 0) continue;
            const int nh = h + dy;
            const int nw = w + dx;
            if (nh < 0 || nh >= H || nw < 0 || nw >= W) continue;
            const float nraw = heatmap[base + nh * W + nw];
            if (nraw > local_max) local_max = nraw;
        }
    }

    // Not a local maximum — skip
    if (center_raw < local_max) return;

    // Compute sigmoid only for genuine peaks
    const float score = 1.0f / (1.0f + expf(-center_raw));

    // Atomically claim a slot
    const int slot = atomicAdd(&counts[b], 1);
    if (slot >= max_det) return;

    const int pk_base = b * max_det;
    peaks_out[pk_base * 3 + slot * 3 + 0] = h;
    peaks_out[pk_base * 3 + slot * 3 + 1] = w;
    peaks_out[pk_base * 3 + slot * 3 + 2] = c;
    scores_out[pk_base + slot]             = score;
}


void launch_peak_finding(
    const float*           heatmap,
    int*                   peaks_out,
    float*                 scores_out,
    int*                   counts,
    const CenterHeadConfig& cfg,
    int                    B,
    cudaStream_t           stream
) {
    const int HW    = cfg.bev_h * cfg.bev_w;
    const int BLOCK = 256;
    const int GRID_X = (HW + BLOCK - 1) / BLOCK;

    dim3 grid(GRID_X, cfg.num_classes, B);
    dim3 block(BLOCK, 1, 1);

    peak_finding_kernel<<<grid, block, 0, stream>>>(
        heatmap, peaks_out, scores_out, counts,
        B, cfg.num_classes, cfg.bev_h, cfg.bev_w,
        cfg.max_detections, cfg.score_threshold, cfg.pool_radius
    );
}


// ── Kernel 2: Box decode ──────────────────────────────────────────────────────

/**
 * One thread per surviving peak per batch element.
 *
 * Thread layout:
 *   b       = blockIdx.y
 *   det_idx = blockIdx.x * blockDim.x + threadIdx.x
 *
 * Decode math (exactly matching decode_outputs() in eval.py):
 *   cx  = (col + reg[t*2,   row,col]) * stride * voxel_x + x_min
 *   cy  = (row + reg[t*2+1, row,col]) * stride * voxel_y + y_min
 *   cz  = height[t, row, col]
 *   l,w,h = exp(clip(dim[t*3+{0,1,2}, row,col], -5, 5))
 *   yaw = atan2(rot[t*2, row,col], rot[t*2+1, row,col])
 *   vx  = vel[t*2,   row, col]
 *   vy  = vel[t*2+1, row, col]
 *   where t = CLASS_TO_TASK[cls]
 *
 * Output per box (10 floats):
 *   [cx, cy, cz, d_len, d_wid, d_hgt, yaw, vx, vy, 0]
 *
 * NOTE: nuScenes size = [width, length, height].  This kernel outputs
 * (d_len, d_wid, d_hgt) = (length, width, height) — the Python wrapper
 * in postprocess_wrapper.py applies the (d_wid, d_len, d_hgt) swap when
 * building the submission JSON, exactly as eval.py does.
 */
__global__ void box_decode_kernel(
    const float* __restrict__ reg,        // [B, 12, H, W]
    const float* __restrict__ height_map, // [B,  6, H, W]
    const float* __restrict__ dim,        // [B, 18, H, W]
    const float* __restrict__ rot,        // [B, 12, H, W]
    const float* __restrict__ vel,        // [B, 12, H, W]
    const int*   __restrict__ peaks,      // [B, max_det, 3]
    const int*   __restrict__ counts,     // [B]
    float*       __restrict__ boxes_out,  // [B, max_det, 10]
    int B, int H, int W, int max_det,
    float voxel_x, float voxel_y,
    float x_min, float y_min, float stride
) {
    const int b       = blockIdx.y;
    const int det_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B) return;
    const int n_peaks = min(counts[b], max_det);
    if (det_idx >= n_peaks) return;

    // Read peak (row, col, class_id)
    const int pk_base = b * max_det;
    const int row = peaks[pk_base * 3 + det_idx * 3 + 0];
    const int col = peaks[pk_base * 3 + det_idx * 3 + 1];
    const int cls = peaks[pk_base * 3 + det_idx * 3 + 2];
    const int t   = DEV_CLASS_TO_TASK[cls];

    const int HW = H * W;
    const int rc = row * W + col;  // flat spatial index

    // ── reg [B, 12, H, W]: task t → channels t*2, t*2+1 ─────────────────
    const float reg_x = reg[b * 12 * HW + (t * 2 + 0) * HW + rc];
    const float reg_y = reg[b * 12 * HW + (t * 2 + 1) * HW + rc];

    const float cx = (col + reg_x) * stride * voxel_x + x_min;
    const float cy = (row + reg_y) * stride * voxel_y + y_min;

    // ── height [B, 6, H, W]: task t → channel t ──────────────────────────
    const float cz = height_map[b * 6 * HW + t * HW + rc];

    // ── dim [B, 18, H, W]: task t → channels t*3, t*3+1, t*3+2 ──────────
    const float raw_l = dim[b * 18 * HW + (t * 3 + 0) * HW + rc];
    const float raw_w = dim[b * 18 * HW + (t * 3 + 1) * HW + rc];
    const float raw_h = dim[b * 18 * HW + (t * 3 + 2) * HW + rc];
    const float d_len = expf(fminf(fmaxf(raw_l, -5.0f), 5.0f));
    const float d_wid = expf(fminf(fmaxf(raw_w, -5.0f), 5.0f));
    const float d_hgt = expf(fminf(fmaxf(raw_h, -5.0f), 5.0f));

    // ── rot [B, 12, H, W]: task t → channels t*2 (sin), t*2+1 (cos) ─────
    const float sin_yaw = rot[b * 12 * HW + (t * 2 + 0) * HW + rc];
    const float cos_yaw = rot[b * 12 * HW + (t * 2 + 1) * HW + rc];
    const float yaw     = atan2f(sin_yaw, cos_yaw);

    // ── vel [B, 12, H, W]: task t → channels t*2 (vx), t*2+1 (vy) ───────
    const float vx = vel[b * 12 * HW + (t * 2 + 0) * HW + rc];
    const float vy = vel[b * 12 * HW + (t * 2 + 1) * HW + rc];

    // Write output
    const int out_base = (b * max_det + det_idx) * 10;
    boxes_out[out_base + 0] = cx;
    boxes_out[out_base + 1] = cy;
    boxes_out[out_base + 2] = cz;
    boxes_out[out_base + 3] = d_len;
    boxes_out[out_base + 4] = d_wid;
    boxes_out[out_base + 5] = d_hgt;
    boxes_out[out_base + 6] = yaw;
    boxes_out[out_base + 7] = vx;
    boxes_out[out_base + 8] = vy;
    boxes_out[out_base + 9] = 0.0f;
}


void launch_box_decode(
    const float*           reg,
    const float*           height,
    const float*           dim,
    const float*           rot,
    const float*           vel,
    const int*             peaks,
    const int*             counts,
    float*                 boxes_out,
    const CenterHeadConfig& cfg,
    int                    B,
    cudaStream_t           stream
) {
    // Launch max_det threads per batch element; each checks counts[b] itself.
    const int BLOCK  = 128;
    const int GRID_X = (cfg.max_detections + BLOCK - 1) / BLOCK;

    dim3 grid(GRID_X, B, 1);
    dim3 block(BLOCK, 1, 1);

    box_decode_kernel<<<grid, block, 0, stream>>>(
        reg, height, dim, rot, vel,
        peaks, counts, boxes_out,
        B, cfg.bev_h, cfg.bev_w, cfg.max_detections,
        cfg.voxel_x, cfg.voxel_y,
        cfg.x_min, cfg.y_min, cfg.head_stride
    );
}

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
) {
    CenterHeadConfig cfg;
    cfg.bev_h           = H;
    cfg.bev_w           = W;
    cfg.num_classes     = C;
    cfg.max_detections  = max_det;
    cfg.score_threshold = score_thr;
    cfg.pool_radius     = pool_radius;
    launch_peak_finding(heatmap, peaks_out, scores_out, counts, cfg, B, stream);
}

void ef_launch_box_decode(
    const float* reg,
    const float* height_map,
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
) {
    CenterHeadConfig cfg;
    cfg.bev_h          = H;
    cfg.bev_w          = W;
    cfg.max_detections = max_det;
    cfg.voxel_x        = voxel_x;
    cfg.voxel_y        = voxel_y;
    cfg.x_min          = x_min;
    cfg.y_min          = y_min;
    cfg.head_stride    = head_stride;
    launch_box_decode(reg, height_map, dim, rot, vel,
                      peaks, counts, boxes_out, cfg, B, stream);
}

}  // extern "C"

}  // namespace edge_fusion

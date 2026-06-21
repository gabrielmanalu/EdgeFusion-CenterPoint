/**
 * centerpoint_trt.hpp — full online inference pipeline for EdgeFusion-CenterPoint.
 *
 * Pipeline per frame:
 *   1. CPU voxelize   → voxel_features [max_vox, max_pts, 9]
 *   2. Encoder TRT    → pillar_features [max_vox, 64]
 *   3. CPU scatter    → bev_buf_ [1, 64, 512, 512]   (htod after scatter)
 *   4. Backbone TRT   → heatmap + reg + height + dim + rot + vel  (stay on GPU)
 *   5. CUDA postproc  → peaks → boxes  (managed memory, no extra copy)
 *   6. CPU circle NMS → final detections
 */
#pragma once

#include <cuda_runtime.h>
#include <NvInfer.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "voxelizer.hpp"
#include "center_head_postprocess.h"   // from deployment/plugins/

namespace edge_fusion {

// ── Detection result ─────────────────────────────────────────────────────────

struct Detection3D {
    float cx, cy, cz;
    float length, width, height;
    float yaw;
    float vx, vy;
    float score;
    int   class_id;
};

// ── Config ────────────────────────────────────────────────────────────────────

struct CenterpointConfig {
    std::string encoder_engine_path;
    std::string backbone_engine_path;
    std::string postproc_so_path;

    float score_threshold = 0.35f;
    float nms_radius      = 2.0f;
    int   max_detections  = 500;

    VoxelizerConfig  vox_cfg;
    CenterHeadConfig head_cfg;
    int pillar_feat_dim = 64;
};

// ── TRT logger ───────────────────────────────────────────────────────────────

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// ── Main inference class ──────────────────────────────────────────────────────

class CenterpointTRT {
public:
    // Public data struct — used by free helper find_head() in the .cpp.
    struct HeadBuf { std::string name; float* ptr = nullptr; size_t bytes = 0; };

    explicit CenterpointTRT(const CenterpointConfig& cfg);
    ~CenterpointTRT();

    CenterpointTRT(const CenterpointTRT&) = delete;
    CenterpointTRT& operator=(const CenterpointTRT&) = delete;

    void detect(const float* points, int n_pts,
                std::vector<Detection3D>& out);

private:
    void load_engines();
    void infer_encoder(int n_voxels);
    void scatter_pillars(int n_voxels);
    void infer_backbone();
    void load_postproc_so();
    void run_postproc();
    void circle_nms(std::vector<Detection3D>& dets, float radius) const;

    CenterpointConfig  cfg_;
    Voxelizer          vox_;
    TRTLogger          logger_;
    cudaStream_t       stream_{nullptr};

    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>       enc_engine_, bb_engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> enc_ctx_,    bb_ctx_;

    // Encoder path
    std::vector<float> vox_features_h_;
    std::vector<int>   vox_coords_h_;
    std::vector<int>   num_points_h_;
    float* vox_features_d_ = nullptr;
    float* pillar_feat_d_  = nullptr;

    // Backbone path — note: bev_buf_ is the HOST buffer; bev_h_ is the height INT.
    std::vector<float> bev_buf_;   // [1, bev_c_, bev_h_, bev_w_]  host
    float* bev_d_ = nullptr;

    std::vector<HeadBuf> head_bufs_d_;

    // BEV spatial dimensions (read from backbone engine bindings at load time).
    int bev_c_ = 64, bev_h_ = 512, bev_w_ = 512;

    // dlopen handles
    void* postproc_so_handle_ = nullptr;

    using PeakFindingFn = void(*)(
        const float*, int*, float*, int*, int, int, int, int,
        int, float, int, cudaStream_t);
    using BoxDecodeFn = void(*)(
        const float*, const float*, const float*, const float*, const float*,
        const int*, const int*, float*, int, int, int, int,
        float, float, float, float, float, cudaStream_t);

    PeakFindingFn fn_peak_finding_ = nullptr;
    BoxDecodeFn   fn_box_decode_   = nullptr;

    // CUDA-managed postproc buffers
    int*   peaks_m_  = nullptr;
    float* scores_m_ = nullptr;
    int*   counts_m_ = nullptr;
    float* boxes_m_  = nullptr;
};

}  // namespace edge_fusion
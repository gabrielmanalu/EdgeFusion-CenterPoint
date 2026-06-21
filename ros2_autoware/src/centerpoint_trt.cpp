/**
 * centerpoint_trt.cpp — full online inference pipeline.
 *
 * Engine I/O conventions (auto-enumerated from TRT bindings at load time):
 *
 * pts_voxel_encoder.engine
 *   INPUT  "voxels"      [max_vox, max_pts, 9]  float32
 *   OUTPUT "pillar_feat" [max_vox, 64]           float32
 *
 * pts_backbone_neck_head.engine
 *   INPUT  "bev_input"   [1, 64, 512, 512]       float32
 *   OUTPUT "heatmap"     [1, 10, 128, 128]        float32
 *   OUTPUT "reg"         [1, 12, 128, 128]        float32
 *   OUTPUT "height"      [1,  6, 128, 128]        float32
 *   OUTPUT "dim"         [1, 18, 128, 128]        float32
 *   OUTPUT "rot"         [1, 12, 128, 128]        float32
 *   OUTPUT "vel"         [1, 12, 128, 128]        float32
 *
 * Actual binding names are enumerated from the engine; the code matches by
 * substring (case-insensitive). If your export uses different names, update
 * find_binding() below.
 */
#include "centerpoint_trt.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#include <rclcpp/logging.hpp>

// Convenience macro
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t e = (call);                                           \
        if (e != cudaSuccess)                                             \
            throw std::runtime_error(std::string("CUDA error: ")         \
                                     + cudaGetErrorString(e)              \
                                     + " at " __FILE__ ":" + std::to_string(__LINE__));\
    } while (false)

namespace edge_fusion {

// ── TRT logger ───────────────────────────────────────────────────────────────

void TRTLogger::log(Severity sev, const char* msg) noexcept {
    if (sev == Severity::kERROR || sev == Severity::kINTERNAL_ERROR)
        RCLCPP_ERROR(rclcpp::get_logger("trt"), "%s", msg);
    else if (sev == Severity::kWARNING)
        RCLCPP_WARN(rclcpp::get_logger("trt"), "%s", msg);
}

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("Cannot open engine: " + path);
    auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    return buf;
}

static size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
    return v;
}

static bool name_contains(const std::string& name, const std::string& sub) {
    std::string a = name, b = sub;
    std::transform(a.begin(), a.end(), a.begin(), ::tolower);
    std::transform(b.begin(), b.end(), b.begin(), ::tolower);
    return a.find(b) != std::string::npos;
}

// Fill only the dynamic (-1) dimensions of engine_dims from the values list.
// Static dimensions are kept as-is; mismatching them causes a TRT API error.
static nvinfer1::Dims resolve_dynamic(const nvinfer1::Dims& engine_dims,
                                       std::initializer_list<int64_t> fill) {
    nvinfer1::Dims out = engine_dims;
    auto it = fill.begin();
    for (int i = 0; i < out.nbDims && it != fill.end(); ++i) {
        if (out.d[i] < 0)
            out.d[i] = *it++;
    }
    return out;
}

// ── Constructor ──────────────────────────────────────────────────────────────

CenterpointTRT::CenterpointTRT(const CenterpointConfig& cfg)
    : cfg_(cfg), vox_(cfg.vox_cfg)
{
    CUDA_CHECK(cudaStreamCreate(&stream_));
    load_engines();
    load_postproc_so();
    RCLCPP_INFO(rclcpp::get_logger("centerpoint_trt"),
                "CenterpointTRT ready (BEV %d×%d×%d).", bev_c_, bev_h_, bev_w_);
}

CenterpointTRT::~CenterpointTRT() {
    if (peaks_m_)   cudaFree(peaks_m_);
    if (scores_m_)  cudaFree(scores_m_);
    if (counts_m_)  cudaFree(counts_m_);
    if (boxes_m_)   cudaFree(boxes_m_);

    if (vox_features_d_) cudaFree(vox_features_d_);
    if (pillar_feat_d_)  cudaFree(pillar_feat_d_);
    if (bev_d_)          cudaFree(bev_d_);

    for (auto& b : head_bufs_d_)
        if (b.ptr) cudaFree(b.ptr);

    if (stream_)         cudaStreamDestroy(stream_);
    if (postproc_so_handle_) dlclose(postproc_so_handle_);
}

// ── Engine loading ────────────────────────────────────────────────────────────

void CenterpointTRT::load_engines() {
    runtime_.reset(nvinfer1::createInferRuntime(logger_));

    // ─ Encoder ─
    {
        auto buf = read_file(cfg_.encoder_engine_path);
        enc_engine_.reset(runtime_->deserializeCudaEngine(buf.data(), buf.size()));
        if (!enc_engine_)
            throw std::runtime_error("Failed to deserialize encoder engine.");
        enc_ctx_.reset(enc_engine_->createExecutionContext());

        const auto& vc = cfg_.vox_cfg;
        int n = enc_engine_->getNbIOTensors();

        // Pass 1 — set input shape so TRT resolves any dynamic (-1) dimensions.
        // Only fill the dynamic dims; keep static dims from the engine as-is.
        for (int i = 0; i < n; ++i) {
            const char* name = enc_engine_->getIOTensorName(i);
            if (enc_engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                auto e_dims = enc_engine_->getTensorShape(name);
                auto resolved = resolve_dynamic(e_dims, {vc.max_voxels});
                enc_ctx_->setInputShape(name, resolved);
            }
        }

        // Pass 2 — allocate from resolved context shapes (never -1).
        for (int i = 0; i < n; ++i) {
            const char* name = enc_engine_->getIOTensorName(i);
            auto mode  = enc_engine_->getTensorIOMode(name);
            auto dims  = enc_ctx_->getTensorShape(name);   // resolved, no -1
            size_t bytes = volume(dims) * sizeof(float);

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                vox_features_h_.resize(volume(dims), 0.f);
                CUDA_CHECK(cudaMalloc(&vox_features_d_, bytes));
                enc_ctx_->setTensorAddress(name, vox_features_d_);
            } else {
                CUDA_CHECK(cudaMalloc(&pillar_feat_d_, bytes));
                enc_ctx_->setTensorAddress(name, pillar_feat_d_);
            }
        }
    }

    // ─ Backbone ─
    {
        auto buf = read_file(cfg_.backbone_engine_path);
        bb_engine_.reset(runtime_->deserializeCudaEngine(buf.data(), buf.size()));
        if (!bb_engine_)
            throw std::runtime_error("Failed to deserialize backbone engine.");
        bb_ctx_.reset(bb_engine_->createExecutionContext());

        const int gx = vox_.grid_x();
        const int gy = vox_.grid_y();
        int n = bb_engine_->getNbIOTensors();

        // Pass 1 — set backbone input shape: only dynamic dims filled, static kept.
        for (int i = 0; i < n; ++i) {
            const char* name = bb_engine_->getIOTensorName(i);
            if (bb_engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                auto e_dims = bb_engine_->getTensorShape(name);
                auto resolved = resolve_dynamic(e_dims, {1, cfg_.pillar_feat_dim, gy, gx});
                bb_ctx_->setInputShape(name, resolved);
            }
        }

        // Pass 2 — allocate from resolved shapes.
        for (int i = 0; i < n; ++i) {
            const char* name = bb_engine_->getIOTensorName(i);
            auto mode  = bb_engine_->getTensorIOMode(name);
            auto dims  = bb_ctx_->getTensorShape(name);   // resolved, no -1
            size_t bytes = volume(dims) * sizeof(float);

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                bev_c_ = static_cast<int>(dims.d[1]);
                bev_h_ = static_cast<int>(dims.d[2]);
                bev_w_ = static_cast<int>(dims.d[3]);
                bev_buf_.assign(static_cast<size_t>(bev_c_) * bev_h_ * bev_w_, 0.f);
                CUDA_CHECK(cudaMalloc(&bev_d_, bytes));
                bb_ctx_->setTensorAddress(name, bev_d_);
            } else {
                float* ptr = nullptr;
                CUDA_CHECK(cudaMalloc(&ptr, bytes));
                bb_ctx_->setTensorAddress(name, ptr);
                head_bufs_d_.push_back({name, ptr, bytes});
            }
        }
    }

    // Host-side voxelizer buffers.
    const int mv = cfg_.vox_cfg.max_voxels;
    vox_coords_h_.assign(mv * 4, 0);
    num_points_h_.assign(mv, 0);

    // CUDA-managed postproc buffers.
    const int md = cfg_.max_detections;
    CUDA_CHECK(cudaMallocManaged(&peaks_m_,  md * 3 * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&scores_m_, md * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&counts_m_, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&boxes_m_,  md * 10 * sizeof(float)));
}

// ── Postproc .so loading ──────────────────────────────────────────────────────

void CenterpointTRT::load_postproc_so() {
    postproc_so_handle_ = dlopen(cfg_.postproc_so_path.c_str(), RTLD_LAZY);
    if (!postproc_so_handle_)
        throw std::runtime_error(
            "dlopen failed for " + cfg_.postproc_so_path + ": " + dlerror());

    fn_peak_finding_ = reinterpret_cast<PeakFindingFn>(
        dlsym(postproc_so_handle_, "ef_launch_peak_finding"));
    fn_box_decode_ = reinterpret_cast<BoxDecodeFn>(
        dlsym(postproc_so_handle_, "ef_launch_box_decode"));

    if (!fn_peak_finding_ || !fn_box_decode_)
        throw std::runtime_error("dlsym failed: " + std::string(dlerror()));
}

// ── Inference helpers ─────────────────────────────────────────────────────────

void CenterpointTRT::infer_encoder(int n_voxels) {
    // htod: only the valid portion of voxel_features.
    const int mv     = cfg_.vox_cfg.max_voxels;
    const int mp     = cfg_.vox_cfg.max_pts;
    const int of     = cfg_.vox_cfg.out_features;
    const size_t bytes = static_cast<size_t>(mv) * mp * of * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(vox_features_d_, vox_features_h_.data(),
                               bytes, cudaMemcpyHostToDevice, stream_));
    if (!enc_ctx_->enqueueV3(stream_))
        throw std::runtime_error("Encoder enqueueV3 failed.");
}

void CenterpointTRT::scatter_pillars(int n_voxels) {
    // dtoh pillar features from encoder output.
    const size_t pf_bytes =
        static_cast<size_t>(cfg_.vox_cfg.max_voxels)
        * cfg_.pillar_feat_dim * sizeof(float);
    std::vector<float> pf(cfg_.vox_cfg.max_voxels * cfg_.pillar_feat_dim, 0.f);
    CUDA_CHECK(cudaMemcpyAsync(pf.data(), pillar_feat_d_,
                               pf_bytes, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Scatter into BEV [1, C, H, W].
    std::fill(bev_buf_.begin(), bev_buf_.end(), 0.f);
    const int C = bev_c_, H = bev_h_, W = bev_w_;

    for (int v = 0; v < n_voxels; ++v) {
        const int iy = vox_coords_h_[v * 4 + 2];   // y grid index
        const int ix = vox_coords_h_[v * 4 + 3];   // x grid index
        if (ix < 0 || ix >= W || iy < 0 || iy >= H) continue;

        const float* src = pf.data() + v * C;
        for (int c = 0; c < C; ++c)
            bev_buf_[c * H * W + iy * W + ix] = src[c];
    }

    // htod BEV
    const size_t bev_bytes =
        static_cast<size_t>(C) * H * W * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(bev_d_, bev_buf_.data(),
                               bev_bytes, cudaMemcpyHostToDevice, stream_));
}

void CenterpointTRT::infer_backbone() {
    if (!bb_ctx_->enqueueV3(stream_))
        throw std::runtime_error("Backbone enqueueV3 failed.");
}

// ── Find head tensors by name ─────────────────────────────────────────────────

static float* find_head(const std::vector<CenterpointTRT::HeadBuf>& bufs,
                         const std::string& key) {
    for (const auto& b : bufs)
        if (name_contains(b.name, key)) return b.ptr;
    return nullptr;
}

void CenterpointTRT::run_postproc() {
    const int  md  = cfg_.max_detections;
    const int  H   = cfg_.head_cfg.bev_h;
    const int  W   = cfg_.head_cfg.bev_w;
    const int  C   = cfg_.head_cfg.num_classes;
    const auto& hc = cfg_.head_cfg;

    // Zero managed counters before kernel launch.
    CUDA_CHECK(cudaMemsetAsync(peaks_m_,  0, md * 3 * sizeof(int),  stream_));
    CUDA_CHECK(cudaMemsetAsync(scores_m_, 0, md * sizeof(float),     stream_));
    CUDA_CHECK(cudaMemsetAsync(counts_m_, 0, sizeof(int),            stream_));

    const float* heatmap = find_head(head_bufs_d_, "heatmap");
    const float* reg     = find_head(head_bufs_d_, "reg");
    const float* height  = find_head(head_bufs_d_, "height");
    const float* dim     = find_head(head_bufs_d_, "dim");
    const float* rot     = find_head(head_bufs_d_, "rot");
    const float* vel     = find_head(head_bufs_d_, "vel");

    if (!heatmap || !reg || !height || !dim || !rot || !vel)
        throw std::runtime_error(
            "Could not find all head tensors. Check backbone output names.");

    fn_peak_finding_(heatmap, peaks_m_, scores_m_, counts_m_,
                     /*B=*/1, C, H, W,
                     md, hc.score_threshold, hc.pool_radius, stream_);

    fn_box_decode_(reg, height, dim, rot, vel,
                   peaks_m_, counts_m_, boxes_m_,
                   /*B=*/1, H, W, md,
                   hc.voxel_x, hc.voxel_y,
                   hc.x_min, hc.y_min, hc.head_stride, stream_);

    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ── Circle NMS ───────────────────────────────────────────────────────────────

void CenterpointTRT::circle_nms(std::vector<Detection3D>& dets,
                                  float radius) const {
    // Sort descending by score (inplace).
    std::stable_sort(dets.begin(), dets.end(),
                     [](const Detection3D& a, const Detection3D& b) {
                         return a.score > b.score;
                     });

    const float r2 = radius * radius;
    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection3D> keep;
    keep.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            float dx = dets[i].cx - dets[j].cx;
            float dy = dets[i].cy - dets[j].cy;
            if (dx * dx + dy * dy < r2)
                suppressed[j] = true;
        }
    }
    dets = std::move(keep);
}

// ── Main detect() ─────────────────────────────────────────────────────────────

void CenterpointTRT::detect(const float* points, int n_pts,
                              std::vector<Detection3D>& out) {
    out.clear();

    const int mv = cfg_.vox_cfg.max_voxels;
    const int mp = cfg_.vox_cfg.max_pts;
    const int of = cfg_.vox_cfg.out_features;

    // 1. CPU voxelize.
    std::fill(vox_features_h_.begin(), vox_features_h_.end(), 0.f);
    std::fill(vox_coords_h_.begin(),   vox_coords_h_.end(),   0);
    std::fill(num_points_h_.begin(),   num_points_h_.end(),   0);

    int n_voxels = vox_.voxelize(points, n_pts,
                                  vox_features_h_.data(),
                                  vox_coords_h_.data(),
                                  num_points_h_.data());
    if (n_voxels == 0) return;

    // 2. Encoder TRT.
    infer_encoder(n_voxels);

    // 3. CPU pillar scatter → htod BEV.
    scatter_pillars(n_voxels);

    // 4. Backbone TRT (head tensors stay on GPU).
    infer_backbone();

    // 5. CUDA postproc (managed memory).
    run_postproc();

    // 6. Collect detections from managed memory.
    const int n_det = std::min(*counts_m_, cfg_.max_detections);
    out.reserve(n_det);

    for (int i = 0; i < n_det; ++i) {
        const int*   pk = peaks_m_ + i * 3;
        const float* bx = boxes_m_ + i * 10;

        Detection3D d;
        d.cx     = bx[0];
        d.cy     = bx[1];
        d.cz     = bx[2];
        d.length = bx[3];   // d_len (nuScenes l = length along heading)
        d.width  = bx[4];   // d_wid
        d.height = bx[5];   // d_hgt
        d.yaw    = bx[6];
        d.vx     = bx[7];
        d.vy     = bx[8];
        d.score  = scores_m_[i];
        d.class_id = pk[2];

        if (d.score >= cfg_.score_threshold)
            out.push_back(d);
    }

    // 7. Circle NMS.
    circle_nms(out, cfg_.nms_radius);
}

}  // namespace edge_fusion
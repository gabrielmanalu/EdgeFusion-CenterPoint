/**
 * voxelizer.hpp — CPU pillar voxelizer for CenterPoint pillar02 / nuScenes.
 *
 * Converts a raw point cloud (x, y, z, intensity[, time]) into the sparse
 * pillar tensor format expected by the pts_voxel_encoder TRT engine.
 *
 * Config defaults match nuScenes CenterPoint pillar02:
 *   pc_range    : [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
 *   voxel_size  : [0.2, 0.2, 8.0]  → grid 512 × 512 × 1
 *   max_voxels  : 30 000
 *   max_pts     : 20
 *
 * Augmented point features written per point (9 total):
 *   [x, y, z, intensity,
 *    Δx (from pillar centre x), Δy (from pillar centre y), Δz (= z),
 *    x (range proxy),           y (range proxy)]
 * Matches mmdet3d PillarFeature(num_input_features=9).
 *
 * Output buffers (caller-allocated):
 *   voxel_features  [max_voxels, max_pts, out_features]   float32
 *   voxel_coords    [max_voxels, 4]  int32  (batch=0, z, y, x)
 *   num_points      [max_voxels]     int32  (valid pts, for masking)
 *   Returns  n_voxels (≤ max_voxels)
 */
#pragma once

#include <cstdint>
#include <unordered_map>

namespace edge_fusion {

struct VoxelizerConfig {
    float pc_range[6]   = {-51.2f, -51.2f, -5.0f, 51.2f, 51.2f, 3.0f};
    float voxel_size[3] = {0.2f, 0.2f, 8.0f};
    int   max_voxels    = 30000;
    int   max_pts       = 32;   // from encoder binding: [-1, 32, 11]
    int   in_features   = 5;    // per point from sensor: x y z intensity time
    int   out_features  = 11;   // after augmentation (see voxelizer.cpp)
};

class Voxelizer {
public:
    explicit Voxelizer(const VoxelizerConfig& cfg = VoxelizerConfig{});

    /**
     * Voxelize. All output buffers must be pre-zeroed by the caller.
     * Returns the number of non-empty voxels placed (≤ cfg.max_voxels).
     */
    int voxelize(const float* points, int n_pts,
                 float* voxel_features, int* voxel_coords,
                 int* num_points) const;

    const VoxelizerConfig& config() const { return cfg_; }
    int grid_x() const { return gx_; }
    int grid_y() const { return gy_; }
    int grid_z() const { return gz_; }

private:
    VoxelizerConfig cfg_;
    int gx_, gy_, gz_;
};

}  // namespace edge_fusion
/**
 * voxelizer.cpp — CPU pillar voxelizer implementation.
 *
 * Uses a flat hash map  (voxel_key → slot index)  to accumulate points
 * into voxels in a single O(N) pass.  The key encodes (iz*gy + iy)*gx + ix.
 */
#include "voxelizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace edge_fusion {

Voxelizer::Voxelizer(const VoxelizerConfig& cfg) : cfg_(cfg) {
    gx_ = static_cast<int>(
        std::round((cfg_.pc_range[3] - cfg_.pc_range[0]) / cfg_.voxel_size[0]));
    gy_ = static_cast<int>(
        std::round((cfg_.pc_range[4] - cfg_.pc_range[1]) / cfg_.voxel_size[1]));
    gz_ = static_cast<int>(
        std::round((cfg_.pc_range[5] - cfg_.pc_range[2]) / cfg_.voxel_size[2]));
}

int Voxelizer::voxelize(const float* points, int n_pts,
                         float* voxel_features, int* voxel_coords,
                         int* num_points) const
{
    const int max_vox  = cfg_.max_voxels;
    const int max_pts  = cfg_.max_pts;
    const int in_feat  = cfg_.in_features;
    const int out_feat = cfg_.out_features;

    // slot_map: voxel key → slot index in [0, max_vox)
    std::unordered_map<int64_t, int> slot_map;
    slot_map.reserve(max_vox * 2);
    int n_voxels = 0;

    for (int i = 0; i < n_pts; ++i) {
        const float* p = points + i * in_feat;
        const float x = p[0], y = p[1], z = p[2];

        // Check range
        if (x < cfg_.pc_range[0] || x >= cfg_.pc_range[3] ||
            y < cfg_.pc_range[1] || y >= cfg_.pc_range[4] ||
            z < cfg_.pc_range[2] || z >= cfg_.pc_range[5])
            continue;

        // Voxel grid index
        int ix = static_cast<int>((x - cfg_.pc_range[0]) / cfg_.voxel_size[0]);
        int iy = static_cast<int>((y - cfg_.pc_range[1]) / cfg_.voxel_size[1]);
        int iz = static_cast<int>((z - cfg_.pc_range[2]) / cfg_.voxel_size[2]);
        ix = std::min(ix, gx_ - 1);
        iy = std::min(iy, gy_ - 1);
        iz = std::min(iz, gz_ - 1);

        int64_t key = (static_cast<int64_t>(iz) * gy_ + iy) * gx_ + ix;

        int slot;
        auto it = slot_map.find(key);
        if (it == slot_map.end()) {
            if (n_voxels >= max_vox)
                continue;   // voxel budget exhausted — skip point
            slot = n_voxels++;
            slot_map[key] = slot;
            // Write coords: (batch=0, z, y, x)
            voxel_coords[slot * 4 + 0] = 0;
            voxel_coords[slot * 4 + 1] = iz;
            voxel_coords[slot * 4 + 2] = iy;
            voxel_coords[slot * 4 + 3] = ix;
        } else {
            slot = it->second;
        }

        int np = num_points[slot];
        if (np >= max_pts)
            continue;   // voxel point budget — skip

        // Write raw features (up to in_feat, zero-pad the rest up to out_feat).
        float* dst = voxel_features + (slot * max_pts + np) * out_feat;
        dst[0] = x;
        dst[1] = y;
        dst[2] = z;
        dst[3] = (in_feat >= 4) ? p[3] : 0.f;   // intensity
        dst[4] = (in_feat >= 5) ? p[4] : 0.f;   // time / sweep offset

        // Augmented features [5..10] filled in second pass.

        ++num_points[slot];
    }

    // Second pass: fill augmented features [5..10].
    //
    // Output feature layout (11 total):
    //   0-4 : x, y, z, intensity, time         (raw from sensor)
    //   5-7 : x-mean_x, y-mean_y, z-mean_z     (cluster-centre offsets)
    //   8-10: x-vcx,    y-vcy,    z-vcz         (voxel-centre offsets)
    //
    // This matches mmdet3d CenterPoint PillarFeature with num_input_features=11.
    if (out_feat >= 11) {
        for (int slot = 0; slot < n_voxels; ++slot) {
            const int np = num_points[slot];
            if (np == 0) continue;

            // Voxel geometric centre (cx, cy from voxel grid, cz = 0 for single Z bin).
            const int ix = voxel_coords[slot * 4 + 3];
            const int iy = voxel_coords[slot * 4 + 2];
            const float vcx = cfg_.pc_range[0] + (ix + 0.5f) * cfg_.voxel_size[0];
            const float vcy = cfg_.pc_range[1] + (iy + 0.5f) * cfg_.voxel_size[1];
            const float vcz = cfg_.pc_range[2] + 0.5f * cfg_.voxel_size[2];

            // Point-cloud mean within this voxel.
            float mx = 0.f, my = 0.f, mz = 0.f;
            for (int k = 0; k < np; ++k) {
                const float* row = voxel_features + (slot * max_pts + k) * out_feat;
                mx += row[0]; my += row[1]; mz += row[2];
            }
            mx /= np; my /= np; mz /= np;

            for (int k = 0; k < np; ++k) {
                float* row = voxel_features + (slot * max_pts + k) * out_feat;
                // Cluster-centre offsets
                row[5] = row[0] - mx;
                row[6] = row[1] - my;
                row[7] = row[2] - mz;
                // Voxel-centre offsets
                row[8]  = row[0] - vcx;
                row[9]  = row[1] - vcy;
                row[10] = row[2] - vcz;
            }
        }
    }

    return n_voxels;
}

}  // namespace edge_fusion
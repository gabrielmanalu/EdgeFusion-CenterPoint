/**
 * int8_calibrator.h
 *
 * Custom TensorRT INT8 calibrator for CenterPoint.
 *
 * Feeds nuScenes LiDAR calibration point clouds through the voxelization
 * pipeline to collect per-layer activation statistics, then writes a
 * calibration cache that accelerates subsequent engine builds.
 *
 * Implements nvinfer1::IInt8EntropyCalibrator2.
 */

#pragma once

// TODO: #include <NvInfer.h>
#include <string>
#include <vector>

namespace edge_fusion {

class CenterPointInt8Calibrator
    /* : public nvinfer1::IInt8EntropyCalibrator2 */
{
public:
    /**
     * @param calib_data_dir  Directory of .bin LiDAR calibration files
     * @param n_samples       Number of calibration samples (recommended: 512)
     * @param cache_file      Path for reading / writing the calibration cache
     */
    CenterPointInt8Calibrator(
        const std::string& calib_data_dir,
        int                n_samples,
        const std::string& cache_file
    );

    ~CenterPointInt8Calibrator();

    // TODO: int    getBatchSize() const noexcept override;
    // TODO: bool   getBatch(void* bindings[], const char* names[],
    //                       int nb_bindings) noexcept override;
    // TODO: const void* readCalibrationCache(size_t& length) noexcept override;
    // TODO: void   writeCalibrationCache(const void* cache,
    //                                    size_t length) noexcept override;

private:
    std::string  calib_data_dir_;
    std::string  cache_file_;
    int          n_samples_;
    int          current_idx_{0};

    // TODO: void*  gpu_input_buffer_{nullptr};
    // TODO: std::vector<char> calibration_cache_;
};

}  // namespace edge_fusion

/**
 * build_engine.cpp
 *
 * Builds TensorRT INT8 / FP16 engines from Autoware-format ONNX models.
 *
 * Produces:
 *   pts_voxel_encoder_centerpoint.engine
 *   pts_backbone_neck_head_centerpoint.engine
 *
 * INT8 calibration is provided by CenterPointInt8Calibrator (int8_calibrator.h).
 *
 * Usage (on Jetson or dev machine with TRT installed):
 *   ./build_engine \
 *       --encoder  pts_voxel_encoder_centerpoint.onnx \
 *       --backbone pts_backbone_neck_head_centerpoint.onnx \
 *       --precision int8 \
 *       --calib-dir /data/nuscenes/sweeps/LIDAR_TOP/ \
 *       --out ./engines/
 */

// TODO: #include <NvInfer.h>
// TODO: #include <NvOnnxParser.h>
// TODO: #include "int8_calibrator.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    // TODO: parse --encoder, --backbone, --precision, --calib-dir, --out

    // TODO: create nvinfer1::IBuilder, INetworkDefinition, IBuilderConfig

    // TODO: parse each ONNX via nvonnxparser::createParser

    // TODO: set precision flag (kINT8 / kFP16 / kFP32)
    //       if kINT8: builder_config->setInt8Calibrator(&calibrator)

    // TODO: buildSerializedNetwork → IHostMemory → write .engine file

    std::cerr << "[build_engine] Not yet implemented.\n";
    return 1;
}

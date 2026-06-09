/**
 * lidar_centerpoint_node.cpp
 *
 * ROS 2 Humble node wrapping the compressed CenterPoint TensorRT engine.
 * Subscribes to PointCloud2 and publishes DetectedObjects.
 *
 * Pipeline:
 *   PointCloud2
 *     → voxelization (pillar scatter CUDA, adapted from EdgeDrive-Perception)
 *     → pts_voxel_encoder TRT engine
 *     → pts_backbone_neck_head TRT engine
 *     → center-head CUDA postprocessing (deployment/plugins/)
 *     → autoware_perception_msgs::msg::DetectedObjects
 *
 * Topic conventions match autoware_lidar_centerpoint:
 *   sub: /sensing/lidar/top/pointcloud_raw_ex
 *   pub: /perception/object_recognition/detection/objects
 */

// TODO: #include <rclcpp/rclcpp.hpp>
// TODO: #include <sensor_msgs/msg/point_cloud2.hpp>
// TODO: #include <autoware_perception_msgs/msg/detected_objects.hpp>
// TODO: #include "centerpoint_trt.h"   // TRT engine wrapper (deployment/inference/)

#include <iostream>

namespace edge_fusion {

class LidarCenterpointNode /* : public rclcpp::Node */ {
public:
    explicit LidarCenterpointNode(/* const rclcpp::NodeOptions& options */)
    /* : rclcpp::Node("lidar_centerpoint", options) */
    {
        // TODO: declare_parameter for engine paths, score_threshold, etc.
        // TODO: load TRT engines via CenterpointTRT wrapper
        // TODO: create_subscription<PointCloud2>(...)
        // TODO: create_publisher<DetectedObjects>(...)
    }

private:
    void pointcloud_callback(
        /* const sensor_msgs::msg::PointCloud2::SharedPtr msg */
    ) {
        // TODO: convert PointCloud2 → float* on GPU
        // TODO: voxelize → encoder engine → backbone engine
        // TODO: center-head CUDA postprocess
        // TODO: map output → DetectedObjects:
        //         shape.type = BOUNDING_BOX
        //         pose.position = decoded (x, y, z)
        //         shape.dimensions = (l, w, h)
        //         existence_probability = score
        //         classification = class_id → autoware label
        // TODO: publish
    }

    // TODO: member vars:
    //   std::unique_ptr<CenterpointTRT> engine_;
    //   rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
    //   rclcpp::Publisher<DetectedObjects>::SharedPtr pub_;
};

}  // namespace edge_fusion


int main(int argc, char** argv) {
    // TODO: rclcpp::init(argc, argv);
    //       auto node = std::make_shared<edge_fusion::LidarCenterpointNode>(...);
    //       rclcpp::spin(node);
    //       rclcpp::shutdown();
    std::cerr << "[lidar_centerpoint_node] Not yet implemented.\n";
    return 1;
}

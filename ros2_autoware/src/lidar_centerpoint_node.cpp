/**
 * lidar_centerpoint_node.cpp — ROS 2 Humble node for EdgeFusion-CenterPoint.
 *
 * Topic interface (mirrors autoware_lidar_centerpoint):
 *   SUB  /sensing/lidar/top/pointcloud_raw_ex   sensor_msgs/PointCloud2
 *   PUB  /perception/object_recognition/detection/objects
 *              autoware_perception_msgs/DetectedObjects
 *
 * nuScenes → Autoware class mapping:
 *   car(0)→CAR  truck(1)→TRUCK  construction(2)→UNKNOWN  bus(3)→BUS
 *   trailer(4)→TRAILER  barrier(5)→UNKNOWN  motorcycle(6)→MOTORCYCLE
 *   bicycle(7)→BICYCLE  pedestrian(8)→PEDESTRIAN  cone(9)→UNKNOWN
 */
#include "centerpoint_trt.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/detected_object.hpp>
#include <autoware_perception_msgs/msg/detected_object_kinematics.hpp>
#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/shape.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace edge_fusion {

namespace aw = autoware_perception_msgs::msg;

// nuScenes class index (0-9) → Autoware ObjectClassification::label
static constexpr uint8_t NS_TO_AW[10] = {
    aw::ObjectClassification::CAR,         // 0  car
    aw::ObjectClassification::TRUCK,       // 1  truck
    aw::ObjectClassification::UNKNOWN,     // 2  construction_vehicle
    aw::ObjectClassification::BUS,         // 3  bus
    aw::ObjectClassification::TRAILER,     // 4  trailer
    aw::ObjectClassification::UNKNOWN,     // 5  barrier
    aw::ObjectClassification::MOTORCYCLE,  // 6  motorcycle
    aw::ObjectClassification::BICYCLE,     // 7  bicycle
    aw::ObjectClassification::PEDESTRIAN,  // 8  pedestrian
    aw::ObjectClassification::UNKNOWN,     // 9  traffic_cone
};

// yaw (rad, ENU) → geometry_msgs quaternion
static geometry_msgs::msg::Quaternion yaw_to_quat(float yaw) {
    geometry_msgs::msg::Quaternion q;
    q.x = 0.0;  q.y = 0.0;
    q.z = std::sin(static_cast<double>(yaw) * 0.5);
    q.w = std::cos(static_cast<double>(yaw) * 0.5);
    return q;
}

class LidarCenterpointNode : public rclcpp::Node {
public:
    explicit LidarCenterpointNode(const rclcpp::NodeOptions& opts)
    : rclcpp::Node("lidar_centerpoint", opts)
    {
        declare_ros_params();
        init_engine();
        create_pub_sub();
        RCLCPP_INFO(get_logger(),
                    "[lidar_centerpoint] Ready. Sub: %s  Pub: %s",
                    sub_topic_.c_str(), pub_topic_.c_str());
    }

private:
    void declare_ros_params() {
        declare_parameter("encoder_engine_path",  std::string(""));
        declare_parameter("backbone_engine_path", std::string(""));
        declare_parameter("postproc_so_path",
            std::string("/workspace/plugins/libcenter_head_postprocess.so"));
        declare_parameter("score_threshold",  0.35);
        declare_parameter("nms_radius",       2.0);
        declare_parameter("max_detections",   500);
        declare_parameter("point_cloud_range",
            std::vector<double>{-51.2, -51.2, -5.0, 51.2, 51.2, 3.0});
        declare_parameter("voxel_size",
            std::vector<double>{0.2, 0.2, 8.0});
        declare_parameter("max_num_voxels",       30000);
        declare_parameter("max_points_per_voxel", 20);
        declare_parameter("sub_topic",
            std::string("/sensing/lidar/top/pointcloud_raw_ex"));
        declare_parameter("pub_topic",
            std::string("/perception/object_recognition/detection/objects"));
    }

    void init_engine() {
        CenterpointConfig cfg;
        cfg.encoder_engine_path  = get_parameter("encoder_engine_path").as_string();
        cfg.backbone_engine_path = get_parameter("backbone_engine_path").as_string();
        cfg.postproc_so_path     = get_parameter("postproc_so_path").as_string();
        cfg.score_threshold = static_cast<float>(
            get_parameter("score_threshold").as_double());
        cfg.nms_radius = static_cast<float>(
            get_parameter("nms_radius").as_double());
        cfg.max_detections = get_parameter("max_detections").as_int();

        auto pcr = get_parameter("point_cloud_range").as_double_array();
        auto vs  = get_parameter("voxel_size").as_double_array();
        for (int i = 0; i < 6; ++i) cfg.vox_cfg.pc_range[i]   = static_cast<float>(pcr[i]);
        for (int i = 0; i < 3; ++i) cfg.vox_cfg.voxel_size[i] = static_cast<float>(vs[i]);
        cfg.vox_cfg.max_voxels = get_parameter("max_num_voxels").as_int();
        cfg.vox_cfg.max_pts    = get_parameter("max_points_per_voxel").as_int();

        sub_topic_ = get_parameter("sub_topic").as_string();
        pub_topic_ = get_parameter("pub_topic").as_string();

        if (cfg.encoder_engine_path.empty() || cfg.backbone_engine_path.empty())
            throw std::runtime_error(
                "Engine paths not set — check centerpoint.param.yaml");

        engine_ = std::make_unique<CenterpointTRT>(cfg);
    }

    void create_pub_sub() {
        sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            sub_topic_, 10,
            std::bind(&LidarCenterpointNode::on_pointcloud, this,
                      std::placeholders::_1));
        pub_ = create_publisher<aw::DetectedObjects>(pub_topic_, 10);
    }

    // Extract (x, y, z, intensity, time) from any PointCloud2 layout.
    // Always writes 5 floats per point — matches VoxelizerConfig::in_features=5.
    // 'time' is read from field named "time" or "t" if present, else zero-padded.
    int extract_points(const sensor_msgs::msg::PointCloud2& msg,
                       std::vector<float>& buf) {
        const int n = msg.width * msg.height;
        buf.resize(n * 5);

        sensor_msgs::PointCloud2ConstIterator<float> ix(msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iy(msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iz(msg, "z");

        bool has_i = false, has_t = false;
        for (const auto& f : msg.fields) {
            if (f.name == "intensity" || f.name == "i") has_i = true;
            if (f.name == "time"      || f.name == "t") has_t = true;
        }

        int k = 0;
        if (has_i && has_t) {
            sensor_msgs::PointCloud2ConstIterator<float> ii(msg, "intensity");
            sensor_msgs::PointCloud2ConstIterator<float> it(msg, "time");
            for (; ix != ix.end(); ++ix, ++iy, ++iz, ++ii, ++it) {
                if (!std::isfinite(*ix) || !std::isfinite(*iy) || !std::isfinite(*iz))
                    continue;
                buf[k*5+0]=*ix; buf[k*5+1]=*iy; buf[k*5+2]=*iz;
                buf[k*5+3]=*ii; buf[k*5+4]=*it;
                ++k;
            }
        } else if (has_i) {
            sensor_msgs::PointCloud2ConstIterator<float> ii(msg, "intensity");
            for (; ix != ix.end(); ++ix, ++iy, ++iz, ++ii) {
                if (!std::isfinite(*ix) || !std::isfinite(*iy) || !std::isfinite(*iz))
                    continue;
                buf[k*5+0]=*ix; buf[k*5+1]=*iy; buf[k*5+2]=*iz;
                buf[k*5+3]=*ii; buf[k*5+4]=0.f;
                ++k;
            }
        } else {
            for (; ix != ix.end(); ++ix, ++iy, ++iz) {
                if (!std::isfinite(*ix) || !std::isfinite(*iy) || !std::isfinite(*iz))
                    continue;
                buf[k*5+0]=*ix; buf[k*5+1]=*iy; buf[k*5+2]=*iz;
                buf[k*5+3]=0.f; buf[k*5+4]=0.f;
                ++k;
            }
        }
        return k;
    }

    // Detection3D → autoware_perception_msgs DetectedObject.
    // Shape dimensions: Autoware x=width, y=length, z=height.
    // CUDA postproc outputs [d_len, d_wid, d_hgt] → swap on assignment.
    static aw::DetectedObject to_detected_object(const Detection3D& d) {
        aw::DetectedObject obj;
        obj.existence_probability = d.score;

        aw::ObjectClassification cls;
        cls.label = (d.class_id >= 0 && d.class_id < 10)
                    ? NS_TO_AW[d.class_id]
                    : static_cast<uint8_t>(aw::ObjectClassification::UNKNOWN);
        cls.probability = d.score;
        obj.classification.push_back(cls);

        // Pose
        auto& pose = obj.kinematics.pose_with_covariance.pose;
        pose.position.x = d.cx;
        pose.position.y = d.cy;
        pose.position.z = d.cz;
        pose.orientation = yaw_to_quat(d.yaw);

        obj.kinematics.has_position_covariance = false;
        obj.kinematics.orientation_availability =
            aw::DetectedObjectKinematics::AVAILABLE;

        // Velocity
        if (d.vx != 0.f || d.vy != 0.f) {
            obj.kinematics.twist_with_covariance.twist.linear.x = d.vx;
            obj.kinematics.twist_with_covariance.twist.linear.y = d.vy;
            obj.kinematics.has_twist = true;
        }

        // Shape
        obj.shape.type = aw::Shape::BOUNDING_BOX;
        obj.shape.dimensions.x = d.width;   // Autoware x = width
        obj.shape.dimensions.y = d.length;  // Autoware y = length
        obj.shape.dimensions.z = d.height;

        return obj;
    }

    void on_pointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        auto t0 = std::chrono::steady_clock::now();

        std::vector<float> pts;
        const int n_pts = extract_points(*msg, pts);
        if (n_pts == 0) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                 "Empty point cloud.");
            return;
        }

        std::vector<Detection3D> dets;
        try {
            engine_->detect(pts.data(), n_pts, dets);
        } catch (const std::exception& ex) {
            RCLCPP_ERROR(get_logger(), "detect() failed: %s", ex.what());
            return;
        }

        aw::DetectedObjects out;
        out.header = msg->header;
        out.objects.reserve(dets.size());
        for (const auto& d : dets) out.objects.push_back(to_detected_object(d));
        pub_->publish(out);

        auto ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0).count();
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                             "OK: %d pts → %zu dets  %.1f ms/frame",
                             n_pts, dets.size(), ms);
    }

    std::unique_ptr<CenterpointTRT> engine_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<aw::DetectedObjects>::SharedPtr pub_;
    std::string sub_topic_, pub_topic_;
};

}  // namespace edge_fusion

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(
        std::make_shared<edge_fusion::LidarCenterpointNode>(
            rclcpp::NodeOptions{}));
    rclcpp::shutdown();
    return 0;
}
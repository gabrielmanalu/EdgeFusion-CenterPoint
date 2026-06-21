# EdgeFusion-CenterPoint — ROS 2 / Autoware Node

Wraps the QAT INT8 TensorRT engine as a ROS 2 Humble node with the same topic
interface as Autoware's `autoware_lidar_centerpoint` — a drop-in replacement
requiring no changes to downstream tracking, prediction, or planning nodes.

---

## Pipeline

```
PointCloud2  →  CPU Voxelizer (30k×32×11)
                      ↓
         pts_voxel_encoder.engine  (TRT INT8)
                      ↓
          CPU pillar scatter  →  BEV [1, 64, 512, 512]
                      ↓
    pts_backbone_neck_head.engine  (TRT INT8)
                      ↓
       CUDA center-head postproc  (libcenter_head_postprocess.so)
                      ↓
         CPU circle NMS
                      ↓
      DetectedObjects  →  publish
```

**Measured end-to-end** (Jetson Orin Nano 8GB, 25W):
`p50 = 34.12 ms · p99 = 43.93 ms · ~12 Hz on nuScenes sweeps · 16.29W VDD_IN`

---

## Topic interface

| | Topic | Type |
|---|---|---|
| Subscribe | `/sensing/lidar/top/pointcloud_raw_ex` | `sensor_msgs/PointCloud2` |
| Publish | `/perception/object_recognition/detection/objects` | `autoware_perception_msgs/DetectedObjects` |

Override via `sub_topic` / `pub_topic` parameters.

---

## nuScenes → Autoware class mapping

| nuScenes | Autoware |
|---|---|
| car | CAR |
| truck | TRUCK |
| bus | BUS |
| trailer | TRAILER |
| motorcycle | MOTORCYCLE |
| bicycle | BICYCLE |
| pedestrian | PEDESTRIAN |
| construction\_vehicle · barrier · traffic\_cone | UNKNOWN |

---

## Build

### Prerequisites

| | Version |
|---|---|
| Base Docker image | `edgedrive-ros2:latest` (on Jetson) |
| ROS 2 | Humble |
| JetPack | R36.4.0 |
| TensorRT | 10.3.0 |

See [EdgeDrive-Perception](https://github.com/gabrielmanalu/EdgeDrive-Perception) for base docker image information.

Engines must be built first (see `deployment/README.md`):
```
deployment/output/engines/qat_best/
    pts_voxel_encoder.engine
    pts_backbone_neck_head.engine
```

Postprocessor `.so` must be compiled (see `deployment/plugins/`):
```bash
cd deployment/plugins && nvcc -O3 --use_fast_math -arch=sm_87 \
    -Xcompiler -fPIC -shared \
    center_head_postprocess.cu -o libcenter_head_postprocess.so
```

Clone `autoware_msgs` for the message package (needs internet, one-time):
```bash
git clone --depth 1 \
    https://github.com/autowarefoundation/autoware_msgs.git \
    ros2_autoware/autoware_msgs_src
echo "ros2_autoware/autoware_msgs_src/" >> .gitignore
```

### Docker build

```bash
docker build --network=host -t edge_fusion_ros2 \
    -f ros2_autoware/docker/Dockerfile .
```

### Incremental rebuild (after C++ source edits)

```bash
# Copy changed file into the running container
docker cp ros2_autoware/src/lidar_centerpoint_node.cpp \
    efc:/ros2_ws/src/edge_fusion_centerpoint/src/

# Recompile — ~30 seconds
docker exec -it efc bash -c "
  source /opt/ros/humble/setup.bash &&
  source /opt/aw_msgs_ws/install/setup.bash &&
  cd /ros2_ws &&
  colcon build --packages-select edge_fusion_centerpoint \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
                 -DPOSTPROC_HEADER_DIR=/workspace/plugins"
```

---

## Run

### Persistent container (required for DDS discovery)

All processes must share one container. DDS multicast between separate
`docker run` instances on the same host is unreliable.

```bash
# Start persistent container
docker run -d --name efc \
  --network host --ipc host --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/deployment/output/engines:/workspace/engines:ro \
  -v $(pwd)/deployment/plugins:/workspace/plugins:ro \
  -v $(pwd)/bags:/bags:ro \
  -e ROS_DOMAIN_ID=42 \
  edge_fusion_ros2 sleep infinity
```

### Terminal 1 — detection node

```bash
docker exec -it efc bash -c "
  source /opt/ros/humble/setup.bash &&
  source /opt/aw_msgs_ws/install/setup.bash &&
  source /ros2_ws/install/setup.bash &&
  ros2 launch edge_fusion_centerpoint centerpoint.launch.xml \
    encoder_engine:=/workspace/engines/qat_best/pts_voxel_encoder.engine \
    backbone_engine:=/workspace/engines/qat_best/pts_backbone_neck_head.engine \
    postproc_so:=/workspace/plugins/libcenter_head_postprocess.so"
```

### Terminal 2 — rosbag replay

```bash
docker exec -it efc bash -c "
  source /opt/ros/humble/setup.bash &&
  ros2 bag play /bags/nuscenes_scene0.db3 --rate 1.0 --loop"
```

Convert nuScenes mini to a bag first (one-time, needs `pip3 install rosbags nuscenes-devkit`):
```bash
python3 ros2_autoware/scripts/nuscenes_to_ros2bag.py \
    --dataroot /data/sets/nuscenes \
    --version v1.0-mini \
    --output bags/nuscenes_scene0.db3 \
    --scene-idx 0
```

### Terminal 3 — RViz2 visualisation

```bash
xhost +local:docker   # allow display access (once per session)

docker exec -it efc bash -c "
  source /opt/ros/humble/setup.bash &&
  source /opt/aw_msgs_ws/install/setup.bash &&
  source /ros2_ws/install/setup.bash &&
  rviz2"
```

In RViz2:
- Fixed Frame → `lidar_top`
- Add **PointCloud2** → `/sensing/lidar/top/pointcloud_raw_ex` · Color: AxisColor Z · range −2/3m
- Add **`autoware_perception_rviz_plugin/DetectedObjects`** → `/perception/object_recognition/detection/objects`
- Add **Image** → `/sensing/camera/front/image_raw`
- File → Save Config As → `ros2_autoware/config/centerpoint.rviz`

### Terminal 4 — static TF

```bash
docker exec -it efc bash -c "
  source /opt/ros/humble/setup.bash &&
  ros2 run tf2_ros static_transform_publisher \
    0 0 0 0 0 0 base_link lidar_top"
```

---

## Alternative visualisation — Foxglove Studio

Install bridge inside the container:
```bash
docker exec -it efc bash -c "apt-get install -y ros-humble-foxglove-bridge"
```

Run bridge:
```bash
docker exec -it efc bash -c "
  source /opt/ros/humble/setup.bash &&
  ros2 run foxglove_bridge foxglove_bridge"
```

Open [studio.foxglove.dev](https://studio.foxglove.dev) → Open connection →
`ws://<jetson-ip>:8765` → import `ros2_autoware/config/foxglove_layout.json`.

For generic visualisers (no Autoware plugin required), run the MarkerArray converter:
```bash
docker exec -it efc bash -c "
  source /opt/ros/humble/setup.bash &&
  source /opt/aw_msgs_ws/install/setup.bash &&
  source /ros2_ws/install/setup.bash &&
  python3 /ros2_ws/src/edge_fusion_centerpoint/scripts/detections_to_markers.py"
```

Subscribes to `DetectedObjects`, publishes `/perception/markers` (MarkerArray)
with class-coloured wireframe boxes, text labels, and velocity arrows.
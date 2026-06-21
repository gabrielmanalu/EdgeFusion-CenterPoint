# Extends the EdgeDrive ROS 2 image already on this Jetson.
# All CUDA / TRT / ROS 2 Humble layers are inherited — no re-download.
# No network access needed during the build: all sources are COPY'd from host.
FROM edgedrive-ros2:latest

# ── Visualisation tools ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-rviz2 \
        ros-humble-autoware-perception-rviz-plugin \
    && rm -rf /var/lib/apt/lists/*

# ── autoware_perception_msgs ───────────────────────────────────────────────────
# Pre-cloned on the host into ros2_autoware/autoware_msgs_src/ before building:
#   git clone --depth 1 --branch humble \
#       https://github.com/autowarefoundation/autoware_msgs.git \
#       ros2_autoware/autoware_msgs_src
COPY ros2_autoware/autoware_msgs_src /opt/aw_msgs_ws/src/autoware_msgs

RUN bash -c " \
    source /opt/ros/humble/setup.bash && \
    cd /opt/aw_msgs_ws && \
    colcon build \
        --packages-select autoware_perception_msgs \
        --cmake-args -DCMAKE_BUILD_TYPE=Release" && \
    echo "source /opt/aw_msgs_ws/install/setup.bash" >> /root/.bashrc

# ── EdgeFusion-CenterPoint ROS 2 package ──────────────────────────────────────
ARG WS=/ros2_ws
WORKDIR ${WS}/src

COPY ros2_autoware/                     ./edge_fusion_centerpoint/
COPY deployment/plugins/center_head_postprocess.h \
     /workspace/plugins/center_head_postprocess.h

WORKDIR ${WS}

RUN bash -c " \
    source /opt/ros/humble/setup.bash && \
    source /opt/aw_msgs_ws/install/setup.bash && \
    colcon build \
        --packages-select edge_fusion_centerpoint \
        --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPOSTPROC_HEADER_DIR=/workspace/plugins && \
    echo '[docker] Build OK'"

ENV WS=${WS}
RUN echo "source /opt/ros/humble/setup.bash"         >> /root/.bashrc && \
    echo "source /opt/aw_msgs_ws/install/setup.bash"  >> /root/.bashrc && \
    echo "source ${WS}/install/setup.bash"            >> /root/.bashrc

CMD ["bash", "-c", \
     "source /opt/ros/humble/setup.bash && \
      source /opt/aw_msgs_ws/install/setup.bash && \
      source ${WS}/install/setup.bash && \
      ros2 launch edge_fusion_centerpoint centerpoint.launch.xml"]
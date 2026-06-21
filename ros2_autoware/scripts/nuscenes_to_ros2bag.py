#!/usr/bin/env python3
"""
nuscenes_to_ros2bag.py — Convert nuScenes scenes to a ROS 2 bag.

LiDAR is written at full sweep frequency (~20 Hz) by following the
sample_data chain, not just keyframe samples.
Camera (CAM_FRONT) is written at keyframe rate (~2 Hz).

Topics:
  /sensing/lidar/top/pointcloud_raw_ex   sensor_msgs/PointCloud2
  /sensing/camera/front/image_raw        sensor_msgs/Image

Usage:
  python3 ros2_autoware/scripts/nuscenes_to_ros2bag.py \
      --dataroot /data/sets/nuscenes \
      --version  v1.0-mini \
      --output   bags/nuscenes_scene0.db3 \
      --scene-idx 0

  # All scenes:
  python3 ros2_autoware/scripts/nuscenes_to_ros2bag.py --all-scenes \
      --output bags/nuscenes_all.db3

Requirements:
  pip3 install rosbags nuscenes-devkit
"""

import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path


def check_dependencies():
    missing = []
    try:
        from nuscenes.nuscenes import NuScenes   # noqa: F401
    except ImportError:
        missing.append("nuscenes-devkit")
    try:
        from rosbags.rosbag2 import Writer        # noqa: F401
    except ImportError:
        missing.append("rosbags")
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print(f"Install: pip3 install {' '.join(missing)}")
        sys.exit(1)


def make_header(typestore, sec, nsec, frame_id):
    Header = typestore.types['std_msgs/msg/Header']
    Time   = typestore.types['builtin_interfaces/msg/Time']
    return Header(stamp=Time(sec=sec, nanosec=nsec), frame_id=frame_id)


def write_lidar(writer, conn, typestore, pcd_path, ts_ns, frame_id):
    """Write one .bin sweep as PointCloud2 (x y z intensity ring)."""
    PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
    PointField  = typestore.types['sensor_msgs/msg/PointField']

    points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)

    sec  = int(ts_ns // 1_000_000_000)
    nsec = int(ts_ns %  1_000_000_000)
    header = make_header(typestore, sec, nsec, frame_id)

    fields = [
        PointField(name="x",         offset=0,  datatype=7, count=1),
        PointField(name="y",         offset=4,  datatype=7, count=1),
        PointField(name="z",         offset=8,  datatype=7, count=1),
        PointField(name="intensity", offset=12, datatype=7, count=1),
        PointField(name="ring",      offset=16, datatype=7, count=1),
    ]

    msg = PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        fields=fields,
        is_bigendian=False,
        point_step=20,
        row_step=points.shape[0] * 20,
        data=points.flatten().view(np.uint8),
        is_dense=True)

    writer.write(conn, ts_ns,
                 typestore.serialize_cdr(msg, PointCloud2.__msgtype__))


def write_camera(writer, conn, typestore, img_path, ts_ns, frame_id):
    """Write one JPEG frame as sensor_msgs/Image."""
    Image = typestore.types['sensor_msgs/msg/Image']

    img = cv2.imread(img_path)
    if img is None:
        return

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sec  = int(ts_ns // 1_000_000_000)
    nsec = int(ts_ns %  1_000_000_000)
    header = make_header(typestore, sec, nsec, frame_id)

    msg = Image(
        header=header,
        height=h, width=w,
        encoding="rgb8",
        is_bigendian=False,
        step=w * 3,
        data=img_rgb.flatten().astype(np.uint8))

    writer.write(conn, ts_ns,
                 typestore.serialize_cdr(msg, Image.__msgtype__))


def convert_scene(nusc, scene, writer, lidar_conn, cam_conn, typestore,
                  dataroot, camera):
    """Write all sweeps (~20 Hz LiDAR) + keyframe camera for one scene."""
    lidar_written = 0
    cam_written   = 0

    # ── LiDAR: full sweep chain (~20 Hz) ─────────────────────────────────────
    first_sample    = nusc.get('sample', scene['first_sample_token'])
    lidar_sd_token  = first_sample['data']['LIDAR_TOP']

    while lidar_sd_token:
        sd      = nusc.get('sample_data', lidar_sd_token)
        ts_ns   = sd['timestamp'] * 1000          # μs → ns
        pcd_path = os.path.join(dataroot, sd['filename'])

        if os.path.exists(pcd_path):
            write_lidar(writer, lidar_conn, typestore,
                        pcd_path, ts_ns, "lidar_top")
            lidar_written += 1

        lidar_sd_token = sd['next'] if sd['next'] else None

    # ── Camera: keyframes only (~2 Hz) ────────────────────────────────────────
    sample_token = scene['first_sample_token']
    while sample_token:
        sample = nusc.get('sample', sample_token)
        ts_ns  = sample['timestamp'] * 1000

        if camera in sample['data']:
            cam_sd   = nusc.get('sample_data', sample['data'][camera])
            img_path = os.path.join(dataroot, cam_sd['filename'])
            write_camera(writer, cam_conn, typestore,
                         img_path, ts_ns, "cam_front")
            cam_written += 1

        sample_token = sample['next'] if sample['next'] else None

    return lidar_written, cam_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",   default="/data/sets/nuscenes")
    parser.add_argument("--version",    default="v1.0-mini")
    parser.add_argument("--output",     default="bags/nuscenes_scene0.db3")
    parser.add_argument("--scene-idx",  type=int, default=0)
    parser.add_argument("--camera",     default="CAM_FRONT")
    parser.add_argument("--all-scenes", action="store_true")
    args = parser.parse_args()

    check_dependencies()

    from nuscenes.nuscenes import NuScenes
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    print(f"Loading nuScenes {args.version} from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    scenes = nusc.scene if args.all_scenes else [nusc.scene[args.scene_idx]]
    print(f"Converting {len(scenes)} scene(s) "
          f"(LiDAR at full sweep rate ~20 Hz, camera at keyframe rate ~2 Hz)...")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Writer(str(out_path), version=8) as writer:
        lidar_conn = writer.add_connection(
            "/sensing/lidar/top/pointcloud_raw_ex",
            typestore.types['sensor_msgs/msg/PointCloud2'].__msgtype__,
            typestore=typestore)
        cam_conn = writer.add_connection(
            "/sensing/camera/front/image_raw",
            typestore.types['sensor_msgs/msg/Image'].__msgtype__,
            typestore=typestore)

        total_lidar = total_cam = 0
        for scene in scenes:
            print(f"  {scene['name']} ...", end=' ', flush=True)
            l, c = convert_scene(nusc, scene, writer, lidar_conn, cam_conn,
                                  typestore, args.dataroot, args.camera)
            total_lidar += l
            total_cam   += c
            print(f"{l} lidar sweeps, {c} camera frames")

    print(f"\nDone: {total_lidar} LiDAR sweeps + {total_cam} camera frames")
    print(f"  → {args.output}")
    print(f"\nReplay (from repo root):")
    print(f"  ros2 bag play {args.output} --rate 1.0")


if __name__ == "__main__":
    main()

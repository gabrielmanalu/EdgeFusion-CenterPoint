#!/usr/bin/env python3
"""
detections_to_markers.py

Converts autoware_perception_msgs/DetectedObjects to
visualization_msgs/MarkerArray for display in RViz or Foxglove Studio.

Publishes:
  /perception/markers          MarkerArray  — wireframe boxes + labels + velocity arrows
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from autoware_perception_msgs.msg import DetectedObjects, ObjectClassification
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# ── Per-class colour palette (RGBA, 0-1) ─────────────────────────────────────
# Matches the colour scheme used in Autoware / nuScenes visualisations.
CLASS_RGBA = {
    ObjectClassification.CAR:         (0.15, 0.60, 1.00, 0.85),   # blue
    ObjectClassification.TRUCK:       (1.00, 0.55, 0.10, 0.85),   # orange
    ObjectClassification.BUS:         (1.00, 0.90, 0.10, 0.85),   # yellow
    ObjectClassification.TRAILER:     (0.70, 0.20, 0.90, 0.85),   # purple
    ObjectClassification.MOTORCYCLE:  (0.10, 0.90, 0.40, 0.85),   # green
    ObjectClassification.BICYCLE:     (0.50, 1.00, 0.10, 0.85),   # lime
    ObjectClassification.PEDESTRIAN:  (1.00, 0.20, 0.30, 0.85),   # red
    ObjectClassification.UNKNOWN:     (0.70, 0.70, 0.70, 0.60),   # grey
}

CLASS_NAME = {
    ObjectClassification.CAR:         'Car',
    ObjectClassification.TRUCK:       'Truck',
    ObjectClassification.BUS:         'Bus',
    ObjectClassification.TRAILER:     'Trailer',
    ObjectClassification.MOTORCYCLE:  'Moto',
    ObjectClassification.BICYCLE:     'Bike',
    ObjectClassification.PEDESTRIAN:  'Ped',
    ObjectClassification.UNKNOWN:     '?',
}

# Pedestrians shown as cylinders; everything else as wireframe boxes.
CYLINDER_CLASSES = {ObjectClassification.PEDESTRIAN}


def _rgba(label):
    return CLASS_RGBA.get(label, CLASS_RGBA[ObjectClassification.UNKNOWN])


def _name(label):
    return CLASS_NAME.get(label, '?')


def _yaw_to_quat(yaw):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def _box_wireframe_points(cx, cy, cz, length, width, height, yaw):
    """Return 24 Point pairs (LINE_LIST) for a wireframe oriented box."""
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    # 4 corners in local XY plane (l = along heading, w = lateral)
    local = [
        ( length / 2,  width / 2),
        ( length / 2, -width / 2),
        (-length / 2, -width / 2),
        (-length / 2,  width / 2),
    ]

    def rotate(lx, ly):
        return (cos_y * lx - sin_y * ly + cx,
                sin_y * lx + cos_y * ly + cy)

    bot = [(*rotate(lx, ly), cz - height / 2) for lx, ly in local]
    top = [(*rotate(lx, ly), cz + height / 2) for lx, ly in local]

    def pt(xyz):
        p = Point()
        p.x, p.y, p.z = xyz
        return p

    pairs = []
    for i in range(4):
        j = (i + 1) % 4
        # bottom edge
        pairs += [pt(bot[i]), pt(bot[j])]
        # top edge
        pairs += [pt(top[i]), pt(top[j])]
        # vertical
        pairs += [pt(bot[i]), pt(top[i])]
    return pairs          # 12 edges × 2 = 24 points


class DetectionsToMarkers(Node):
    def __init__(self):
        super().__init__('detections_to_markers')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5)

        self.sub = self.create_subscription(
            DetectedObjects,
            '/perception/object_recognition/detection/objects',
            self._callback, qos)

        self.pub = self.create_publisher(MarkerArray, '/perception/markers', 10)
        self._prev_count = 0
        self.get_logger().info('detections_to_markers ready.')

    def _callback(self, msg):
        markers = []
        mid = 0

        for obj in msg.objects:
            label = (obj.classification[0].label
                     if obj.classification
                     else ObjectClassification.UNKNOWN)
            score = (obj.classification[0].probability
                     if obj.classification else 0.0)

            r, g, b, a = _rgba(label)
            pose   = obj.kinematics.pose_with_covariance.pose
            dims   = obj.shape.dimensions
            cx, cy, cz = pose.position.x, pose.position.y, pose.position.z
            # Autoware dims: x=width, y=length, z=height
            width, length, height = dims.x, dims.y, dims.z

            # Extract yaw from quaternion
            qz, qw = pose.orientation.z, pose.orientation.w
            yaw = 2.0 * math.atan2(qz, qw)

            if label in CYLINDER_CLASSES:
                # ── Cylinder for pedestrians ──────────────────────────────
                m = Marker()
                m.header      = msg.header
                m.ns          = 'detections'
                m.id          = mid; mid += 1
                m.type        = Marker.CYLINDER
                m.action      = Marker.ADD
                m.pose        = pose
                m.scale.x     = max(width,  0.3)
                m.scale.y     = max(length, 0.3)
                m.scale.z     = max(height, 0.1)
                m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
                m.lifetime.sec = 0
                m.lifetime.nanosec = 300_000_000
                markers.append(m)
            else:
                # ── Wireframe box ─────────────────────────────────────────
                m = Marker()
                m.header      = msg.header
                m.ns          = 'detections'
                m.id          = mid; mid += 1
                m.type        = Marker.LINE_LIST
                m.action      = Marker.ADD
                m.pose.orientation.w = 1.0
                m.scale.x     = 0.06        # line width (metres)
                m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
                m.points      = _box_wireframe_points(
                    cx, cy, cz, length, width, height, yaw)
                m.lifetime.sec = 0
                m.lifetime.nanosec = 300_000_000
                markers.append(m)

            # ── Label ─────────────────────────────────────────────────────
            t = Marker()
            t.header    = msg.header
            t.ns        = 'labels'
            t.id        = mid; mid += 1
            t.type      = Marker.TEXT_VIEW_FACING
            t.action    = Marker.ADD
            t.pose.position.x = cx
            t.pose.position.y = cy
            t.pose.position.z = cz + height / 2 + 0.3
            t.pose.orientation.w = 1.0
            t.scale.z   = 0.5
            t.color.r = t.color.g = t.color.b = 1.0
            t.color.a   = 0.9
            t.text      = f"{_name(label)} {score:.2f}"
            t.lifetime.sec = 0
            t.lifetime.nanosec = 300_000_000
            markers.append(t)

            # ── Velocity arrow ────────────────────────────────────────────
            if obj.kinematics.has_twist:
                tw = obj.kinematics.twist_with_covariance.twist
                spd = math.hypot(tw.linear.x, tw.linear.y)
                if spd > 0.5:
                    arr = Marker()
                    arr.header  = msg.header
                    arr.ns      = 'velocity'
                    arr.id      = mid; mid += 1
                    arr.type    = Marker.ARROW
                    arr.action  = Marker.ADD
                    arr.scale.x = 0.15   # shaft diameter
                    arr.scale.y = 0.25   # head diameter
                    arr.scale.z = 0.30   # head length
                    arr.color.r, arr.color.g, arr.color.b = 1.0, 1.0, 0.0
                    arr.color.a = 0.85
                    start = Point(x=cx, y=cy, z=cz)
                    end   = Point(x=cx + tw.linear.x,
                                  y=cy + tw.linear.y,
                                  z=cz)
                    arr.points  = [start, end]
                    arr.lifetime.sec = 0
                    arr.lifetime.nanosec = 300_000_000
                    markers.append(arr)

        # Delete stale markers from previous frame
        for stale_id in range(mid, self._prev_count):
            d = Marker()
            d.ns = 'detections'; d.id = stale_id
            d.action = Marker.DELETE
            markers.append(d)
            d2 = Marker()
            d2.ns = 'labels'; d2.id = stale_id
            d2.action = Marker.DELETE
            markers.append(d2)

        self._prev_count = mid

        out = MarkerArray()
        out.markers = markers
        self.pub.publish(out)


def main():
    rclpy.init()
    node = DetectionsToMarkers()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

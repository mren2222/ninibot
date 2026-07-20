# flake8: noqa
import time

import numpy as np
from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_geometry_msgs  # noqa: F401 - registers geometry message transforms.


def quaternion_to_rotation_matrix(x, y, z, w):
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        return np.eye(3, dtype=float)

    scale = 2.0 / norm
    xx = x * x * scale
    yy = y * y * scale
    zz = z * z * scale
    xy = x * y * scale
    xz = x * z * scale
    yz = y * z * scale
    wx = w * x * scale
    wy = w * y * scale
    wz = w * z * scale

    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=float,
    )


def transform_msg_to_matrix(transform_msg):
    translation = transform_msg.translation
    rotation = transform_msg.rotation

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = quaternion_to_rotation_matrix(
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    )
    transform[:3, 3] = [translation.x, translation.y, translation.z]
    return transform


def lookup_transform_matrix(
    node,
    tf_buffer,
    target_frame,
    source_frame,
    timeout_sec,
    spin=True,
):
    if not spin:
        stamped = tf_buffer.lookup_transform(target_frame, source_frame, Time())
        return transform_msg_to_matrix(stamped.transform)

    deadline = time.monotonic() + max(0.0, timeout_sec)
    last_error = None
    first_try = True

    while first_try or time.monotonic() < deadline:
        first_try = False
        if spin:
            rclpy.spin_once(node, timeout_sec=0.02)
        try:
            stamped = tf_buffer.lookup_transform(target_frame, source_frame, Time())
            return transform_msg_to_matrix(stamped.transform)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Waiting for TF {target_frame} <- {source_frame}: {last_error}")


def transform_camera_point_to_base_xyz(
    node,
    tf_buffer,
    marker_camera_xyz,
    base_frame,
    camera_frame,
    timeout_sec,
    base_xyz_offset=(0.0, 0.0, 0.0),
    spin=False,
):
    marker_camera_xyz = np.asarray(marker_camera_xyz, dtype=float)

    marker_camera_point = PointStamped()
    marker_camera_point.header.frame_id = camera_frame
    marker_camera_point.header.stamp = Time().to_msg()
    marker_camera_point.point.x = float(marker_camera_xyz[0])
    marker_camera_point.point.y = float(marker_camera_xyz[1])
    marker_camera_point.point.z = float(marker_camera_xyz[2])

    marker_base_point = tf_buffer.transform(
        marker_camera_point,
        base_frame,
        timeout=Duration(seconds=timeout_sec),
    )
    if spin:
        rclpy.spin_once(node, timeout_sec=0.0)

    return np.array(
        [
            marker_base_point.point.x,
            marker_base_point.point.y,
            marker_base_point.point.z,
        ],
        dtype=float,
    ) + np.asarray(base_xyz_offset, dtype=float)

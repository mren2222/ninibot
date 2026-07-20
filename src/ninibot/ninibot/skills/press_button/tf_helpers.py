import numpy as np
from rclpy.time import Time


def lookup_transform_xyz(tf_buffer, target_frame, source_frame):
    try:
        stamped = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
        )
    except Exception:
        return None
    t = stamped.transform.translation
    return np.array([t.x, t.y, t.z], dtype=float)


def lookup_gripper_base_xyz(tf_buffer, base_frame, gripper_frame):
    return lookup_transform_xyz(tf_buffer, base_frame, gripper_frame)


def lookup_gripper_camera_xyz(tf_buffer, camera_frame, gripper_frame):
    return lookup_transform_xyz(tf_buffer, camera_frame, gripper_frame)

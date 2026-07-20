# flake8: noqa
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import tf2_ros


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


def rotation_matrix_to_rpy(rotation):
    pitch = math.atan2(
        -rotation[2, 0],
        math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0]),
    )
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return roll, pitch, yaw


def rotation_matrix_to_quaternion(rotation):
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def make_point(vector):
    point = Point()
    point.x = float(vector[0])
    point.y = float(vector[1])
    point.z = float(vector[2])
    return point


class GripperPoseDebugNode(Node):
    def __init__(self):
        super().__init__("gripper_pose_debug_node")

        self.declare_parameter("base_frame", "base")
        self.declare_parameter("gripper_frame", "grippertip")
        self.declare_parameter("camera_frame", "camera")
        self.declare_parameter("optical_frame", "camera_optical_frame")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("print_period", 0.5)
        self.declare_parameter("axis_length", 0.08)
        self.declare_parameter("camera_axis_length", 0.06)
        self.declare_parameter("optical_axis_length", 0.05)
        self.declare_parameter("axis_radius", 0.006)
        self.declare_parameter("text_scale", 0.035)
        self.declare_parameter("marker_topic", "/gripper_pose_debug/markers")
        self.declare_parameter("pose_topic", "/gripper_pose_debug/pose")
        self.declare_parameter("camera_pose_topic", "/gripper_pose_debug/camera_pose")
        self.declare_parameter("optical_pose_topic", "/gripper_pose_debug/optical_pose")

        self.base_frame = self.get_parameter("base_frame").value
        self.gripper_frame = self.get_parameter("gripper_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.optical_frame = self.get_parameter("optical_frame").value or None
        publish_rate = float(self.get_parameter("publish_rate").value)
        self.print_period = float(self.get_parameter("print_period").value)
        self.axis_length = float(self.get_parameter("axis_length").value)
        self.camera_axis_length = float(self.get_parameter("camera_axis_length").value)
        self.optical_axis_length = float(self.get_parameter("optical_axis_length").value)
        self.axis_radius = float(self.get_parameter("axis_radius").value)
        self.text_scale = float(self.get_parameter("text_scale").value)
        marker_topic = self.get_parameter("marker_topic").value
        pose_topic = self.get_parameter("pose_topic").value
        camera_pose_topic = self.get_parameter("camera_pose_topic").value
        optical_pose_topic = self.get_parameter("optical_pose_topic").value
        joint_states_topic = self.get_parameter("joint_states_topic").value
        self.joint_states_topic = joint_states_topic

        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.camera_pose_pub = self.create_publisher(PoseStamped, camera_pose_topic, 10)
        self.optical_pose_pub = self.create_publisher(PoseStamped, optical_pose_topic, 10)
        self.joint_states_sub = self.create_subscription(
            JointState,
            joint_states_topic,
            self.handle_joint_states,
            10,
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.last_log_time = 0.0
        self.last_joint_state_time = None
        self.last_joint_state_names = []

        timer_period = 1.0 / max(1e-6, publish_rate)
        self.timer = self.create_timer(timer_period, self.publish_debug)

        self.get_logger().info(
            "gripper_pose_debug_node started; "
            f"TF={self.base_frame}<-{self.gripper_frame}, "
            f"camera_tf={self.base_frame}<-{self.camera_frame}, "
            f"optical_tf={self.base_frame}<-{self.optical_frame or 'disabled'}, "
            f"markers={marker_topic}, pose={pose_topic}, "
            f"camera_pose={camera_pose_topic}, optical_pose={optical_pose_topic}, "
            f"joint_states={joint_states_topic}"
        )

    def handle_joint_states(self, msg):
        self.last_joint_state_time = self.get_clock().now()
        self.last_joint_state_names = list(msg.name)

    def publish_debug(self):
        try:
            stamped = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.gripper_frame,
                Time(),
            )
        except Exception as exc:
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self.last_log_time >= self.print_period:
                self.get_logger().warn(
                    f"Waiting for TF {self.base_frame}<-{self.gripper_frame}: {exc}. "
                    f"{self.joint_state_status()}"
                )
                self.last_log_time = now
            return

        xyz, rot = self.transform_to_xyz_rot(stamped)
        rpy = rotation_matrix_to_rpy(rot)
        rpy_deg = [math.degrees(value) for value in rpy]

        try:
            camera_stamped = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                Time(),
            )
        except Exception as exc:
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self.last_log_time >= self.print_period:
                self.get_logger().warn(
                    f"Waiting for TF {self.base_frame}<-{self.camera_frame}: {exc}. "
                    "If you just changed the camera frame in the URDF, rebuild and restart the launch."
                )
                self.last_log_time = now
            return

        camera_xyz, camera_rot = self.transform_to_xyz_rot(camera_stamped)
        camera_rpy_deg = [
            math.degrees(value) for value in rotation_matrix_to_rpy(camera_rot)
        ]

        optical_stamped = None
        optical_xyz = None
        optical_rot = None
        optical_rpy_deg = None
        if self.optical_frame:
            try:
                optical_stamped = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.optical_frame,
                    Time(),
                )
                optical_xyz, optical_rot = self.transform_to_xyz_rot(optical_stamped)
                optical_rpy_deg = [
                    math.degrees(value)
                    for value in rotation_matrix_to_rpy(optical_rot)
                ]
            except Exception as exc:
                now = self.get_clock().now().nanoseconds / 1e9
                if now - self.last_log_time >= self.print_period:
                    self.get_logger().warn(
                        f"Waiting for TF {self.base_frame}<-{self.optical_frame}: {exc}. "
                        "Set optical_frame:= if you want to disable this lookup."
                    )
                    self.last_log_time = now
                return

        self.publish_pose(self.pose_pub, stamped.header.stamp, xyz, rot)
        self.publish_pose(
            self.camera_pose_pub,
            camera_stamped.header.stamp,
            camera_xyz,
            camera_rot,
        )
        if optical_stamped is not None:
            self.publish_pose(
                self.optical_pose_pub,
                optical_stamped.header.stamp,
                optical_xyz,
                optical_rot,
            )

        self.marker_pub.publish(
            self.make_markers(
                stamped.header.stamp,
                xyz,
                rot,
                rpy_deg,
                camera_xyz,
                camera_rot,
                camera_rpy_deg,
                optical_xyz,
                optical_rot,
                optical_rpy_deg,
            )
        )

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_log_time >= self.print_period:
            self.get_logger().info(
                f"\n{self.gripper_frame} in {self.base_frame}: "
                f"xyz=({xyz[0] * 100.0:+.1f}, {xyz[1] * 100.0:+.1f}, "
                f"{xyz[2] * 100.0:+.1f}) cm, "
                f"rpy=({rpy_deg[0]:+.1f}, {rpy_deg[1]:+.1f}, {rpy_deg[2]:+.1f}) deg"
                f"\n{self.camera_frame} in {self.base_frame}: "
                f"xyz=({camera_xyz[0] * 100.0:+.1f}, "
                f"{camera_xyz[1] * 100.0:+.1f}, "
                f"{camera_xyz[2] * 100.0:+.1f}) cm, "
                f"rpy=({camera_rpy_deg[0]:+.1f}, {camera_rpy_deg[1]:+.1f}, "
                f"{camera_rpy_deg[2]:+.1f}) deg"
                + (
                    ""
                    if optical_xyz is None
                    else (
                        f"\n{self.optical_frame} in {self.base_frame}: "
                        f"xyz=({optical_xyz[0] * 100.0:+.1f}, "
                        f"{optical_xyz[1] * 100.0:+.1f}, "
                        f"{optical_xyz[2] * 100.0:+.1f}) cm, "
                        f"rpy=({optical_rpy_deg[0]:+.1f}, "
                        f"{optical_rpy_deg[1]:+.1f}, "
                        f"{optical_rpy_deg[2]:+.1f}) deg"
                    )
                )
            )
            self.last_log_time = now

    def joint_state_status(self):
        if self.last_joint_state_time is None:
            return (
                f"No {self.joint_states_topic} received yet; "
                "robot_state_publisher cannot connect movable links."
            )

        age = (
            self.get_clock().now().nanoseconds
            - self.last_joint_state_time.nanoseconds
        ) / 1e9
        names = ",".join(self.last_joint_state_names[:6])
        return f"Last {self.joint_states_topic} age={age:.2f}s names=[{names}]"

    def transform_to_xyz_rot(self, stamped):
        translation = stamped.transform.translation
        rotation = stamped.transform.rotation
        xyz = np.array([translation.x, translation.y, translation.z], dtype=float)
        rot = quaternion_to_rotation_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
        return xyz, rot

    def publish_pose(self, publisher, stamp, xyz, rot):
        quat = rotation_matrix_to_quaternion(rot)
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = stamp
        pose.pose.position.x = float(xyz[0])
        pose.pose.position.y = float(xyz[1])
        pose.pose.position.z = float(xyz[2])
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])
        publisher.publish(pose)

    def make_axis_marker(self, marker_id, stamp, namespace, start, axis, color, length):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [
            make_point(start),
            make_point(start + length * axis),
        ]
        marker.scale.x = self.axis_radius
        marker.scale.y = self.axis_radius * 2.8
        marker.scale.z = self.axis_radius * 3.8
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        return marker

    def make_text_marker(self, marker_id, stamp, namespace, label, xyz, rpy_deg, z_offset):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = float(xyz[0])
        marker.pose.position.y = float(xyz[1])
        marker.pose.position.z = float(xyz[2] + z_offset)
        marker.pose.orientation.w = 1.0
        marker.scale.z = self.text_scale
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = (
            f"{label}\n"
            f"xyz {xyz[0] * 100.0:+.1f} {xyz[1] * 100.0:+.1f} "
            f"{xyz[2] * 100.0:+.1f} cm\n"
            f"rpy {rpy_deg[0]:+.1f} {rpy_deg[1]:+.1f} {rpy_deg[2]:+.1f} deg"
        )
        return marker

    def append_frame_markers(
        self,
        markers,
        base_id,
        stamp,
        namespace,
        label,
        xyz,
        rot,
        rpy_deg,
        length,
    ):
        markers.extend(
            [
                self.make_axis_marker(
                    base_id, stamp, namespace, xyz, rot[:, 0], (1.0, 0.0, 0.0), length
                ),
                self.make_axis_marker(
                    base_id + 1,
                    stamp,
                    namespace,
                    xyz,
                    rot[:, 1],
                    (0.0, 1.0, 0.0),
                    length,
                ),
                self.make_axis_marker(
                    base_id + 2,
                    stamp,
                    namespace,
                    xyz,
                    rot[:, 2],
                    (0.1, 0.35, 1.0),
                    length,
                ),
                self.make_text_marker(
                    base_id + 3,
                    stamp,
                    namespace,
                    label,
                    xyz,
                    rpy_deg,
                    length * 1.25,
                ),
            ]
        )

    def make_markers(
        self,
        stamp,
        xyz,
        rot,
        rpy_deg,
        camera_xyz,
        camera_rot,
        camera_rpy_deg,
        optical_xyz,
        optical_rot,
        optical_rpy_deg,
    ):
        marker_list = []
        self.append_frame_markers(
            marker_list,
            0,
            stamp,
            "gripper_axes",
            self.gripper_frame,
            xyz,
            rot,
            rpy_deg,
            self.axis_length,
        )
        self.append_frame_markers(
            marker_list,
            10,
            stamp,
            "camera_axes",
            self.camera_frame,
            camera_xyz,
            camera_rot,
            camera_rpy_deg,
            self.camera_axis_length,
        )
        if optical_xyz is not None:
            self.append_frame_markers(
                marker_list,
                20,
                stamp,
                "optical_axes",
                self.optical_frame,
                optical_xyz,
                optical_rot,
                optical_rpy_deg,
                self.optical_axis_length,
            )
        markers = MarkerArray()
        markers.markers = marker_list
        return markers


def main(args=None):
    rclpy.init(args=args)
    node = GripperPoseDebugNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

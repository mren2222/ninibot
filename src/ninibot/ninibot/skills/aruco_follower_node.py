# flake8: noqa
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import tf2_ros
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from arm_controller.kinematics.cylindrical import CylindricalArmKinematics
from ninibot.task_params import (
    ARUCO_FOLLOWER_PARAMS,
)
from vision_controller.aruco_detection import destroy_windows
from vision_controller.aruco_detection import estimate_marker_poses
from vision_controller.aruco_detection import load_camera_info
from vision_controller.aruco_detection import make_detector
from vision_controller.aruco_detection import normalize_camera_source
from vision_controller.aruco_detection import open_camera as open_cv_camera
from vision_controller.aruco_detection import show_window
from vision_controller.aruco_tf import transform_camera_point_to_base_xyz


@dataclass
class ArucoFollowerDetection:
    marker_base_xyz: np.ndarray
    target_base_xyz: np.ndarray
    marker_camera_xyz: np.ndarray
    target_camera_xyz: np.ndarray
    marker_rvec: np.ndarray
    marker_tvec: np.ndarray
    corners: object
    ids: object


def marker_qos():
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    qos.reliability = ReliabilityPolicy.RELIABLE
    return qos


class ArucoFollowerNode(Node):
    def __init__(self):
        super().__init__("aruco_follower_node")
        self.declare_parameters(
            namespace="",
            parameters=list(ARUCO_FOLLOWER_PARAMS.items()),
        )

        self.command_topic = self.get_parameter("command_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.target_topic = self.get_parameter("target_topic").value
        self.real_joint_states_topic = self.get_parameter("real_joint_states_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.debug_marker_topic = self.get_parameter("debug_marker_topic").value

        self.camera_device = str(self.get_parameter("camera_device").value)
        self.camera_index = int(self.get_parameter("camera_index").value)
        self.frame_width = int(self.get_parameter("frame_width").value)
        self.frame_height = int(self.get_parameter("frame_height").value)
        self.show_window_enabled = bool(self.get_parameter("show_window").value)
        self.window_name = self.get_parameter("window_name").value
        self.publish_debug_image_enabled = bool(
            self.get_parameter("publish_debug_image").value
        )

        self.calibration_path = self.get_parameter("calibration_path").value
        self.aruco_dictionary = self.get_parameter("aruco_dictionary").value
        self.target_marker_id = int(self.get_parameter("target_marker_id").value)
        self.marker_length = float(self.get_parameter("marker_length").value)
        self.base_frame = self.get_parameter("base_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.gripper_frame = self.get_parameter("gripper_frame").value
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.target_marker_offset_xyz = np.asarray(
            [
                float(self.get_parameter("target_x_offset").value),
                float(self.get_parameter("target_y_offset").value),
                float(self.get_parameter("target_z_offset").value),
            ],
            dtype=float,
        )

        self.control_rate = float(self.get_parameter("control_rate").value)
        self.align_tolerance_m = float(self.get_parameter("align_tolerance_m").value)
        self.max_xy_step_m = float(self.get_parameter("max_xy_step_m").value)
        self.min_command_period_sec = float(
            self.get_parameter("min_command_period_sec").value
        )
        self.hold_gripper_percent = float(
            self.get_parameter("hold_gripper_percent").value
        )
        self.carry_z = float(self.get_parameter("carry_z").value)
        self.pre_pan_radius = float(self.get_parameter("pre_pan_radius").value)
        self.prep_joint_goal_tolerance = math.radians(
            float(self.get_parameter("prep_joint_goal_tolerance_deg").value)
        )
        self.prep_settle_cycles = int(self.get_parameter("prep_settle_cycles").value)
        self.prep_target_check_joint_count = max(
            1,
            min(4, int(self.get_parameter("prep_target_check_joint_count").value)),
        )
        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.wrist_roll_deg = float(self.get_parameter("wrist_roll_deg").value)

        self.kinematics = CylindricalArmKinematics(
            l1=float(self.get_parameter("l1").value),
            l2=float(self.get_parameter("l2").value),
            horizontal_tool_offset=float(
                self.get_parameter("horizontal_tool_offset").value
            ),
            z_tool_offset=float(self.get_parameter("z_tool_offset").value),
            theta_joint_sign=float(self.get_parameter("theta_joint_sign").value),
            theta_joint_offset_deg=float(
                self.get_parameter("theta_joint_offset_deg").value
            ),
            base_min_deg=float(self.get_parameter("base_min_deg").value),
            base_max_deg=float(self.get_parameter("base_max_deg").value),
            shoulder_min_deg=float(self.get_parameter("shoulder_min_deg").value),
            shoulder_max_deg=float(self.get_parameter("shoulder_max_deg").value),
            elbow_min_deg=float(self.get_parameter("elbow_min_deg").value),
            elbow_max_deg=float(self.get_parameter("elbow_max_deg").value),
            wrist_flex_min_deg=float(self.get_parameter("wrist_flex_min_deg").value),
            wrist_flex_max_deg=float(self.get_parameter("wrist_flex_max_deg").value),
            horizontal_wrist_pitch_offset_deg=float(
                self.get_parameter("horizontal_wrist_pitch_offset_deg").value
            ),
        )

        self.command_sub = self.create_subscription(
            String,
            self.command_topic,
            self.command_callback,
            10,
        )
        self.real_joint_sub = self.create_subscription(
            JointState,
            self.real_joint_states_topic,
            self.real_joint_callback,
            10,
        )
        self.joint_target_pub = self.create_publisher(JointState, self.target_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.debug_marker_pub = self.create_publisher(
            MarkerArray,
            self.debug_marker_topic,
            marker_qos(),
        )
        self.debug_image_pub = None
        if self.publish_debug_image_enabled:
            self.debug_image_pub = self.create_publisher(
                Image,
                self.debug_image_topic,
                10,
            )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.active = False
        self.status = "IDLE"
        self.last_status_log = ""
        self.cap = None
        self.camera_ok = False
        self.window_ok = True
        self.locked_z = None
        self.latest_joint_positions = None
        self.prep_target = None
        self.prep_done = False
        self.prep_settle_count = 0
        self.last_command_time = 0.0
        self.last_warn_times = {}

        try:
            (
                self.calibration_width,
                self.calibration_height,
                self.camera_matrix,
                self.dist_coeffs,
            ) = load_camera_info(self.calibration_path)
            self.detect_markers = make_detector(self.aruco_dictionary)
        except Exception as exc:
            self.camera_matrix = None
            self.dist_coeffs = None
            self.detect_markers = None
            self.set_status("CAMERA_ERROR")
            self.get_logger().error(
                f"aruco follower: failed to load camera calibration: {exc}"
            )

        self.timer = self.create_timer(1.0 / max(self.control_rate, 1.0), self.loop)
        self.get_logger().info(
            "aruco_follower_node started; "
            f"marker_id={self.target_marker_id}, "
            f"marker_offset_cm={self.format_cm(self.target_marker_offset_xyz)}, "
            f"tip_frame={self.gripper_frame}, "
            f"carry_z={self.carry_z:+.3f}m, "
            f"pre_pan_radius={self.pre_pan_radius:.3f}m, "
            f"command_topic={self.command_topic}, "
            f"debug_marker_topic={self.debug_marker_topic}"
        )

        if self.auto_start:
            self.start()

    def command_callback(self, msg):
        command = msg.data.strip().lower()
        if command in ("start", "go", "align"):
            self.start()
        elif command in ("stop", "cancel"):
            self.stop("STOPPED")
        elif command == "reset":
            self.stop("IDLE")
            self.clear_debug_markers()
        else:
            self.get_logger().warn(f"Unknown aruco follower command: {msg.data}")

    def start(self):
        if self.detect_markers is None:
            self.set_status("CAMERA_ERROR")
            self.get_logger().error("cannot start: ArUco detector is not ready")
            return

        if not self.open_camera():
            self.set_status("CAMERA_ERROR")
            return

        self.locked_z = self.carry_z
        self.prep_target = None
        self.prep_done = False
        self.prep_settle_count = 0
        self.last_command_time = 0.0
        self.active = True
        self.clear_debug_markers()
        self.set_status("PREP_WAITING_FOR_JOINTS")
        self.get_logger().info(
            f"aruco follower: START z={self.locked_z:+.3f}m"
        )

    def stop(self, status):
        self.active = False
        self.locked_z = None
        self.prep_target = None
        self.prep_done = False
        self.prep_settle_count = 0
        self.release_camera()
        self.set_status(status)
        self.get_logger().info(f"aruco follower: {status}")

    def open_camera(self):
        if self.cap is not None and self.cap.isOpened():
            return True

        source = self.camera_device.strip()
        camera_source = source if source else self.camera_index
        self.cap = open_cv_camera(
            normalize_camera_source(camera_source),
            self.frame_width,
            self.frame_height,
            30,
        )
        if not self.cap.isOpened():
            self.get_logger().error(f"Cannot open camera: {camera_source}")
            self.camera_ok = False
            return False

        self.camera_ok = True
        self.window_ok = True
        self.get_logger().info(f"camera opened: {camera_source}")
        return True

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_ok = False
        destroy_windows()

    def set_status(self, status):
        self.status = status
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        if status != self.last_status_log:
            self.get_logger().info(f"status: {status}")
            self.last_status_log = status

    def warn_throttled(self, key, text, period_sec=2.0):
        now = time.monotonic()
        if now - self.last_warn_times.get(key, 0.0) >= period_sec:
            self.get_logger().warn(text)
            self.last_warn_times[key] = now

    def real_joint_callback(self, msg):
        name_to_position = {
            name: position
            for name, position in zip(msg.name, msg.position)
        }
        if all(str(i) in name_to_position for i in range(1, 7)):
            self.latest_joint_positions = [
                name_to_position[str(i)]
                for i in range(1, 7)
            ]

    def current_cylindrical_pose(self):
        if self.latest_joint_positions is None:
            return None

        return self.kinematics.cylindrical_from_joint_degs(
            math.degrees(self.latest_joint_positions[0]),
            math.degrees(self.latest_joint_positions[1]),
            math.degrees(self.latest_joint_positions[2]),
        )

    def safe_pre_pan_radius(self, current_radius):
        if self.pre_pan_radius <= 0.0:
            return current_radius

        return min(current_radius, self.pre_pan_radius)

    def current_tip_base_xyz(self):
        try:
            stamped = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.gripper_frame,
                Time(),
            )
        except Exception as exc:
            self.warn_throttled(
                "tip_tf",
                f"aruco follower: waiting for TF "
                f"{self.base_frame}->{self.gripper_frame}: {exc}",
            )
            return None

        translation = stamped.transform.translation
        return np.asarray(
            [
                translation.x,
                translation.y,
                translation.z,
            ],
            dtype=float,
        )

    def detect_target(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detect_markers(gray)
        if ids is None:
            return None, corners, ids

        rvecs, tvecs = estimate_marker_poses(
            corners,
            self.marker_length,
            self.camera_matrix,
            self.dist_coeffs,
        )
        target_index = None
        for index, marker_id in enumerate(ids.flatten()):
            if int(marker_id) == self.target_marker_id:
                target_index = index
                break

        if target_index is None or target_index >= len(tvecs):
            return None, corners, ids

        marker_tvec = np.asarray(tvecs[target_index], dtype=float).reshape(3)
        marker_rvec = np.asarray(rvecs[target_index], dtype=float).reshape(3)
        marker_rotation_camera, _ = cv2.Rodrigues(marker_rvec.reshape(3, 1))
        target_camera_xyz = marker_tvec + marker_rotation_camera @ self.target_marker_offset_xyz

        try:
            marker_base_xyz = transform_camera_point_to_base_xyz(
                self,
                self.tf_buffer,
                marker_tvec,
                self.base_frame,
                self.camera_frame,
                self.tf_timeout_sec,
                spin=False,
            )
            target_base_xyz = transform_camera_point_to_base_xyz(
                self,
                self.tf_buffer,
                target_camera_xyz,
                self.base_frame,
                self.camera_frame,
                self.tf_timeout_sec,
                spin=False,
            )
        except Exception as exc:
            self.warn_throttled(
                "tf",
                f"aruco follower: waiting for TF: {exc}",
            )
            return None, corners, ids

        return (
            ArucoFollowerDetection(
                marker_base_xyz=np.asarray(marker_base_xyz, dtype=float),
                target_base_xyz=np.asarray(target_base_xyz, dtype=float),
                marker_camera_xyz=marker_tvec,
                target_camera_xyz=np.asarray(target_camera_xyz, dtype=float),
                marker_rvec=marker_rvec,
                marker_tvec=marker_tvec,
                corners=corners,
                ids=ids,
            ),
            corners,
            ids,
        )

    def control_target_xyz(self, detection):
        if detection is None or self.locked_z is None:
            return None

        target = np.asarray(detection.target_base_xyz, dtype=float).copy()
        target[2] = self.locked_z
        return target

    def joint_target_from_solution(self, solution, gripper_percent):
        wrist_flex_deg = self.horizontal_joint4_deg(
            solution.joint2_deg,
            solution.joint3_deg,
        )
        if not self.kinematics.in_range(
            wrist_flex_deg,
            self.kinematics.wrist_flex_min_deg,
            self.kinematics.wrist_flex_max_deg,
        ):
            self.warn_throttled(
                "joint4",
                "aruco follower: realtime horizontal j4 out of range: "
                f"{wrist_flex_deg:.1f} deg",
            )
            return None

        return [
            math.radians(solution.base_deg),
            math.radians(solution.joint2_deg),
            math.radians(solution.joint3_deg),
            math.radians(wrist_flex_deg),
            math.radians(self.wrist_roll_deg),
            self.gripper_percent_to_rad(gripper_percent),
        ]

    def apply_realtime_joint4_target(self, target):
        if target is None or len(target) < 4:
            return False

        if self.latest_joint_positions is not None:
            joint2_rad = self.latest_joint_positions[1]
            joint3_rad = self.latest_joint_positions[2]
        else:
            joint2_rad = target[1]
            joint3_rad = target[2]

        joint4_deg = self.horizontal_joint4_deg(
            math.degrees(joint2_rad),
            math.degrees(joint3_rad),
        )
        if not self.kinematics.in_range(
            joint4_deg,
            self.kinematics.wrist_flex_min_deg,
            self.kinematics.wrist_flex_max_deg,
        ):
            self.warn_throttled(
                "joint4",
                "aruco follower: realtime horizontal j4 out of range: "
                f"{joint4_deg:.1f} deg",
            )
            return False

        target[3] = math.radians(joint4_deg)
        return True

    def publish_joint_target(self, target):
        if target is None:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["1", "2", "3", "4", "5", "6"]
        msg.position = target
        self.joint_target_pub.publish(msg)

    def make_prep_target(self):
        pose = self.current_cylindrical_pose()
        if pose is None:
            return None

        prep_radius = self.safe_pre_pan_radius(pose.radius)
        solution = self.kinematics.cylindrical_ik(
            prep_radius,
            pose.theta_deg,
            self.locked_z,
        )
        if solution is None:
            self.warn_throttled(
                "prep_ik",
                "aruco follower prep IK failed: "
                f"{self.kinematics.last_error}; "
                f"radius={prep_radius:.3f} theta={pose.theta_deg:.1f} "
                f"z={self.locked_z:.3f}",
            )
            return None

        target = self.joint_target_from_solution(
            solution,
            self.hold_gripper_percent,
        )
        if target is None:
            return None

        self.get_logger().info(
            "aruco follower: prep target "
            f"radius={prep_radius:.3f}m theta={pose.theta_deg:.1f}deg "
            f"z={self.locked_z:.3f}m"
        )
        return target

    def joint_errors_deg(self, target):
        if target is None or self.latest_joint_positions is None:
            return []

        count = min(
            self.prep_target_check_joint_count,
            len(target),
            len(self.latest_joint_positions),
        )
        return [
            abs(math.degrees(current - goal))
            for current, goal in zip(
                self.latest_joint_positions[:count],
                target[:count],
            )
        ]

    def prep_target_reached(self):
        errors_deg = self.joint_errors_deg(self.prep_target)
        if not errors_deg:
            return False

        tolerance_deg = math.degrees(self.prep_joint_goal_tolerance)
        return all(error_deg <= tolerance_deg for error_deg in errors_deg)

    def update_prep(self):
        if self.prep_done:
            return True

        if self.latest_joint_positions is None:
            self.set_status("PREP_WAITING_FOR_JOINTS")
            return False

        if self.prep_target is None:
            self.prep_target = self.make_prep_target()
            self.prep_settle_count = 0
            if self.prep_target is None:
                self.set_status("PREP_IK_FAILED")
                return False

        if not self.apply_realtime_joint4_target(self.prep_target):
            self.set_status("PREP_IK_FAILED")
            return False

        self.publish_joint_target(self.prep_target)
        self.set_status("PREP_MOVING")

        if not self.prep_target_reached():
            self.prep_settle_count = 0
            return False

        self.prep_settle_count += 1
        if self.prep_settle_count < self.prep_settle_cycles:
            return False

        self.prep_done = True
        self.last_command_time = 0.0
        self.set_status("SEARCHING_MARKER")
        self.get_logger().info("aruco follower: prep pose reached")
        return True

    def command_toward_target(self, tip_base_xyz, target_base_xyz):
        error_xy = target_base_xyz[:2] - tip_base_xyz[:2]
        distance_xy = float(np.linalg.norm(error_xy))
        if distance_xy <= self.align_tolerance_m:
            return "ALIGNED", distance_xy

        now = time.monotonic()
        if now - self.last_command_time < self.min_command_period_sec:
            return "ALIGNING", distance_xy

        step_xy = error_xy
        if distance_xy > self.max_xy_step_m:
            step_xy = error_xy / distance_xy * self.max_xy_step_m

        next_xyz = np.asarray(
            [
                tip_base_xyz[0] + step_xy[0],
                tip_base_xyz[1] + step_xy[1],
                target_base_xyz[2],
            ],
            dtype=float,
        )
        radius = math.hypot(float(next_xyz[0]), float(next_xyz[1]))
        theta_deg = math.degrees(math.atan2(float(next_xyz[1]), float(next_xyz[0])))
        solution = self.kinematics.cylindrical_ik(radius, theta_deg, float(next_xyz[2]))
        if solution is None:
            self.warn_throttled(
                "ik",
                "aruco follower IK failed: "
                f"{self.kinematics.last_error}; "
                f"next={self.format_cm(next_xyz)} "
                f"target={self.format_cm(target_base_xyz)} "
                f"tip={self.format_cm(tip_base_xyz)}",
            )
            return "IK_FAILED", distance_xy

        target = self.joint_target_from_solution(
            solution,
            self.hold_gripper_percent,
        )
        if target is None:
            return "IK_FAILED", distance_xy

        if not self.apply_realtime_joint4_target(target):
            return "IK_FAILED", distance_xy

        self.publish_joint_target(target)
        self.last_command_time = now
        return "ALIGNING", distance_xy

    def horizontal_joint4_deg(self, joint2_deg, joint3_deg):
        return self.kinematics.wrist_flex_deg(joint2_deg, joint3_deg)

    def gripper_percent_to_rad(self, percent):
        percent = max(0.0, min(100.0, float(percent)))
        gripper_min_rad = -0.174533
        gripper_max_rad = 1.74533
        return gripper_min_rad + (percent / 100.0) * (
            gripper_max_rad - gripper_min_rad
        )

    def image_to_msg(self, image, encoding):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = image.strides[0]
        msg.data = image.tobytes()
        return msg

    def publish_debug_image_msg(self, frame):
        if self.debug_image_pub is None:
            return
        self.debug_image_pub.publish(self.image_to_msg(frame, "bgr8"))

    def make_point(self, xyz):
        point = Point()
        point.x = float(xyz[0])
        point.y = float(xyz[1])
        point.z = float(xyz[2])
        return point

    def apply_marker_lifetime(self, marker):
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 500000000
        return marker

    def sphere_marker(self, marker_id, stamp, namespace, xyz, scale, rgba):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = self.make_point(xyz)
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]
        return self.apply_marker_lifetime(marker)

    def line_marker(self, marker_id, stamp, namespace, points, rgba, width=0.005):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = width
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]
        marker.points = [self.make_point(point) for point in points]
        return self.apply_marker_lifetime(marker)

    def text_marker(self, marker_id, stamp, namespace, label, xyz, rgba):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = float(xyz[0])
        marker.pose.position.y = float(xyz[1])
        marker.pose.position.z = float(xyz[2] + 0.035)
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.03
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]
        marker.text = f"{label}\n{self.format_cm(xyz)} cm"
        return self.apply_marker_lifetime(marker)

    def clear_debug_markers(self):
        marker = Marker()
        marker.action = Marker.DELETEALL
        markers = MarkerArray()
        markers.markers = [marker]
        self.debug_marker_pub.publish(markers)

    def publish_debug_markers(self, detection, tip_base_xyz, target_base_xyz):
        markers = []
        stamp = self.get_clock().now().to_msg()
        if detection is not None:
            markers.extend(
                [
                    self.sphere_marker(
                        0,
                        stamp,
                        "aruco_marker",
                        detection.marker_base_xyz,
                        0.025,
                        (0.1, 0.45, 1.0, 0.9),
                    ),
                    self.sphere_marker(
                        1,
                        stamp,
                        "aruco_target",
                        detection.target_base_xyz,
                        0.024,
                        (1.0, 0.55, 0.05, 0.9),
                    ),
                    self.line_marker(
                        2,
                        stamp,
                        "aruco_offset",
                        [detection.marker_base_xyz, detection.target_base_xyz],
                        (1.0, 0.85, 0.05, 0.9),
                        width=0.006,
                    ),
                    self.text_marker(
                        3,
                        stamp,
                        "aruco_marker",
                        f"marker id{self.target_marker_id}",
                        detection.marker_base_xyz,
                        (0.72, 0.86, 1.0, 0.9),
                    ),
                ]
            )

        if tip_base_xyz is not None:
            markers.extend(
                [
                    self.sphere_marker(
                        10,
                        stamp,
                        "gripper_tip",
                        tip_base_xyz,
                        0.026,
                        (0.0, 0.9, 0.35, 0.95),
                    ),
                    self.text_marker(
                        11,
                        stamp,
                        "gripper_tip",
                        "gripper tip",
                        tip_base_xyz,
                        (0.55, 1.0, 0.62, 0.95),
                    ),
                ]
            )

        if tip_base_xyz is not None and target_base_xyz is not None:
            markers.extend(
                [
                    self.sphere_marker(
                        20,
                        stamp,
                        "control_target",
                        target_base_xyz,
                        0.03,
                        (1.0, 0.1, 0.1, 0.95),
                    ),
                    self.line_marker(
                        21,
                        stamp,
                        "xy_error",
                        [
                            tip_base_xyz,
                            np.asarray(
                                [
                                    target_base_xyz[0],
                                    target_base_xyz[1],
                                    tip_base_xyz[2],
                                ],
                                dtype=float,
                            ),
                        ],
                        (1.0, 0.1, 0.1, 0.95),
                        width=0.006,
                    ),
                    self.text_marker(
                        22,
                        stamp,
                        "control_target",
                        "xy target",
                        target_base_xyz,
                        (1.0, 0.45, 0.45, 0.95),
                    ),
                ]
            )

        marker_array = MarkerArray()
        marker_array.markers = markers
        self.debug_marker_pub.publish(marker_array)

    def draw_overlay(self, frame, detection, tip_base_xyz, target_base_xyz, distance_xy):
        if detection is not None and detection.corners is not None and detection.ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, detection.corners, detection.ids)
            if hasattr(cv2, "drawFrameAxes"):
                cv2.drawFrameAxes(
                    frame,
                    self.camera_matrix,
                    self.dist_coeffs,
                    detection.marker_rvec.reshape(3, 1),
                    detection.marker_tvec.reshape(3, 1),
                    self.marker_length * 0.5,
                )
            try:
                image_points, _ = cv2.projectPoints(
                    self.target_marker_offset_xyz.reshape(1, 3),
                    detection.marker_rvec.reshape(3, 1),
                    detection.marker_tvec.reshape(3, 1),
                    self.camera_matrix,
                    self.dist_coeffs,
                )
                px, py = image_points.reshape(-1, 2)[0]
                cv2.circle(frame, (int(px), int(py)), 6, (0, 165, 255), -1)
            except cv2.error:
                pass

        lines = [
            f"status={self.status}",
            f"marker_id={self.target_marker_id}",
            f"marker={self.format_cm(detection.marker_base_xyz) if detection else '(n/a, n/a, n/a)'} cm",
            f"tip={self.format_cm(tip_base_xyz)} cm",
            f"target={self.format_cm(target_base_xyz)} cm",
            f"xy_dist={(distance_xy * 100.0):.1f} cm" if distance_xy is not None else "xy_dist=n/a",
        ]
        for index, line in enumerate(lines):
            yy = 28 + index * 25
            cv2.putText(
                frame,
                line,
                (14, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (14, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def show_debug_window(self, frame):
        if not self.show_window_enabled or not self.window_ok:
            return

        key, ok = show_window(self.window_name, frame, 1)
        self.window_ok = ok
        if key in (ord("q"), 27):
            self.stop("STOPPED")

    def loop(self):
        if not self.active:
            return

        if not self.camera_ok or self.cap is None:
            self.set_status("CAMERA_ERROR")
            return

        ok, frame = self.cap.read()
        if not ok:
            self.set_status("CAMERA_ERROR")
            self.warn_throttled("camera_read", "Cannot read camera frame")
            return

        result = self.detect_target(frame)
        if result[0] is None:
            detection, corners, ids = None, result[1], result[2]
        else:
            detection, corners, ids = result

        if detection is None and corners is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        tip_base_xyz = self.current_tip_base_xyz()
        target_base_xyz = self.control_target_xyz(detection)
        distance_xy = None

        if not self.prep_done:
            self.update_prep()
        elif detection is None:
            self.set_status("SEARCHING_MARKER")
        elif tip_base_xyz is None:
            self.set_status("WAITING_FOR_TIP_TF")
        elif target_base_xyz is None:
            self.set_status("WAITING_FOR_Z")
        else:
            next_status, distance_xy = self.command_toward_target(
                tip_base_xyz,
                target_base_xyz,
            )
            self.set_status(next_status)

        if tip_base_xyz is not None and target_base_xyz is not None:
            distance_xy = float(np.linalg.norm(target_base_xyz[:2] - tip_base_xyz[:2]))

        self.publish_debug_markers(detection, tip_base_xyz, target_base_xyz)
        self.draw_overlay(frame, detection, tip_base_xyz, target_base_xyz, distance_xy)
        self.publish_debug_image_msg(frame)
        self.show_debug_window(frame)

    def format_cm(self, xyz):
        if xyz is None:
            return "(n/a, n/a, n/a)"
        return (
            f"({xyz[0] * 100.0:+.1f}, "
            f"{xyz[1] * 100.0:+.1f}, "
            f"{xyz[2] * 100.0:+.1f})"
        )

    def destroy_node(self):
        self.release_camera()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoFollowerNode()
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

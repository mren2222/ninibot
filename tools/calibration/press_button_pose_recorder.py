#!/usr/bin/env python3
"""Record manual press-button pose offsets from marker, grippertip, and joints."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import time

import numpy as np

try:
    from geometry_msgs.msg import PointStamped
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.time import Time
    from sensor_msgs.msg import JointState
    import tf2_ros
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This tool needs ROS 2 Python packages. Source ROS first, for example: "
        "source /opt/ros/jazzy/setup.bash"
    ) from exc

TOOL_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = TOOL_DIR.parents[1]
VISION_SRC = WORKSPACE_ROOT / "src" / "vision_controller"
if VISION_SRC.exists():
    sys.path.insert(0, str(VISION_SRC))
NINIBOT_SRC = WORKSPACE_ROOT / "src" / "ninibot"
if NINIBOT_SRC.exists():
    sys.path.insert(0, str(NINIBOT_SRC))

from ninibot.task_params import PRESS_BUTTON_PARAMS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Listen to ArUco marker_base, joint states, and grippertip TF, then "
            "print offsets useful for press_button_node calibration."
        )
    )
    parser.add_argument("--marker-base-topic", default="/aruco_extrinsics/marker_base")
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--base-frame", default=str(PRESS_BUTTON_PARAMS["base_frame"]))
    parser.add_argument("--gripper-frame", default=str(PRESS_BUTTON_PARAMS["gripper_frame"]))
    parser.add_argument(
        "--target-x-offset",
        type=float,
        default=float(PRESS_BUTTON_PARAMS["target_x_offset"]),
    )
    parser.add_argument(
        "--target-y-offset",
        type=float,
        default=float(PRESS_BUTTON_PARAMS["target_y_offset"]),
    )
    parser.add_argument(
        "--target-z-offset",
        type=float,
        default=float(PRESS_BUTTON_PARAMS["target_z_offset"]),
    )
    parser.add_argument(
        "--tf-timeout",
        type=float,
        default=float(PRESS_BUTTON_PARAMS["tf_timeout_sec"]),
    )
    parser.add_argument("--print-every", type=float, default=0.5)
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional CSV file to append printed samples, for example data/calibration/press_button_pose_samples.csv.",
    )
    return parser


def format_cm(vector, digits: int = 1) -> str:
    value = np.asarray(vector, dtype=float) * 100.0
    return f"({value[0]:+.{digits}f}, {value[1]:+.{digits}f}, {value[2]:+.{digits}f})"


def format_m(vector, digits: int = 4) -> str:
    value = np.asarray(vector, dtype=float)
    return f"({value[0]:+.{digits}f}, {value[1]:+.{digits}f}, {value[2]:+.{digits}f})"


class PressButtonPoseRecorder(Node):
    def __init__(self, args):
        super().__init__("press_button_pose_recorder")
        self.args = args
        self.marker_base_xyz = None
        self.marker_stamp = None
        self.joint_positions = {}
        self.joint_stamp = None
        self.last_print_time = 0.0
        self.csv_file = None
        self.csv_writer = None

        self.offset_xyz = np.array(
            [
                args.target_x_offset,
                args.target_y_offset,
                args.target_z_offset,
            ],
            dtype=float,
        )

        self.marker_sub = self.create_subscription(
            PointStamped,
            args.marker_base_topic,
            self.marker_callback,
            10,
        )
        self.joint_sub = self.create_subscription(
            JointState,
            args.joint_states_topic,
            self.joint_callback,
            10,
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.05, self.timer_callback)

        if args.output_csv:
            output_path = Path(args.output_csv).expanduser()
            if not output_path.is_absolute():
                output_path = WORKSPACE_ROOT / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not output_path.exists()
            self.csv_file = output_path.open("a", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            if new_file:
                self.csv_writer.writerow(
                    [
                        "stamp_sec",
                        "marker_x_m",
                        "marker_y_m",
                        "marker_z_m",
                        "grippertip_x_m",
                        "grippertip_y_m",
                        "grippertip_z_m",
                        "tip_minus_marker_x_m",
                        "tip_minus_marker_y_m",
                        "tip_minus_marker_z_m",
                        "target_error_x_m",
                        "target_error_y_m",
                        "target_error_z_m",
                        "target_error_norm_m",
                        "joint1_deg",
                        "joint2_deg",
                        "joint3_deg",
                        "joint4_deg",
                        "align_wrist_pitch_deg",
                    ]
                )
            self.get_logger().info(f"Appending samples to {output_path}")

        self.get_logger().info(
            "press_button_pose_recorder started; "
            f"marker_topic={args.marker_base_topic}, joints={args.joint_states_topic}, "
            f"TF={args.base_frame}<-{args.gripper_frame}, "
            f"current_offset_m={format_m(self.offset_xyz)}"
        )

    def destroy_node(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
        super().destroy_node()

    def marker_callback(self, msg: PointStamped):
        self.marker_base_xyz = np.array(
            [msg.point.x, msg.point.y, msg.point.z],
            dtype=float,
        )
        self.marker_stamp = self.get_clock().now()

    def joint_callback(self, msg: JointState):
        self.joint_positions = {
            name: position
            for name, position in zip(msg.name, msg.position)
        }
        self.joint_stamp = self.get_clock().now()

    def lookup_grippertip_base_xyz(self):
        stamped = self.tf_buffer.lookup_transform(
            self.args.base_frame,
            self.args.gripper_frame,
            Time(),
            timeout=Duration(seconds=self.args.tf_timeout),
        )
        translation = stamped.transform.translation
        return np.array([translation.x, translation.y, translation.z], dtype=float)

    def joint_deg(self, name):
        value = self.joint_positions.get(name)
        if value is None:
            return math.nan
        return math.degrees(float(value))

    def latest_age_sec(self, stamp):
        if stamp is None:
            return math.nan
        return (self.get_clock().now() - stamp).nanoseconds / 1e9

    def timer_callback(self):
        now = time.monotonic()
        if now - self.last_print_time < max(0.05, self.args.print_every):
            return
        self.last_print_time = now

        if self.marker_base_xyz is None:
            self.get_logger().info(
                f"waiting for marker on {self.args.marker_base_topic}"
            )
            return
        if not self.joint_positions:
            self.get_logger().info(
                f"waiting for joint states on {self.args.joint_states_topic}"
            )
            return

        try:
            grippertip_base_xyz = self.lookup_grippertip_base_xyz()
        except Exception as exc:
            self.get_logger().info(
                f"waiting for TF {self.args.base_frame}<-{self.args.gripper_frame}: {exc}"
            )
            return

        marker_base_xyz = self.marker_base_xyz
        target_base_xyz = marker_base_xyz + self.offset_xyz
        tip_minus_marker = grippertip_base_xyz - marker_base_xyz
        target_error = grippertip_base_xyz - target_base_xyz
        target_error_norm = float(np.linalg.norm(target_error))

        joint1_deg = self.joint_deg("1")
        joint2_deg = self.joint_deg("2")
        joint3_deg = self.joint_deg("3")
        joint4_deg = self.joint_deg("4")
        align_wrist_pitch_deg = joint2_deg + joint3_deg + joint4_deg

        message = (
            "pose sample: "
            f"marker_base_cm={format_cm(marker_base_xyz)} "
            f"grippertip_base_cm={format_cm(grippertip_base_xyz)} "
            f"tip_minus_marker_cm={format_cm(tip_minus_marker)} "
            f"current_target_error_cm={format_cm(target_error)} "
            f"err={target_error_norm * 100.0:.1f}cm "
            f"q1234_deg=[{joint1_deg:+.1f}, {joint2_deg:+.1f}, "
            f"{joint3_deg:+.1f}, {joint4_deg:+.1f}] "
            f"q234_deg=[{joint2_deg:+.1f}, {joint3_deg:+.1f}, {joint4_deg:+.1f}] "
            f"suggested_offsets_m={format_m(tip_minus_marker)} "
            f"suggested_align_wrist_pitch_deg={align_wrist_pitch_deg:+.1f} "
            f"ages(marker={self.latest_age_sec(self.marker_stamp):.2f}s, "
            f"joints={self.latest_age_sec(self.joint_stamp):.2f}s)"
        )
        self.get_logger().info(message)

        if self.csv_writer is not None:
            self.csv_writer.writerow(
                [
                    f"{self.get_clock().now().nanoseconds / 1e9:.9f}",
                    *[f"{value:.9f}" for value in marker_base_xyz],
                    *[f"{value:.9f}" for value in grippertip_base_xyz],
                    *[f"{value:.9f}" for value in tip_minus_marker],
                    *[f"{value:.9f}" for value in target_error],
                    f"{target_error_norm:.9f}",
                    f"{joint1_deg:.6f}",
                    f"{joint2_deg:.6f}",
                    f"{joint3_deg:.6f}",
                    f"{joint4_deg:.6f}",
                    f"{align_wrist_pitch_deg:.6f}",
                ]
            )
            self.csv_file.flush()


def main() -> int:
    args, ros_args = build_arg_parser().parse_known_args()
    rclpy.init(args=ros_args if ros_args else None)
    node = PressButtonPoseRecorder(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

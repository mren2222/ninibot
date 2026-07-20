# flake8: noqa
import csv
import json
import math
from pathlib import Path
import re
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import String


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
LABELS = {
    "success",
    "failed_detection",
    "failed_grasp",
    "failed_move",
    "failed_press",
    "manual_abort",
    "unknown",
}


class EpisodeLogger(Node):
    def __init__(self):
        super().__init__("episode_logger")

        self.declare_parameter("episode_root", "data/episodes")
        self.declare_parameter("command_topic", "/episode_logger/command")
        self.declare_parameter("image_topic", "/cup_servo/debug_image")
        self.declare_parameter("real_joint_states_topic", "/real_joint_states")
        self.declare_parameter("target_joint_states_topic", "/target_joint_states")
        self.declare_parameter("cup_status_topic", "/cup_servo_status")
        self.declare_parameter("cup_debug_topic", "/cup_servo_debug")
        self.declare_parameter("move_status_topic", "/move_cup_status")
        self.declare_parameter("task_status_topic", "/task_manager_status")
        self.declare_parameter("log_rate", 10.0)
        self.declare_parameter("save_images", False)
        self.declare_parameter("image_stride", 1)
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("auto_start_on_task", True)
        self.declare_parameter("auto_stop_on_task", True)
        self.declare_parameter("task_name", "grasp_and_move_cup")
        self.declare_parameter("detector_backend", "hsv")

        self.episode_root = Path(
            str(self.get_parameter("episode_root").value)
        ).expanduser()
        self.command_topic = self.get_parameter("command_topic").value
        self.image_topic = self.get_parameter("image_topic").value
        self.real_joint_states_topic = self.get_parameter("real_joint_states_topic").value
        self.target_joint_states_topic = self.get_parameter(
            "target_joint_states_topic"
        ).value
        self.cup_status_topic = self.get_parameter("cup_status_topic").value
        self.cup_debug_topic = self.get_parameter("cup_debug_topic").value
        self.move_status_topic = self.get_parameter("move_status_topic").value
        self.task_status_topic = self.get_parameter("task_status_topic").value
        self.log_rate = float(self.get_parameter("log_rate").value)
        self.save_images = bool(self.get_parameter("save_images").value)
        self.image_stride = max(1, int(self.get_parameter("image_stride").value))
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.auto_start_on_task = bool(
            self.get_parameter("auto_start_on_task").value
        )
        self.auto_stop_on_task = bool(
            self.get_parameter("auto_stop_on_task").value
        )
        self.task_name = self.get_parameter("task_name").value
        self.detector_backend = self.get_parameter("detector_backend").value

        self.latest_real_joints = {}
        self.latest_target_joints = {}
        self.prev_target_positions = None
        self.prev_servo_radius = math.nan
        self.prev_servo_z = math.nan
        self.cup_status = ""
        self.cup_debug = ""
        self.move_status = ""
        self.task_status = ""
        self.latest_image_filename = ""
        self.latest_image_stamp_sec = math.nan
        self.received_images = 0
        self.saved_images = 0
        self.rows_written = 0

        self.active = False
        self.episode_id = ""
        self.episode_dir = None
        self.image_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.started_wall = None
        self.started_monotonic = None
        self.ended_wall = None
        self.label = "unknown"
        self.stop_reason = ""

        self.command_sub = self.create_subscription(
            String,
            self.command_topic,
            self.command_callback,
            10,
        )
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )
        self.real_joint_sub = self.create_subscription(
            JointState,
            self.real_joint_states_topic,
            self.real_joint_callback,
            10,
        )
        self.target_joint_sub = self.create_subscription(
            JointState,
            self.target_joint_states_topic,
            self.target_joint_callback,
            10,
        )
        self.cup_status_sub = self.create_subscription(
            String,
            self.cup_status_topic,
            self.cup_status_callback,
            10,
        )
        self.cup_debug_sub = self.create_subscription(
            String,
            self.cup_debug_topic,
            self.cup_debug_callback,
            10,
        )
        self.move_status_sub = self.create_subscription(
            String,
            self.move_status_topic,
            self.move_status_callback,
            10,
        )
        self.task_status_sub = self.create_subscription(
            String,
            self.task_status_topic,
            self.task_status_callback,
            10,
        )

        self.timer = self.create_timer(1.0 / self.log_rate, self.log_tick)
        self.get_logger().info(
            f"episode_logger started; root={self.episode_root}, "
            f"auto_start_on_task={self.auto_start_on_task}, "
            f"save_images={self.save_images}"
        )

    def command_callback(self, msg):
        parts = msg.data.strip().split()
        if not parts:
            return

        command = parts[0].lower()
        if command == "start":
            self.start_episode(reason="manual")
        elif command == "stop":
            label = parts[1].lower() if len(parts) > 1 else "unknown"
            self.stop_episode(label=label, reason="manual")
        elif command == "abort":
            self.stop_episode(label="manual_abort", reason="manual_abort")
        else:
            self.get_logger().warn(f"Unknown episode logger command: {msg.data}")

    def real_joint_callback(self, msg):
        self.latest_real_joints = self.joint_state_to_dict(msg)

    def target_joint_callback(self, msg):
        self.latest_target_joints = self.joint_state_to_dict(msg)

    def cup_status_callback(self, msg):
        self.cup_status = msg.data.strip()
        if (
            self.auto_start_on_task
            and not self.active
            and self.cup_status
            in (
                "SEARCH_PREP",
                "SEARCHING",
                "TRACKING_CENTER",
                "FINE_ALIGN_PREP",
                "FINE_ALIGN",
            )
        ):
            self.start_episode(reason=f"cup_status:{self.cup_status}")

    def cup_debug_callback(self, msg):
        self.cup_debug = msg.data.strip()

    def move_status_callback(self, msg):
        self.move_status = msg.data.strip()
        if (
            self.auto_start_on_task
            and not self.active
            and self.move_status
            in (
                "GRIPPING",
                "LIFTING",
                "RETRACTING_RADIUS",
                "MOVING_TO_TARGET",
                "EXTENDING_RADIUS",
                "LOWERING_TO_RELEASE",
                "RELEASING",
                "RETURNING_SEARCH_PREP",
            )
        ):
            self.start_episode(reason=f"move_status:{self.move_status}")

    def task_status_callback(self, msg):
        status = msg.data.strip()
        self.task_status = status

        if (
            self.auto_start_on_task
            and not self.active
            and status in ("GRASPING_CUP", "MOVING_CUP")
        ):
            self.start_episode(reason=f"task_status:{status}")

        if self.auto_stop_on_task and self.active:
            if status == "DONE":
                self.stop_episode(label="success", reason="task_status:DONE")
            elif status == "FAILED":
                self.stop_episode(label="unknown", reason="task_status:FAILED")

    def joint_state_to_dict(self, msg):
        return {
            str(name): float(position)
            for name, position in zip(msg.name, msg.position)
        }

    def next_episode_id(self):
        self.episode_root.mkdir(parents=True, exist_ok=True)
        existing = []
        for path in self.episode_root.glob("episode_*"):
            if not path.is_dir():
                continue
            match = re.match(r"episode_(\d+)$", path.name)
            if match:
                existing.append(int(match.group(1)))
        next_index = max(existing, default=0) + 1
        return f"episode_{next_index:06d}"

    def start_episode(self, reason):
        if self.active:
            self.get_logger().warn(f"episode already active: {self.episode_id}")
            return

        self.episode_id = self.next_episode_id()
        self.episode_dir = self.episode_root / self.episode_id
        self.image_dir = self.episode_dir / "images"
        if self.save_images:
            self.image_dir.mkdir(parents=True, exist_ok=False)
        else:
            self.episode_dir.mkdir(parents=True, exist_ok=False)

        self.csv_file = (self.episode_dir / "data.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fields())
        self.csv_writer.writeheader()

        self.active = True
        self.started_wall = time.time()
        self.started_monotonic = time.monotonic()
        self.ended_wall = None
        self.label = "unknown"
        self.stop_reason = ""
        self.latest_image_filename = ""
        self.latest_image_stamp_sec = math.nan
        self.received_images = 0
        self.saved_images = 0
        self.rows_written = 0
        self.prev_target_positions = None
        self.prev_servo_radius = math.nan
        self.prev_servo_z = math.nan

        self.write_metadata(label="recording", reason=reason)
        self.get_logger().info(f"episode started: {self.episode_dir}")

    def stop_episode(self, label, reason):
        if not self.active:
            self.get_logger().warn("no active episode to stop")
            return

        label = label if label in LABELS else "unknown"
        self.active = False
        self.ended_wall = time.time()
        self.label = label
        self.stop_reason = reason

        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

        self.write_metadata(label=label, reason=reason)
        self.get_logger().info(
            f"episode stopped: {self.episode_id}, label={label}, "
            f"frames={self.saved_images}, rows={self.rows_written}"
        )

    def write_metadata(self, label, reason):
        if self.episode_dir is None:
            return

        duration_sec = 0.0
        if self.started_wall is not None:
            end_wall = self.ended_wall if self.ended_wall is not None else time.time()
            duration_sec = end_wall - self.started_wall

        metadata = {
            "episode_id": self.episode_id,
            "task": self.task_name,
            "detector_backend": self.detector_backend,
            "label": label,
            "reason": reason,
            "started_at_unix": self.started_wall,
            "ended_at_unix": self.ended_wall,
            "duration_sec": duration_sec,
            "images": self.saved_images,
            "save_images": self.save_images,
            "rows": self.rows_written,
            "topics": {
                "image": self.image_topic,
                "real_joint_states": self.real_joint_states_topic,
                "target_joint_states": self.target_joint_states_topic,
                "cup_status": self.cup_status_topic,
                "cup_debug": self.cup_debug_topic,
                "move_status": self.move_status_topic,
                "task_status": self.task_status_topic,
            },
            "labels": sorted(LABELS),
        }
        metadata_path = self.episode_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def image_callback(self, msg):
        if not self.active or not self.save_images:
            return

        self.received_images += 1
        if self.received_images % self.image_stride != 0:
            return

        image = self.image_msg_to_cv2(msg)
        if image is None:
            return

        filename = f"{self.saved_images:06d}.jpg"
        path = self.image_dir / filename
        ok = cv2.imwrite(
            str(path),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            self.get_logger().warn(f"failed to save image: {path}")
            return

        self.latest_image_filename = f"images/{filename}"
        self.latest_image_stamp_sec = self.stamp_to_sec(msg.header.stamp)
        self.saved_images += 1

    def image_msg_to_cv2(self, msg):
        try:
            if msg.encoding in ("bgr8", "rgb8"):
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height,
                    msg.width,
                    3,
                )
                if msg.encoding == "rgb8":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                return image

            if msg.encoding == "mono8":
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height,
                    msg.width,
                )
                return image

            self.get_logger().warn(f"unsupported image encoding: {msg.encoding}")
            return None
        except ValueError as exc:
            self.get_logger().warn(f"failed to convert image: {exc}")
            return None

    def stamp_to_sec(self, stamp):
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def csv_fields(self):
        fields = [
            "row_index",
            "elapsed_sec",
            "wall_time_unix",
            "image",
            "image_stamp_sec",
            "cup_status",
            "move_status",
            "task_status",
            "error_x",
            "bbox_w",
            "bbox_error",
            "theta_deg",
            "servo_radius",
            "servo_z",
            "joint2_target_deg",
            "joint3_target_deg",
            "joint4_target_deg",
            "gripper_target_rad",
        ]
        for prefix in ("real", "target", "cmd_delta", "target_delta"):
            for index in range(1, 7):
                fields.append(f"{prefix}_j{index}")
        fields.extend([
            "action_delta_theta",
            "action_delta_radius",
            "action_delta_z",
            "action_delta_joint2",
            "action_delta_joint3",
            "action_delta_gripper",
        ])
        return fields

    def log_tick(self):
        if not self.active or self.csv_writer is None:
            return

        now = time.monotonic()
        elapsed_sec = now - self.started_monotonic
        target_positions = self.positions_for(self.latest_target_joints)
        real_positions = self.positions_for(self.latest_real_joints)

        cmd_delta = self.delta_positions(target_positions, real_positions)
        target_delta = self.delta_positions(target_positions, self.prev_target_positions)
        servo_radius = self.parse_debug_value("servo_radius")
        servo_z = self.parse_debug_value("servo_z")

        row = {
            "row_index": self.rows_written,
            "elapsed_sec": f"{elapsed_sec:.6f}",
            "wall_time_unix": f"{time.time():.6f}",
            "image": self.latest_image_filename,
            "image_stamp_sec": self.format_float(self.latest_image_stamp_sec),
            "cup_status": self.cup_status,
            "move_status": self.move_status,
            "task_status": self.task_status,
            "error_x": self.format_float(self.parse_debug_value("error_x")),
            "bbox_w": self.format_float(self.parse_debug_value("bbox_w")),
            "bbox_error": self.format_float(self.parse_debug_value("bbox_error")),
            "theta_deg": self.format_float(self.parse_debug_value("theta_deg")),
            "servo_radius": self.format_float(servo_radius),
            "servo_z": self.format_float(servo_z),
            "joint2_target_deg": self.format_float(
                self.parse_debug_value("joint2_target")
            ),
            "joint3_target_deg": self.format_float(
                self.parse_debug_value("joint3_target")
            ),
            "joint4_target_deg": self.format_float(
                self.parse_debug_value("joint4_target")
            ),
            "gripper_target_rad": self.format_float(target_positions[5]),
        }

        self.add_positions(row, "real", real_positions)
        self.add_positions(row, "target", target_positions)
        self.add_positions(row, "cmd_delta", cmd_delta)
        self.add_positions(row, "target_delta", target_delta)

        row["action_delta_theta"] = self.format_float(target_delta[0])
        row["action_delta_radius"] = self.format_float(
            self.delta_float(servo_radius, self.prev_servo_radius)
        )
        row["action_delta_z"] = self.format_float(
            self.delta_float(servo_z, self.prev_servo_z)
        )
        row["action_delta_joint2"] = self.format_float(target_delta[1])
        row["action_delta_joint3"] = self.format_float(target_delta[2])
        row["action_delta_gripper"] = self.format_float(target_delta[5])

        self.csv_writer.writerow(row)
        self.rows_written += 1
        self.prev_target_positions = target_positions
        self.prev_servo_radius = servo_radius
        self.prev_servo_z = servo_z

    def positions_for(self, joint_dict):
        values = []
        for index in range(1, 7):
            values.append(joint_dict.get(str(index), math.nan))
        return values

    def delta_positions(self, left, right):
        if right is None:
            return [math.nan] * 6

        values = []
        for left_value, right_value in zip(left, right):
            if math.isnan(left_value) or math.isnan(right_value):
                values.append(math.nan)
            else:
                values.append(left_value - right_value)
        return values

    def delta_float(self, left, right):
        if math.isnan(left) or math.isnan(right):
            return math.nan

        return left - right

    def add_positions(self, row, prefix, values):
        for index, value in enumerate(values, start=1):
            row[f"{prefix}_j{index}"] = self.format_float(value)

    def parse_debug_value(self, name):
        match = re.search(rf"{re.escape(name)}=({FLOAT_RE})", self.cup_debug)
        if not match:
            return math.nan
        return float(match.group(1))

    def format_float(self, value):
        if value is None or math.isnan(float(value)):
            return ""
        return f"{float(value):.8f}"

    def destroy_node(self):
        if self.active:
            self.stop_episode(label="manual_abort", reason="node_shutdown")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeLogger()

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

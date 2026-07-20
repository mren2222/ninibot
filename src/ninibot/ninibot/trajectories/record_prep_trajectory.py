# flake8: noqa
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from arm_controller.kinematics.cylindrical import CylindricalArmKinematics
from ninibot.task_params import CYLINDRICAL_VISION_PARAMS
from ninibot.trajectories.prep_trajectory import write_latest_copy
from ninibot.trajectories.prep_trajectory import write_metadata_json
from ninibot.trajectories.prep_trajectory import write_raw_csv


GRIPPER_MIN_RAD = -0.174533
GRIPPER_MAX_RAD = 1.74533


def gripper_percent_to_rad(percent):
    percent = max(0.0, min(100.0, float(percent)))
    return GRIPPER_MIN_RAD + (percent / 100.0) * (
        GRIPPER_MAX_RAD - GRIPPER_MIN_RAD
    )


class RecordPrepTrajectory(Node):
    def __init__(self):
        super().__init__("record_prep_trajectory")

        self.declare_parameter("command_topic", "/fine_align_prep_teach/command")
        self.declare_parameter("status_topic", "/fine_align_prep_teach/status")
        self.declare_parameter("target_topic", "/target_joint_states")
        self.declare_parameter("real_joint_states_topic", "/real_joint_states")
        self.declare_parameter("torque_command_topic", "/motor_torque_command")
        self.declare_parameter("output_dir", "data/trajectories/fine_align_prep/sessions/current_recording")
        self.declare_parameter("sample_rate", 100.0)
        self.declare_parameter("target_publish_rate", 20.0)
        self.declare_parameter("auto_start", True)
        self.declare_parameter("disable_torque_on_ready", True)
        self.declare_parameter("enable_torque_on_finish", False)
        self.declare_parameter("record_start_delay_sec", 0.5)
        self.declare_parameter("min_record_duration_sec", 0.5)
        self.declare_parameter("joint_goal_tolerance_deg", 3.0)
        self.declare_parameter("search_base_source", "current")
        self.declare_parameter("search_base_deg", 0.0)
        self.declare_parameter("wrist_roll_deg", CYLINDRICAL_VISION_PARAMS["wrist_roll_deg"])
        self.declare_parameter("gripper_percent", CYLINDRICAL_VISION_PARAMS["gripper_percent"])

        self.command_topic = self.get_parameter("command_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.target_topic = self.get_parameter("target_topic").value
        self.real_joint_states_topic = self.get_parameter("real_joint_states_topic").value
        self.torque_command_topic = self.get_parameter("torque_command_topic").value
        self.output_dir = Path(self.get_parameter("output_dir").value)
        self.sample_rate = float(self.get_parameter("sample_rate").value)
        self.target_publish_rate = float(self.get_parameter("target_publish_rate").value)
        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.disable_torque_on_ready = bool(self.get_parameter("disable_torque_on_ready").value)
        self.enable_torque_on_finish = bool(self.get_parameter("enable_torque_on_finish").value)
        self.record_start_delay_sec = float(self.get_parameter("record_start_delay_sec").value)
        self.min_record_duration_sec = float(self.get_parameter("min_record_duration_sec").value)
        self.joint_goal_tolerance = math.radians(
            float(self.get_parameter("joint_goal_tolerance_deg").value)
        )
        self.search_base_source = str(self.get_parameter("search_base_source").value).strip().lower()
        self.search_base_deg = float(self.get_parameter("search_base_deg").value)
        self.wrist_roll_deg = float(self.get_parameter("wrist_roll_deg").value)
        self.gripper_percent = float(self.get_parameter("gripper_percent").value)
        self.gripper_rad = gripper_percent_to_rad(self.gripper_percent)

        self.params = dict(CYLINDRICAL_VISION_PARAMS)
        self.search_joint2_deg = float(self.params["search_joint2_deg"])
        self.search_joint3_deg = float(self.params["search_joint3_deg"])
        self.search_joint4_deg = float(self.params["search_joint4_deg"])
        self.kinematics = CylindricalArmKinematics(
            l1=self.params["l1"],
            l2=self.params["l2"],
            horizontal_tool_offset=self.params["horizontal_tool_offset"],
            z_tool_offset=self.params["z_tool_offset"],
            theta_joint_sign=self.params["theta_joint_sign"],
            theta_joint_offset_deg=self.params["theta_joint_offset_deg"],
            base_min_deg=self.params["base_min_deg"],
            base_max_deg=self.params["base_max_deg"],
            shoulder_min_deg=self.params["shoulder_min_deg"],
            shoulder_max_deg=self.params["shoulder_max_deg"],
            elbow_min_deg=self.params["elbow_min_deg"],
            elbow_max_deg=self.params["elbow_max_deg"],
            wrist_flex_min_deg=self.params["wrist_flex_min_deg"],
            wrist_flex_max_deg=self.params["wrist_flex_max_deg"],
            horizontal_wrist_pitch_offset_deg=self.params["horizontal_wrist_pitch_offset_deg"],
        )

        self.latest_positions = None
        self.latest_stamp_sec = None
        self.state = "IDLE"
        self.state_started_at = time.monotonic()
        self.record_started_at = None
        self.record_rows = []
        self.saved_raw_csv = ""
        self.saved_metadata_json = ""
        self.last_disable_command_at = 0.0

        self.target_pub = self.create_publisher(JointState, self.target_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.torque_pub = self.create_publisher(String, self.torque_command_topic, 10)
        self.command_sub = self.create_subscription(
            String,
            self.command_topic,
            self.command_callback,
            10,
        )
        self.real_sub = self.create_subscription(
            JointState,
            self.real_joint_states_topic,
            self.real_joint_callback,
            10,
        )

        self.target_timer = self.create_timer(
            1.0 / max(1e-6, self.target_publish_rate),
            self.target_tick,
        )
        self.record_timer = self.create_timer(
            1.0 / max(1e-6, self.sample_rate),
            self.record_tick,
        )
        self.status_timer = self.create_timer(0.25, self.publish_status)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.auto_start:
            self.start_teach_sequence()
        else:
            self.publish_status()

        self.get_logger().info(
            "record_prep_trajectory ready: "
            f"command_topic={self.command_topic}, output_dir={self.output_dir}"
        )

    def set_state(self, state):
        if self.state == state:
            return

        self.state = state
        self.state_started_at = time.monotonic()
        self.publish_status()
        self.get_logger().info(f"teach state -> {state}")

    def command_callback(self, msg):
        command = msg.data.strip().lower()

        if command in ("start", "go_search"):
            self.start_teach_sequence()
        elif command in ("record", "start_record"):
            self.start_recording()
        elif command in ("finish", "stop", "save"):
            self.finish_recording()
        elif command == "cancel":
            self.cancel()
        elif command == "disable":
            self.publish_torque_command("disable")
        elif command == "enable":
            self.publish_torque_command("enable")
        else:
            self.get_logger().warn(f"Unknown teach command: {msg.data}")

    def start_teach_sequence(self):
        self.record_rows = []
        self.record_started_at = None
        self.saved_raw_csv = ""
        self.saved_metadata_json = ""
        self.set_state("MOVING_TO_SEARCH")

    def cancel(self):
        self.record_rows = []
        self.record_started_at = None
        self.set_state("CANCELED")

    def real_joint_callback(self, msg):
        name_to_position = {
            name: position
            for name, position in zip(msg.name, msg.position)
        }

        if not all(str(index) in name_to_position for index in range(1, 7)):
            return

        self.latest_positions = [
            name_to_position[str(index)]
            for index in range(1, 7)
        ]
        self.latest_stamp_sec = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1e-9
        )

    def current_base_deg(self):
        if self.search_base_source == "fixed":
            return self.search_base_deg

        if self.latest_positions is not None:
            return math.degrees(self.latest_positions[0])

        return self.search_base_deg

    def search_joint4_rad(self):
        horizontal_j4_deg = self.kinematics.wrist_flex_deg(
            self.search_joint2_deg,
            self.search_joint3_deg,
        )
        return math.radians(horizontal_j4_deg + self.search_joint4_deg)

    def search_positions(self):
        return [
            math.radians(self.current_base_deg()),
            math.radians(self.search_joint2_deg),
            math.radians(self.search_joint3_deg),
            self.search_joint4_rad(),
            math.radians(self.wrist_roll_deg),
            self.gripper_rad,
        ]

    def target_tick(self):
        if self.state != "MOVING_TO_SEARCH":
            return

        target = self.search_positions()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["1", "2", "3", "4", "5", "6"]
        msg.position = target
        self.target_pub.publish(msg)

        if self.search_ready(target):
            if self.disable_torque_on_ready:
                self.publish_torque_command("disable")
                self.set_state("WAITING_TO_RECORD")
            else:
                self.set_state("READY_TO_TEACH")

    def search_ready(self, target):
        if self.latest_positions is None:
            return False

        for index in (1, 2, 3):
            if abs(self.latest_positions[index] - target[index]) > self.joint_goal_tolerance:
                return False

        return True

    def publish_torque_command(self, command):
        msg = String()
        msg.data = command
        self.torque_pub.publish(msg)
        self.get_logger().info(f"published torque command: {command}")

    def start_recording(self):
        if self.latest_positions is None:
            self.get_logger().warn("Cannot start recording before first /real_joint_states")
            return

        self.record_rows = []
        self.record_started_at = time.monotonic()
        self.set_state("RECORDING")

    def record_tick(self):
        if self.state == "WAITING_TO_RECORD":
            if time.monotonic() - self.last_disable_command_at >= 0.25:
                self.publish_torque_command("disable")
                self.last_disable_command_at = time.monotonic()

            if time.monotonic() - self.state_started_at >= self.record_start_delay_sec:
                self.start_recording()
            return

        if self.state != "RECORDING" or self.latest_positions is None:
            return

        now = time.monotonic()
        t_sec = now - self.record_started_at
        j2 = self.latest_positions[1]
        j3 = self.latest_positions[2]
        j4 = self.latest_positions[3]
        row = {
            "t_sec": f"{t_sec:.6f}",
            "stamp_sec": f"{0.0 if self.latest_stamp_sec is None else self.latest_stamp_sec:.9f}",
            "joint2_rad": f"{j2:.9f}",
            "joint3_rad": f"{j3:.9f}",
            "joint4_rad": f"{j4:.9f}",
            "joint2_deg": f"{math.degrees(j2):.6f}",
            "joint3_deg": f"{math.degrees(j3):.6f}",
            "joint4_deg": f"{math.degrees(j4):.6f}",
        }
        self.record_rows.append(row)

    def finish_recording(self):
        if self.state != "RECORDING":
            self.get_logger().warn(f"Cannot finish recording while state={self.state}")
            return

        duration = time.monotonic() - self.record_started_at
        if duration < self.min_record_duration_sec or len(self.record_rows) < 2:
            self.get_logger().warn(
                f"Recording too short: duration={duration:.2f}s, samples={len(self.record_rows)}"
            )
            self.set_state("FAILED")
            return

        stamp = time.strftime("%Y%m%d_%H%M%S")
        raw_csv = self.output_dir / f"fine_align_prep_raw_{stamp}.csv"
        metadata_json = self.output_dir / f"fine_align_prep_raw_{stamp}.json"
        write_raw_csv(raw_csv, self.record_rows)

        metadata = {
            "schema": "ninibot.fine_align_prep_raw.v1",
            "raw_csv": str(raw_csv),
            "sample_rate_hz": self.sample_rate,
            "duration_sec": duration,
            "samples": len(self.record_rows),
            "joint_names": ["2", "3", "4"],
            "search_joint2_deg": self.search_joint2_deg,
            "search_joint3_deg": self.search_joint3_deg,
            "search_joint4_deg": self.search_joint4_deg,
            "search_base_source": self.search_base_source,
            "search_base_deg": self.search_base_deg,
            "disable_torque_on_ready": self.disable_torque_on_ready,
        }
        write_metadata_json(metadata_json, metadata)
        write_latest_copy(raw_csv, self.output_dir / "latest_raw.csv")
        write_latest_copy(metadata_json, self.output_dir / "latest_raw.json")

        self.saved_raw_csv = str(raw_csv)
        self.saved_metadata_json = str(metadata_json)

        if self.enable_torque_on_finish:
            self.publish_torque_command("enable")

        self.set_state("SAVED")
        self.get_logger().info(f"Saved raw fine-align-prep recording: {raw_csv}")

    def publish_status(self):
        msg = String()
        msg.data = (
            f"state={self.state}, "
            f"samples={len(self.record_rows)}, "
            f"raw_csv={self.saved_raw_csv}"
        )
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RecordPrepTrajectory()

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

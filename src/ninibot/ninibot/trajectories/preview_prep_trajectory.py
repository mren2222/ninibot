# flake8: noqa
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

from arm_controller.kinematics.cylindrical import CylindricalArmKinematics
from arm_controller.motor.motor_driver import JointLimiter
from ninibot.task_params import CYLINDRICAL_VISION_PARAMS
from ninibot.trajectories.prep_trajectory import finite_difference
from ninibot.trajectories.prep_trajectory import load_processed_template
from ninibot.trajectories.prep_trajectory import retarget_trajectory


GRIPPER_MIN_RAD = -0.174533
GRIPPER_MAX_RAD = 1.74533


def gripper_percent_to_rad(percent):
    percent = max(0.0, min(100.0, float(percent)))
    return GRIPPER_MIN_RAD + (percent / 100.0) * (
        GRIPPER_MAX_RAD - GRIPPER_MIN_RAD
    )


class PreviewPrepTrajectory(Node):
    def __init__(self):
        super().__init__("preview_prep_trajectory")

        self.declare_parameter(
            "trajectory_path",
            "data/trajectories/fine_align_prep/sessions/2026-06-15_8cm_base_raw/processed_template.json",
        )
        self.declare_parameter(
            "builtin_reference_path",
            "data/trajectories/fine_align_prep/references/rule_based_fine_align_prep_reference.json",
        )
        self.declare_parameter("save_builtin_reference", True)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("sample_rate", 100.0)
        self.declare_parameter("duration_sec", 0.0)
        self.declare_parameter("shape_scale", 1.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("frame_id", "base")
        self.declare_parameter("base_deg", 0.0)
        self.declare_parameter("cup_control_rate", 100.0)
        self.declare_parameter("motor_control_rate", 500.0)
        self.declare_parameter("builtin_wrist_step_deg", 0.60)
        self.declare_parameter("builtin_speed_scale", 2.0)
        self.declare_parameter("builtin_accel_scale", 2.0)
        self.declare_parameter("builtin_j2_max_vel_deg", 30.0)
        self.declare_parameter("builtin_j3_max_vel_deg", 30.0)
        self.declare_parameter("builtin_j4_max_vel_deg", 90.0)
        self.declare_parameter("builtin_j2_max_acc_deg", 20.0)
        self.declare_parameter("builtin_j3_max_acc_deg", 20.0)
        self.declare_parameter("builtin_j4_max_acc_deg", 90.0)
        self.declare_parameter("builtin_joint_states_topic", "/fine_align_prep_builtin_joint_states")
        self.declare_parameter("processed_joint_states_topic", "/fine_align_prep_processed_joint_states")
        self.declare_parameter("builtin_path_topic", "/fine_align_prep_builtin_path")
        self.declare_parameter("processed_path_topic", "/fine_align_prep_processed_path")

        self.trajectory_path = self.get_parameter("trajectory_path").value
        self.builtin_reference_path = self.get_parameter("builtin_reference_path").value
        self.save_builtin_reference_enabled = bool(self.get_parameter("save_builtin_reference").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.sample_rate = float(self.get_parameter("sample_rate").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.shape_scale = float(self.get_parameter("shape_scale").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.base_deg = float(self.get_parameter("base_deg").value)
        self.cup_control_rate = float(self.get_parameter("cup_control_rate").value)
        self.motor_control_rate = float(self.get_parameter("motor_control_rate").value)
        self.builtin_wrist_step_deg = float(self.get_parameter("builtin_wrist_step_deg").value)
        self.builtin_speed_scale = float(self.get_parameter("builtin_speed_scale").value)
        self.builtin_accel_scale = float(self.get_parameter("builtin_accel_scale").value)

        self.params = dict(CYLINDRICAL_VISION_PARAMS)
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

        self.template = load_processed_template(self.trajectory_path)
        self.start_j234 = np.asarray(self.search_j234(), dtype=float)
        self.end_j234 = np.asarray(self.fine_align_prep_j234(), dtype=float)
        self.duration = self.duration_sec if self.duration_sec > 0.0 else self.template["duration_sec"]
        self.processed_times, self.processed_positions = retarget_trajectory(
            self.template,
            self.start_j234,
            self.end_j234,
            duration_sec=self.duration,
            sample_rate_hz=self.sample_rate,
            shape_scale=self.shape_scale,
        )
        self.builtin_times, self.builtin_positions = self.make_builtin_trajectory()

        if self.save_builtin_reference_enabled:
            self.save_builtin_reference()

        self.builtin_pub = self.create_publisher(
            JointState,
            self.get_parameter("builtin_joint_states_topic").value,
            10,
        )
        self.processed_pub = self.create_publisher(
            JointState,
            self.get_parameter("processed_joint_states_topic").value,
            10,
        )
        self.builtin_marker_pub = self.create_publisher(
            Marker,
            self.get_parameter("builtin_path_topic").value,
            10,
        )
        self.processed_marker_pub = self.create_publisher(
            Marker,
            self.get_parameter("processed_path_topic").value,
            10,
        )

        self.start_time = time.monotonic()
        self.timer = self.create_timer(1.0 / max(1e-6, self.publish_rate), self.tick)
        self.publish_paths()

        self.get_logger().info(
            f"preview_prep_trajectory loaded processed={self.trajectory_path}, "
            f"builtin_reference={self.builtin_reference_path}, duration={self.duration:.2f}s"
        )

    def search_j234(self):
        joint2 = float(self.params["search_joint2_deg"])
        joint3 = float(self.params["search_joint3_deg"])
        joint4 = (
            self.kinematics.wrist_flex_deg(joint2, joint3)
            + float(self.params["search_joint4_deg"])
        )
        return [math.radians(joint2), math.radians(joint3), math.radians(joint4)]

    def fine_align_prep_j234(self):
        values = self.template.get("processed_end_rad")

        if not isinstance(values, list) or len(values) < 2:
            raise RuntimeError("Processed trajectory must contain processed_end_rad")

        joint2 = math.degrees(float(values[0]))
        joint3 = math.degrees(float(values[1]))

        joint4 = self.kinematics.wrist_flex_deg(joint2, joint3)
        return [math.radians(joint2), math.radians(joint3), math.radians(joint4)]

    def make_builtin_trajectory(self):
        count = len(self.processed_times)
        output_times = np.linspace(0.0, self.duration, count)
        motor_dt = 1.0 / max(self.motor_control_rate, 1e-9)
        cup_dt = 1.0 / max(self.cup_control_rate, 1e-9)
        wrist_step = math.radians(abs(self.builtin_wrist_step_deg))
        wrist_offset = math.radians(float(self.params["search_joint4_deg"]))

        max_vel_deg = [
            float(self.get_parameter("builtin_j2_max_vel_deg").value),
            float(self.get_parameter("builtin_j3_max_vel_deg").value),
            float(self.get_parameter("builtin_j4_max_vel_deg").value),
        ]
        max_acc_deg = [
            float(self.get_parameter("builtin_j2_max_acc_deg").value),
            float(self.get_parameter("builtin_j3_max_acc_deg").value),
            float(self.get_parameter("builtin_j4_max_acc_deg").value),
        ]
        limiters = [
            JointLimiter(
                max_vel=math.radians(max_vel * self.builtin_speed_scale),
                max_acc=math.radians(max_acc * self.builtin_accel_scale),
            )
            for max_vel, max_acc in zip(max_vel_deg, max_acc_deg)
        ]

        for limiter, start in zip(limiters, self.start_j234):
            limiter.reset(float(start))

        positions = [np.asarray([limiter.pos for limiter in limiters], dtype=float)]
        output_index = 1
        current_time = 0.0
        next_cup_update = 0.0

        while output_index < count:
            if current_time + 1e-12 >= next_cup_update:
                if wrist_step <= 0.0 or abs(wrist_offset) <= wrist_step:
                    wrist_offset = 0.0
                else:
                    wrist_offset -= math.copysign(wrist_step, wrist_offset)

                current_joint2_deg = math.degrees(float(limiters[0].pos))
                current_joint3_deg = math.degrees(float(limiters[1].pos))
                current_horizontal_j4 = math.radians(
                    self.kinematics.wrist_flex_deg(
                        current_joint2_deg,
                        current_joint3_deg,
                    )
                )
                goals = [
                    float(self.end_j234[0]),
                    float(self.end_j234[1]),
                    current_horizontal_j4 + wrist_offset,
                ]

                for limiter, goal in zip(limiters, goals):
                    limiter.set_goal(goal)

                next_cup_update += cup_dt

            current_time += motor_dt

            for limiter in limiters:
                limiter.step(motor_dt)

            while output_index < count and current_time + 1e-12 >= output_times[output_index]:
                positions.append(np.asarray([limiter.pos for limiter in limiters], dtype=float))
                output_index += 1

        return output_times, np.stack(positions, axis=0)

    def trajectory_samples(self, times, positions):
        times = np.asarray(times, dtype=float)
        positions = np.asarray(positions, dtype=float)
        velocity = finite_difference(positions, times)
        acceleration = finite_difference(velocity, times)
        jerk = finite_difference(acceleration, times)
        samples = []

        for index, t_sec in enumerate(times):
            row = {
                "t_sec": float(t_sec),
                "s": float(t_sec / max(float(times[-1]), 1e-9)),
            }

            for joint_index, joint_name in enumerate(("2", "3", "4")):
                value = float(positions[index, joint_index])
                row[f"joint{joint_name}_rad"] = value
                row[f"joint{joint_name}_deg"] = math.degrees(value)
                row[f"joint{joint_name}_vel_rad_s"] = float(velocity[index, joint_index])
                row[f"joint{joint_name}_acc_rad_s2"] = float(acceleration[index, joint_index])
                row[f"joint{joint_name}_jerk_rad_s3"] = float(jerk[index, joint_index])

            samples.append(row)

        return samples

    def save_builtin_reference(self):
        try:
            path = Path(str(self.builtin_reference_path)).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            velocity = finite_difference(self.builtin_positions, self.builtin_times)
            acceleration = finite_difference(velocity, self.builtin_times)
            jerk = finite_difference(acceleration, self.builtin_times)
            data = {
                "schema": "ninibot.fine_align_prep_builtin_reference.v1",
                "description": "Rule-based fine-align-prep reference trajectory kept for RViz comparison and project materials.",
                "source_processed_trajectory": str(self.trajectory_path),
                "joint_names": ["2", "3", "4"],
                "duration_sec": float(self.duration),
                "sample_rate_hz": float(self.sample_rate),
                "start_rad": self.start_j234.tolist(),
                "end_rad": self.end_j234.tolist(),
                "builtin_wrist_step_deg": float(self.builtin_wrist_step_deg),
                "builtin_speed_scale": float(self.builtin_speed_scale),
                "builtin_accel_scale": float(self.builtin_accel_scale),
                "cup_control_rate": float(self.cup_control_rate),
                "motor_control_rate": float(self.motor_control_rate),
                "max_abs_velocity_rad_s": np.max(np.abs(velocity), axis=0).tolist(),
                "max_abs_acceleration_rad_s2": np.max(np.abs(acceleration), axis=0).tolist(),
                "max_abs_jerk_rad_s3": np.max(np.abs(jerk), axis=0).tolist(),
                "samples": self.trajectory_samples(self.builtin_times, self.builtin_positions),
            }

            with path.open("w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2)
                stream.write("\n")

            self.get_logger().info(f"saved builtin reference trajectory to {path}")
        except Exception as exc:
            self.get_logger().warning(f"failed to save builtin reference trajectory: {exc}")

    def full_joint_state(self, j234):
        return [
            math.radians(self.base_deg),
            float(j234[0]),
            float(j234[1]),
            float(j234[2]),
            math.radians(float(self.params["wrist_roll_deg"])),
            gripper_percent_to_rad(float(self.params["gripper_percent"])),
        ]

    def interpolate_positions(self, positions, elapsed):
        if elapsed >= self.duration:
            return positions[-1]

        s = elapsed / max(self.duration, 1e-9)
        index_float = s * (len(positions) - 1)
        lower = int(math.floor(index_float))
        upper = min(len(positions) - 1, lower + 1)
        frac = index_float - lower
        return positions[lower] + frac * (positions[upper] - positions[lower])

    def tick(self):
        elapsed = time.monotonic() - self.start_time

        if self.loop:
            elapsed = elapsed % max(self.duration, 1e-9)
        else:
            elapsed = min(elapsed, self.duration)

        builtin = self.interpolate_positions(self.builtin_positions, elapsed)
        processed = self.interpolate_positions(self.processed_positions, elapsed)
        self.publish_joint_state(self.builtin_pub, builtin)
        self.publish_joint_state(self.processed_pub, processed)
        self.publish_paths()

    def publish_joint_state(self, pub, j234):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["1", "2", "3", "4", "5", "6"]
        msg.position = self.full_joint_state(j234)
        pub.publish(msg)

    def point_from_j234(self, j234):
        pose = self.kinematics.cylindrical_from_joint_degs(
            self.base_deg,
            math.degrees(float(j234[0])),
            math.degrees(float(j234[1])),
        )
        theta = math.radians(pose.theta_deg)
        point = Point()
        point.x = pose.radius * math.cos(theta)
        point.y = pose.radius * math.sin(theta)
        point.z = pose.z
        return point

    def marker(self, marker_id, positions, rgba, frame_id):
        msg = Marker()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = "fine_align_prep"
        msg.id = marker_id
        msg.type = Marker.LINE_STRIP
        msg.action = Marker.ADD
        msg.scale.x = 0.008
        msg.color.r = rgba[0]
        msg.color.g = rgba[1]
        msg.color.b = rgba[2]
        msg.color.a = rgba[3]
        msg.pose.orientation.w = 1.0
        msg.points = [self.point_from_j234(row) for row in positions]
        return msg

    def publish_paths(self):
        self.builtin_marker_pub.publish(
            self.marker(1, self.builtin_positions, (0.1, 0.35, 1.0, 1.0), "builtin_/base")
        )
        self.processed_marker_pub.publish(
            self.marker(2, self.processed_positions, (0.0, 0.9, 0.35, 1.0), "processed_/base")
        )


def main(args=None):
    rclpy.init(args=args)
    node = PreviewPrepTrajectory()

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

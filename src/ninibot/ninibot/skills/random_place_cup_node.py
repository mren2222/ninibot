# flake8: noqa
import math
import random
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from arm_controller.kinematics.cylindrical import CylindricalPose
from arm_controller.kinematics.cylindrical import CylindricalArmKinematics
from ninibot.task_params import RANDOM_PLACE_CUP_PARAMS


RETRACTING_RADIUS = "RETRACTING_RADIUS"
EXTENDING_RADIUS = "EXTENDING_RADIUS"


def safe_pre_pan_radius(current_radius, pre_pan_radius):
    if pre_pan_radius <= 0.0:
        return current_radius

    return min(current_radius, pre_pan_radius)


def retract_radius_pose(start_pose, pre_pan_radius):
    return CylindricalPose(
        radius=safe_pre_pan_radius(start_pose.radius, pre_pan_radius),
        theta_deg=start_pose.theta_deg,
        z=start_pose.z,
    )


def pan_at_current_radius_pose(start_pose, target_theta_deg, z=None):
    return CylindricalPose(
        radius=start_pose.radius,
        theta_deg=target_theta_deg,
        z=start_pose.z if z is None else z,
    )


def radius_at_current_theta_pose(start_pose, target_radius, z=None):
    return CylindricalPose(
        radius=target_radius,
        theta_deg=start_pose.theta_deg,
        z=start_pose.z if z is None else z,
    )


def lower_at_current_radius_pose(start_pose, target_z):
    return CylindricalPose(
        radius=start_pose.radius,
        theta_deg=start_pose.theta_deg,
        z=target_z,
    )


PATH_PHASES = {
    "LIFTING",
    RETRACTING_RADIUS,
    "MOVING_TO_RANDOM_PLACE",
    EXTENDING_RADIUS,
    "LOWERING_TO_RELEASE",
    "LIFTING_TO_HOME",
    "RETRACTING_HOME_RADIUS",
    "PANNING_HOME",
}

ACTIVE_PHASES = {
    "GRIPPING",
    "LIFTING",
    RETRACTING_RADIUS,
    "MOVING_TO_RANDOM_PLACE",
    EXTENDING_RADIUS,
    "LOWERING_TO_RELEASE",
    "RELEASING",
    "LIFTING_TO_HOME",
    "RETRACTING_HOME_RADIUS",
    "PANNING_HOME",
    "RETURNING_HOME",
}


class RandomPlaceCupNode(Node):
    def __init__(self):
        super().__init__("random_place_cup_node")

        self.declare_parameters(
            namespace="",
            parameters=list(RANDOM_PLACE_CUP_PARAMS.items()),
        )

        self.command_topic = self.get_parameter("command_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.debug_topic = self.get_parameter("debug_topic").value
        self.target_topic = self.get_parameter("target_topic").value
        self.real_joint_states_topic = self.get_parameter("real_joint_states_topic").value
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.joint_goal_tolerance = math.radians(
            float(self.get_parameter("joint_goal_tolerance_deg").value)
        )
        self.settle_cycles = int(self.get_parameter("settle_cycles").value)
        self.grip_settle_sec = float(self.get_parameter("grip_settle_sec").value)
        self.release_settle_sec = float(
            self.get_parameter("release_settle_sec").value
        )
        self.target_check_joint_count = max(
            1,
            min(4, int(self.get_parameter("target_check_joint_count").value)),
        )
        self.allow_timeout_advance = bool(
            self.get_parameter("allow_timeout_advance").value
        )
        self.phase_timeout_sec = float(
            self.get_parameter("phase_timeout_sec").value
        )
        self.timeout_advance_tolerance_deg = float(
            self.get_parameter("timeout_advance_tolerance_deg").value
        )

        self.table_forward_min = float(self.get_parameter("table_forward_min").value)
        self.table_forward_max = float(self.get_parameter("table_forward_max").value)
        self.table_lateral_min = float(self.get_parameter("table_lateral_min").value)
        self.table_lateral_max = float(self.get_parameter("table_lateral_max").value)
        self.radius_tip_min = float(self.get_parameter("radius_tip_min").value)
        self.radius_tip_max = float(self.get_parameter("radius_tip_max").value)
        self.theta_min_deg = float(self.get_parameter("theta_min_deg").value)
        self.theta_max_deg = float(self.get_parameter("theta_max_deg").value)
        self.place_z = float(self.get_parameter("place_z").value)
        self.max_sample_attempts = max(
            1,
            int(self.get_parameter("max_sample_attempts").value),
        )
        self.random_seed = int(self.get_parameter("random_seed").value)

        self.lift_z_offset = float(self.get_parameter("lift_z_offset").value)
        self.lift_duration_sec = float(
            self.get_parameter("lift_duration_sec").value
        )
        self.pre_pan_radius = float(self.get_parameter("pre_pan_radius").value)
        self.pre_pan_retract_duration_sec = float(
            self.get_parameter("pre_pan_retract_duration_sec").value
        )
        self.post_pan_extend_duration_sec = float(
            self.get_parameter("post_pan_extend_duration_sec").value
        )
        self.horizontal_move_duration_sec = float(
            self.get_parameter("horizontal_move_duration_sec").value
        )
        self.lower_duration_sec = float(
            self.get_parameter("lower_duration_sec").value
        )
        self.use_dynamic_carry_z = bool(
            self.get_parameter("use_dynamic_carry_z").value
        )
        self.carry_z = float(self.get_parameter("carry_z").value)
        self.return_home_z = float(self.get_parameter("return_home_z").value)
        self.return_home_lift_duration_sec = float(
            self.get_parameter("return_home_lift_duration_sec").value
        )
        self.return_home_speed_scale = max(
            1e-6,
            float(self.get_parameter("return_home_speed_scale").value),
        )
        self.return_pan_deg = float(self.get_parameter("return_pan_deg").value)
        self.return_joint2_deg = float(self.get_parameter("return_joint2_deg").value)
        self.return_joint3_deg = float(self.get_parameter("return_joint3_deg").value)

        self.l1 = float(self.get_parameter("l1").value)
        self.l2 = float(self.get_parameter("l2").value)
        self.horizontal_tool_offset = float(
            self.get_parameter("horizontal_tool_offset").value
        )
        self.z_tool_offset = float(self.get_parameter("z_tool_offset").value)
        self.theta_joint_sign = float(self.get_parameter("theta_joint_sign").value)
        self.theta_joint_offset_deg = float(
            self.get_parameter("theta_joint_offset_deg").value
        )
        self.base_min_deg = float(self.get_parameter("base_min_deg").value)
        self.base_max_deg = float(self.get_parameter("base_max_deg").value)
        self.shoulder_min_deg = float(self.get_parameter("shoulder_min_deg").value)
        self.shoulder_max_deg = float(self.get_parameter("shoulder_max_deg").value)
        self.elbow_min_deg = float(self.get_parameter("elbow_min_deg").value)
        self.elbow_max_deg = float(self.get_parameter("elbow_max_deg").value)
        self.wrist_flex_min_deg = float(
            self.get_parameter("wrist_flex_min_deg").value
        )
        self.wrist_flex_max_deg = float(
            self.get_parameter("wrist_flex_max_deg").value
        )
        self.horizontal_wrist_pitch_offset_deg = float(
            self.get_parameter("horizontal_wrist_pitch_offset_deg").value
        )
        self.wrist_roll_deg = float(self.get_parameter("wrist_roll_deg").value)
        self.hold_gripper_percent = float(
            self.get_parameter("hold_gripper_percent").value
        )
        self.release_gripper_percent = float(
            self.get_parameter("release_gripper_percent").value
        )

        if self.table_forward_min > self.table_forward_max:
            self.table_forward_min, self.table_forward_max = (
                self.table_forward_max,
                self.table_forward_min,
            )
        if self.table_lateral_min > self.table_lateral_max:
            self.table_lateral_min, self.table_lateral_max = (
                self.table_lateral_max,
                self.table_lateral_min,
            )
        if self.radius_tip_min > self.radius_tip_max:
            self.radius_tip_min, self.radius_tip_max = (
                self.radius_tip_max,
                self.radius_tip_min,
            )
        if self.theta_min_deg > self.theta_max_deg:
            self.theta_min_deg, self.theta_max_deg = (
                self.theta_max_deg,
                self.theta_min_deg,
            )

        self.kinematics = CylindricalArmKinematics(
            l1=self.l1,
            l2=self.l2,
            horizontal_tool_offset=self.horizontal_tool_offset,
            z_tool_offset=self.z_tool_offset,
            theta_joint_sign=self.theta_joint_sign,
            theta_joint_offset_deg=self.theta_joint_offset_deg,
            base_min_deg=self.base_min_deg,
            base_max_deg=self.base_max_deg,
            shoulder_min_deg=self.shoulder_min_deg,
            shoulder_max_deg=self.shoulder_max_deg,
            elbow_min_deg=self.elbow_min_deg,
            elbow_max_deg=self.elbow_max_deg,
            wrist_flex_min_deg=self.wrist_flex_min_deg,
            wrist_flex_max_deg=self.wrist_flex_max_deg,
            horizontal_wrist_pitch_offset_deg=self.horizontal_wrist_pitch_offset_deg,
        )

        self.rng = random.Random(None if self.random_seed == 0 else self.random_seed)

        self.target_pub = self.create_publisher(JointState, self.target_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)
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

        self.joint_names = ["1", "2", "3", "4", "5", "6"]
        self.latest_joint_positions = None
        self.current_target = None
        self.phase = "IDLE"
        self.done_reported = False
        self.settle_count = 0
        self.phase_started_at = None
        self.last_progress_log_time = 0.0
        self.path_start_pose = None
        self.path_goal_pose = None
        self.path_duration_sec = 0.0
        self.path_gripper_percent = self.hold_gripper_percent
        self.active_carry_z = None
        self.random_target_pose = None
        self.random_target_forward = 0.0
        self.random_target_lateral = 0.0

        self.timer = self.create_timer(1.0 / self.control_rate, self.loop)

        self.publish_status("IDLE")
        self.get_logger().info(
            "random_place_cup_node started; waiting for /random_place_cup_command=start"
        )
        self.get_logger().info(
            f"placement table forward=[{self.table_forward_min:.2f}, "
            f"{self.table_forward_max:.2f}] m, lateral=[{self.table_lateral_min:.2f}, "
            f"{self.table_lateral_max:.2f}] m, radius_tip=[{self.radius_tip_min:.2f}, "
            f"{self.radius_tip_max:.2f}] m, theta=[{self.theta_min_deg:.1f}, "
            f"{self.theta_max_deg:.1f}] deg"
        )

    def clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def in_range(self, value, lower, upper):
        return lower <= value <= upper

    def gripper_percent_to_rad(self, percent):
        percent = self.clamp(percent, 0.0, 100.0)
        gripper_min_rad = -0.174533
        gripper_max_rad = 1.74533
        return gripper_min_rad + (percent / 100.0) * (
            gripper_max_rad - gripper_min_rad
        )

    def wrist_flex_deg(self, joint2_deg, joint3_deg):
        return self.kinematics.wrist_flex_deg(joint2_deg, joint3_deg)

    def current_cylindrical_pose(self):
        if self.latest_joint_positions is None:
            return None

        shoulder_deg = math.degrees(self.latest_joint_positions[1])
        elbow_deg = math.degrees(self.latest_joint_positions[2])
        return self.kinematics.cylindrical_from_joint_degs(
            math.degrees(self.latest_joint_positions[0]),
            shoulder_deg,
            elbow_deg,
        )

    def make_joint_pose(
        self,
        pan_deg,
        joint2_deg,
        joint3_deg,
        joint4_deg,
        gripper_percent,
    ):
        if not self.in_range(pan_deg, self.base_min_deg, self.base_max_deg):
            self.get_logger().warn(f"Random place base out of range: {pan_deg:.1f} deg")
            return None

        if not self.in_range(joint2_deg, self.shoulder_min_deg, self.shoulder_max_deg):
            self.get_logger().warn(
                f"Random place shoulder out of range: {joint2_deg:.1f} deg"
            )
            return None

        if not self.in_range(joint3_deg, self.elbow_min_deg, self.elbow_max_deg):
            self.get_logger().warn(f"Random place elbow out of range: {joint3_deg:.1f} deg")
            return None

        if not self.in_range(
            joint4_deg,
            self.wrist_flex_min_deg,
            self.wrist_flex_max_deg,
        ):
            self.get_logger().warn(f"Random place wrist out of range: {joint4_deg:.1f} deg")
            return None

        return [
            math.radians(pan_deg),
            math.radians(joint2_deg),
            math.radians(joint3_deg),
            math.radians(joint4_deg),
            math.radians(self.wrist_roll_deg),
            self.gripper_percent_to_rad(gripper_percent),
        ]

    def make_pose(self, pan_deg, joint2_deg, joint3_deg, gripper_percent):
        return self.make_joint_pose(
            pan_deg,
            joint2_deg,
            joint3_deg,
            self.wrist_flex_deg(joint2_deg, joint3_deg),
            gripper_percent,
        )

    def make_cylindrical_pose(self, pose, gripper_percent):
        solution = self.kinematics.cylindrical_ik(
            pose.radius,
            pose.theta_deg,
            pose.z,
        )

        if solution is None:
            self.get_logger().warn(f"Random place IK failed: {self.kinematics.last_error}")
            return None

        return self.make_pose(
            solution.base_deg,
            solution.joint2_deg,
            solution.joint3_deg,
            gripper_percent,
        )

    def make_grip_pose_from_current(self):
        pose = self.current_cylindrical_pose()

        if pose is None:
            return None

        return self.make_cylindrical_pose(
            pose,
            self.hold_gripper_percent,
        )

    def make_return_pose(self):
        return self.make_pose(
            self.return_pan_deg,
            self.return_joint2_deg,
            self.return_joint3_deg,
            self.release_gripper_percent,
        )

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def publish_debug(self, text):
        msg = String()
        msg.data = text
        self.debug_pub.publish(msg)

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

    def command_callback(self, msg):
        command = msg.data.strip().lower()

        if command in ("start", "place", "random_place"):
            self.start_place()
        elif command in ("cancel", "stop"):
            self.cancel_place()
        elif command == "reset":
            self.reset_to_idle()
        else:
            self.get_logger().warn(f"Unknown random place command: {msg.data}")

    def sample_target_pose(self):
        current_pose = self.current_cylindrical_pose()
        carry_z = self.carry_z
        if current_pose is not None and self.use_dynamic_carry_z:
            carry_z = current_pose.z + self.lift_z_offset

        for _ in range(self.max_sample_attempts):
            forward = self.rng.uniform(
                self.table_forward_min,
                self.table_forward_max,
            )
            lateral = self.rng.uniform(
                self.table_lateral_min,
                self.table_lateral_max,
            )
            radius = math.hypot(forward, lateral)
            if radius <= 1e-6:
                continue
            if not self.in_range(radius, self.radius_tip_min, self.radius_tip_max):
                continue

            theta_deg = math.degrees(math.atan2(lateral, forward))
            if not self.in_range(theta_deg, self.theta_min_deg, self.theta_max_deg):
                continue

            place_pose = CylindricalPose(
                radius=radius,
                theta_deg=theta_deg,
                z=self.place_z,
            )
            carry_pose = CylindricalPose(
                radius=radius,
                theta_deg=theta_deg,
                z=carry_z,
            )

            if self.kinematics.cylindrical_ik(
                place_pose.radius,
                place_pose.theta_deg,
                place_pose.z,
            ) is None:
                continue

            if self.kinematics.cylindrical_ik(
                carry_pose.radius,
                carry_pose.theta_deg,
                carry_pose.z,
            ) is None:
                continue

            return place_pose, forward, lateral

        return None, 0.0, 0.0

    def start_place(self):
        if self.phase not in ("IDLE", "DONE", "FAILED", "CANCELED"):
            self.get_logger().warn(f"Random place already active: {self.phase}")
            return

        target_pose, forward, lateral = self.sample_target_pose()
        if target_pose is None:
            self.phase = "FAILED"
            self.current_target = None
            self.phase_started_at = time.monotonic()
            self.publish_status(self.phase)
            self.publish_debug(
                "sample_failed=true, "
                f"attempts={self.max_sample_attempts}, "
                f"last_error={self.kinematics.last_error}"
            )
            self.get_logger().warn(
                "random place failed: could not sample a reachable target in "
                f"{self.max_sample_attempts} attempts"
            )
            return

        self.random_target_pose = target_pose
        self.random_target_forward = forward
        self.random_target_lateral = lateral
        self.phase = "GRIPPING"
        self.done_reported = False
        self.settle_count = 0
        self.last_progress_log_time = 0.0
        self.active_carry_z = None
        self.clear_path()
        self.current_target = self.make_grip_pose_from_current()
        self.phase_started_at = (
            time.monotonic() if self.current_target is not None else None
        )

        self.publish_status(self.phase)
        self.publish_debug(
            f"target_forward={forward:.3f}, "
            f"target_lateral={lateral:.3f}, "
            f"target_radius={target_pose.radius:.3f}, "
            f"target_theta_deg={target_pose.theta_deg:.1f}, "
            f"target_z={target_pose.z:.3f}"
        )
        self.get_logger().info(
            "random place: GRIPPING; sampled "
            f"forward={forward:.3f} m, lateral={lateral:.3f} m, "
            f"radius={target_pose.radius:.3f} m, theta={target_pose.theta_deg:.1f} deg"
        )

    def cancel_place(self):
        if self.phase in ("IDLE", "DONE", "CANCELED"):
            return

        self.phase = "CANCELED"
        self.current_target = None
        self.clear_path()
        self.phase_started_at = time.monotonic()
        self.publish_status(self.phase)
        self.get_logger().info("random place: CANCELED")

    def reset_to_idle(self):
        self.phase = "IDLE"
        self.current_target = None
        self.done_reported = False
        self.settle_count = 0
        self.phase_started_at = None
        self.last_progress_log_time = 0.0
        self.active_carry_z = None
        self.random_target_pose = None
        self.clear_path()
        self.publish_status(self.phase)
        self.get_logger().info("random place: IDLE")

    def clear_path(self):
        self.path_start_pose = None
        self.path_goal_pose = None
        self.path_duration_sec = 0.0

    def pose_or_path_goal(self):
        if self.path_goal_pose is not None:
            return self.path_goal_pose

        pose = self.current_cylindrical_pose()

        if pose is not None:
            return pose

        return None

    def set_cylindrical_path_phase(
        self,
        phase,
        start_pose,
        goal_pose,
        duration_sec,
        gripper_percent,
    ):
        if start_pose is None or goal_pose is None:
            self.phase = "FAILED"
            self.current_target = None
            self.clear_path()
            self.publish_status(self.phase)
            self.get_logger().warn("random place failed: invalid cylindrical path")
            return

        self.phase = phase
        self.path_start_pose = start_pose
        self.path_goal_pose = goal_pose
        self.path_duration_sec = max(0.0, float(duration_sec))
        self.path_gripper_percent = gripper_percent
        self.settle_count = 0
        self.phase_started_at = time.monotonic()
        self.last_progress_log_time = 0.0
        self.current_target = self.make_cylindrical_pose(
            start_pose,
            gripper_percent,
        )

        if self.current_target is None:
            self.phase = "FAILED"
            self.clear_path()
            self.publish_status(self.phase)
            self.get_logger().warn("random place failed: invalid path start target")
            return

        self.publish_status(self.phase)
        self.get_logger().info(
            f"random place: {self.phase} path "
            f"start=(r={start_pose.radius:.3f}, theta={start_pose.theta_deg:.1f}, z={start_pose.z:.3f}) "
            f"goal=(r={goal_pose.radius:.3f}, theta={goal_pose.theta_deg:.1f}, z={goal_pose.z:.3f}) "
            f"duration={self.path_duration_sec:.2f}s"
        )

    def start_lifting_phase(self):
        start_pose = self.pose_or_path_goal()
        if start_pose is None:
            self.phase = "FAILED"
            self.publish_status(self.phase)
            self.get_logger().warn("random place failed: no current pose for lifting")
            return

        self.active_carry_z = start_pose.z + self.lift_z_offset
        goal_pose = CylindricalPose(
            radius=start_pose.radius,
            theta_deg=start_pose.theta_deg,
            z=self.active_carry_z,
        )
        self.set_cylindrical_path_phase(
            "LIFTING",
            start_pose,
            goal_pose,
            self.lift_duration_sec,
            self.hold_gripper_percent,
        )

    def start_retract_radius_phase(self):
        start_pose = self.pose_or_path_goal()
        if start_pose is None:
            self.phase = "FAILED"
            self.publish_status(self.phase)
            self.get_logger().warn("random place failed: no current pose for radius retract")
            return

        goal_pose = retract_radius_pose(start_pose, self.pre_pan_radius)
        self.set_cylindrical_path_phase(
            RETRACTING_RADIUS,
            start_pose,
            goal_pose,
            self.pre_pan_retract_duration_sec,
            self.hold_gripper_percent,
        )

    def start_horizontal_move_phase(self):
        start_pose = self.pose_or_path_goal()
        if start_pose is None or self.random_target_pose is None:
            self.phase = "FAILED"
            self.publish_status(self.phase)
            self.get_logger().warn("random place failed: missing move target")
            return

        if self.use_dynamic_carry_z:
            self.active_carry_z = start_pose.z
        elif self.active_carry_z is None:
            self.active_carry_z = self.carry_z

        goal_pose = pan_at_current_radius_pose(
            start_pose,
            self.random_target_pose.theta_deg,
            self.active_carry_z,
        )
        self.set_cylindrical_path_phase(
            "MOVING_TO_RANDOM_PLACE",
            start_pose,
            goal_pose,
            self.horizontal_move_duration_sec,
            self.hold_gripper_percent,
        )

    def start_extend_radius_phase(self):
        start_pose = self.pose_or_path_goal()
        if start_pose is None or self.random_target_pose is None:
            self.phase = "FAILED"
            self.publish_status(self.phase)
            self.get_logger().warn("random place failed: missing radius extend target")
            return

        goal_pose = radius_at_current_theta_pose(
            start_pose,
            self.random_target_pose.radius,
            self.active_carry_z,
        )
        self.set_cylindrical_path_phase(
            EXTENDING_RADIUS,
            start_pose,
            goal_pose,
            self.post_pan_extend_duration_sec,
            self.hold_gripper_percent,
        )

    def start_lowering_phase(self):
        start_pose = self.pose_or_path_goal()
        if start_pose is None or self.random_target_pose is None:
            self.phase = "FAILED"
            self.publish_status(self.phase)
            self.get_logger().warn("random place failed: missing lowering target")
            return

        goal_pose = lower_at_current_radius_pose(start_pose, self.random_target_pose.z)
        self.set_cylindrical_path_phase(
            "LOWERING_TO_RELEASE",
            start_pose,
            goal_pose,
            self.lower_duration_sec,
            self.hold_gripper_percent,
        )

    def path_complete(self):
        if self.path_goal_pose is None or self.phase_started_at is None:
            return True

        return self.phase_elapsed_sec() >= self.path_duration_sec

    def update_path_target(self):
        if self.phase not in PATH_PHASES or self.path_goal_pose is None:
            return

        if self.path_duration_sec <= 0.0:
            alpha = 1.0
        else:
            alpha = self.clamp(
                self.phase_elapsed_sec() / self.path_duration_sec,
                0.0,
                1.0,
            )

        start = self.path_start_pose
        goal = self.path_goal_pose
        pose = CylindricalPose(
            radius=start.radius + alpha * (goal.radius - start.radius),
            theta_deg=start.theta_deg + alpha * (goal.theta_deg - start.theta_deg),
            z=start.z + alpha * (goal.z - start.z),
        )
        target = self.make_cylindrical_pose(
            pose,
            self.path_gripper_percent,
        )

        if target is None:
            self.phase = "FAILED"
            self.current_target = None
            self.clear_path()
            self.publish_status(self.phase)
            self.get_logger().warn("random place failed: invalid interpolated target")
            return

        self.current_target = target

    def make_release_pose(self):
        if self.random_target_pose is None:
            return None

        return self.make_cylindrical_pose(
            self.random_target_pose,
            self.release_gripper_percent,
        )

    def apply_realtime_joint4_target(self):
        if self.current_target is None or len(self.current_target) < 4:
            return False

        if self.latest_joint_positions is not None:
            joint2_rad = self.latest_joint_positions[1]
            joint3_rad = self.latest_joint_positions[2]
        else:
            joint2_rad = self.current_target[1]
            joint3_rad = self.current_target[2]

        joint4_deg = self.wrist_flex_deg(
            math.degrees(joint2_rad),
            math.degrees(joint3_rad),
        )
        if not self.in_range(
            joint4_deg,
            self.wrist_flex_min_deg,
            self.wrist_flex_max_deg,
        ):
            self.log_progress(
                f"random place: realtime joint4 out of range: {joint4_deg:.1f} deg"
            )
            return False

        self.current_target[3] = math.radians(joint4_deg)
        return True

    def start_return_home_lift_phase(self):
        start_pose = self.random_target_pose

        if start_pose is None:
            start_pose = self.pose_or_path_goal()

        if start_pose is None:
            self.phase = "FAILED"
            self.publish_status(self.phase)
            self.get_logger().warn("random place failed: missing return-home lift pose")
            return

        goal_pose = CylindricalPose(
            radius=start_pose.radius,
            theta_deg=start_pose.theta_deg,
            z=max(start_pose.z, self.return_home_z),
        )
        self.set_cylindrical_path_phase(
            "LIFTING_TO_HOME",
            start_pose,
            goal_pose,
            self.return_home_duration(self.return_home_lift_duration_sec),
            self.release_gripper_percent,
        )

    def return_home_duration(self, duration_sec):
        return max(0.0, float(duration_sec)) / self.return_home_speed_scale

    def start_return_home_retract_radius_phase(self):
        start_pose = self.pose_or_path_goal()
        if start_pose is None:
            self.phase = "FAILED"
            self.publish_status(self.phase)
            self.get_logger().warn("random place failed: no current pose for home retract")
            return

        goal_pose = retract_radius_pose(start_pose, self.pre_pan_radius)
        self.set_cylindrical_path_phase(
            "RETRACTING_HOME_RADIUS",
            start_pose,
            goal_pose,
            self.return_home_duration(self.pre_pan_retract_duration_sec),
            self.release_gripper_percent,
        )

    def start_return_home_pan_phase(self):
        start_pose = self.pose_or_path_goal()
        if start_pose is None:
            self.phase = "FAILED"
            self.publish_status(self.phase)
            self.get_logger().warn("random place failed: no current pose for home pan")
            return

        return_theta_deg = self.kinematics.theta_deg_from_base(
            self.return_pan_deg,
            fallback_deg=0.0,
        )
        goal_pose = pan_at_current_radius_pose(
            start_pose,
            return_theta_deg,
            start_pose.z,
        )
        self.set_cylindrical_path_phase(
            "PANNING_HOME",
            start_pose,
            goal_pose,
            self.return_home_duration(self.horizontal_move_duration_sec),
            self.release_gripper_percent,
        )

    def joint_errors_deg(self):
        if self.current_target is None or self.latest_joint_positions is None:
            return []

        return [
            abs(math.degrees(current - target))
            for current, target in zip(
                self.latest_joint_positions[:self.target_check_joint_count],
                self.current_target[:self.target_check_joint_count],
            )
        ]

    def at_target(self):
        errors_deg = self.joint_errors_deg()

        if not errors_deg:
            return False

        tolerance_deg = math.degrees(self.joint_goal_tolerance)
        return all(error_deg <= tolerance_deg for error_deg in errors_deg)

    def phase_elapsed_sec(self):
        if self.phase_started_at is None:
            return 0.0

        return time.monotonic() - self.phase_started_at

    def gripper_phase_waiting(self, duration_sec):
        if self.phase_started_at is None:
            return True

        elapsed = self.phase_elapsed_sec()

        if elapsed < duration_sec:
            self.log_progress(
                f"random place: {self.phase} waiting "
                f"{elapsed:.2f}/{duration_sec:.2f} sec"
            )
            return True

        return False

    def log_progress(self, text, period_sec=1.0):
        now = time.monotonic()

        if now - self.last_progress_log_time >= period_sec:
            self.get_logger().info(text)
            self.last_progress_log_time = now

    def log_target_wait(self):
        if self.current_target is None:
            self.log_progress(f"random place: {self.phase} waiting for target")
            return

        if self.latest_joint_positions is None:
            self.log_progress(f"random place: {self.phase} waiting for real joint feedback")
            return

        errors_deg = self.joint_errors_deg()

        if not errors_deg:
            return

        max_index = max(range(len(errors_deg)), key=errors_deg.__getitem__)
        self.log_progress(
            f"random place: {self.phase} waiting for joint {max_index + 1}, "
            f"error={errors_deg[max_index]:.1f} deg, "
            f"tolerance={math.degrees(self.joint_goal_tolerance):.1f} deg"
        )

    def timeout_advance_ready(self):
        if not self.allow_timeout_advance or self.phase_timeout_sec <= 0.0:
            return False

        elapsed = self.phase_elapsed_sec()

        if elapsed < self.phase_timeout_sec:
            return False

        errors_deg = self.joint_errors_deg()

        if not errors_deg:
            return False

        max_index = max(range(len(errors_deg)), key=errors_deg.__getitem__)
        max_error = errors_deg[max_index]

        if max_error <= self.timeout_advance_tolerance_deg:
            self.get_logger().warn(
                f"random place: {self.phase} timeout advance after {elapsed:.1f}s; "
                f"joint {max_index + 1} error={max_error:.1f} deg, "
                f"soft_tolerance={self.timeout_advance_tolerance_deg:.1f} deg"
            )
            return True

        self.log_progress(
            f"random place: {self.phase} timeout reached, but joint {max_index + 1} "
            f"error={max_error:.1f} deg exceeds soft_tolerance="
            f"{self.timeout_advance_tolerance_deg:.1f} deg"
        )
        return False

    def publish_target(self):
        if self.current_target is None:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.current_target
        self.target_pub.publish(msg)

    def set_phase(self, phase, target):
        if target is None:
            self.phase = "FAILED"
            self.publish_status(self.phase)
            self.get_logger().warn("random place failed: invalid target")
            return

        self.phase = phase
        self.current_target = target
        self.clear_path()
        self.settle_count = 0
        self.phase_started_at = time.monotonic()
        self.last_progress_log_time = 0.0
        self.publish_status(self.phase)
        self.get_logger().info(f"random place: {self.phase}")

    def advance_if_ready(self):
        if self.phase == "GRIPPING":
            if self.gripper_phase_waiting(self.grip_settle_sec):
                return

            self.start_lifting_phase()
            return

        if self.phase == "RELEASING":
            if self.gripper_phase_waiting(self.release_settle_sec):
                return

            self.start_return_home_lift_phase()
            return

        if self.phase in PATH_PHASES and not self.path_complete():
            return

        target_ready = self.at_target()

        if not target_ready and self.timeout_advance_ready():
            target_ready = True
            self.settle_count = self.settle_cycles

        if not target_ready:
            self.settle_count = 0
            self.log_target_wait()
            return

        self.settle_count += 1

        if self.settle_count < self.settle_cycles:
            return

        if self.phase == "LIFTING":
            self.start_retract_radius_phase()

        elif self.phase == RETRACTING_RADIUS:
            self.start_horizontal_move_phase()

        elif self.phase == "MOVING_TO_RANDOM_PLACE":
            self.start_extend_radius_phase()

        elif self.phase == EXTENDING_RADIUS:
            self.start_lowering_phase()

        elif self.phase == "LOWERING_TO_RELEASE":
            self.set_phase("RELEASING", self.make_release_pose())

        elif self.phase == "LIFTING_TO_HOME":
            self.start_return_home_retract_radius_phase()

        elif self.phase == "RETRACTING_HOME_RADIUS":
            self.start_return_home_pan_phase()

        elif self.phase == "PANNING_HOME":
            self.set_phase("RETURNING_HOME", self.make_return_pose())

        elif self.phase == "RETURNING_HOME":
            self.phase = "DONE"
            self.phase_started_at = time.monotonic()
            self.publish_status(self.phase)
            self.get_logger().info("random place: DONE")

    def loop(self):
        if self.phase == "GRIPPING" and self.current_target is None:
            self.current_target = self.make_grip_pose_from_current()

            if self.current_target is not None:
                self.phase_started_at = time.monotonic()
                self.last_progress_log_time = 0.0
                self.get_logger().info("random place: GRIPPING target ready")
            else:
                self.log_progress("random place: GRIPPING waiting for real joint feedback")

        if self.phase == "RETURNING_HOME":
            self.current_target = self.make_return_pose()

        if self.phase in ACTIVE_PHASES:
            self.publish_status(self.phase)
            self.update_path_target()

            if self.phase not in ACTIVE_PHASES:
                return

            self.apply_realtime_joint4_target()
            self.publish_target()
            self.advance_if_ready()

        elif self.phase == "DONE" and not self.done_reported:
            self.publish_status("DONE")
            self.done_reported = True


def main(args=None):
    rclpy.init(args=args)
    node = RandomPlaceCupNode()

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

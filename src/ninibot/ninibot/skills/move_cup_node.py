# flake8: noqa
import math
import time

import numpy as np
from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import tf2_ros

from arm_controller.kinematics.cylindrical import CylindricalArmKinematics
from ninibot.skills.move_cup.config import MoveCupConfig
from ninibot.skills.move_cup.carry_planner import MoveCupCarryPlanner
from ninibot.skills.move_cup.carry_planner import MoveCupCarryPlannerConfig
from ninibot.skills.move_cup.phases import ACTIVE_PHASES
from ninibot.skills.move_cup.phases import EXTENDING_RADIUS
from ninibot.skills.move_cup.phases import GRIPPING
from ninibot.skills.move_cup.phases import LIFTING
from ninibot.skills.move_cup.phases import PATH_PHASES
from ninibot.skills.move_cup.phases import POST_RELEASE_CLEARANCE
from ninibot.skills.move_cup.phases import RETRACTING_RADIUS
from ninibot.skills.move_cup.phases import SEARCH_TARGET
from ninibot.skills.move_cup.phases import SEARCH_TARGET_PREP
from ninibot.skills.move_cup.phases import TARGET_LOCKED
from ninibot.skills.move_cup.path_runtime import MoveCupPathRuntime
from ninibot.skills.move_cup.pose_planner import MoveCupPosePlanner
from ninibot.skills.move_cup.pose_planner import MoveCupPosePlannerConfig
from ninibot.skills.move_cup.pose_planner import format_cm as format_xyz_cm
from ninibot.skills.move_cup.runtime import MoveCupTargetRuntime
from ninibot.skills.move_cup.search_prep import MoveCupSearchPrepPlanner
from ninibot.skills.move_cup.search_prep import SearchPrepConfig
from ninibot.skills.move_cup.state_machine import COMMAND_CANCEL
from ninibot.skills.move_cup.state_machine import COMMAND_RESET
from ninibot.skills.move_cup.state_machine import COMMAND_START_LOCKED
from ninibot.skills.move_cup.state_machine import COMMAND_START_MOVE
from ninibot.skills.move_cup.state_machine import COMMAND_START_SCAN
from ninibot.skills.move_cup.state_machine import ACTION_DONE
from ninibot.skills.move_cup.state_machine import ACTION_NONE
from ninibot.skills.move_cup.state_machine import ACTION_START_EXTEND_RADIUS
from ninibot.skills.move_cup.state_machine import ACTION_START_HORIZONTAL_MOVE
from ninibot.skills.move_cup.state_machine import ACTION_START_LIFTING
from ninibot.skills.move_cup.state_machine import ACTION_START_LOWERING
from ninibot.skills.move_cup.state_machine import ACTION_START_POST_RELEASE_CLEARANCE
from ninibot.skills.move_cup.state_machine import ACTION_START_RELEASING
from ninibot.skills.move_cup.state_machine import ACTION_START_RETRACT_RADIUS
from ninibot.skills.move_cup.state_machine import ACTION_START_RETURN_SEARCH_PREP
from ninibot.skills.move_cup.state_machine import MoveCupController
from ninibot.skills.move_cup.state_machine import MoveCupControllerConfig
from ninibot.skills.move_cup.state_machine import MoveCupObservation
from ninibot.skills.move_cup.state_machine import MoveCupState
from ninibot.skills.move_cup.state_machine import can_start_from_phase
from ninibot.skills.move_cup.state_machine import parse_move_cup_command
from ninibot.skills.move_cup.target_scan import TargetScanConfig
from ninibot.skills.move_cup.target_scan import TargetScanPlanner
from ninibot.task_params import MOVE_CUP_PARAMS
from vision_controller.aruco_detection import normalize_camera_source


class MoveCupNode(Node):
    def __init__(self):
        super().__init__("move_cup_node")

        self.declare_parameters(
            namespace="",
            parameters=list(MOVE_CUP_PARAMS.items()),
        )

        self.config = MoveCupConfig.from_node(self)
        self.config.apply_to_node(self)
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
        self.pose_planner = MoveCupPosePlanner(
            MoveCupPosePlannerConfig(
                base_min_deg=self.base_min_deg,
                base_max_deg=self.base_max_deg,
                shoulder_min_deg=self.shoulder_min_deg,
                shoulder_max_deg=self.shoulder_max_deg,
                elbow_min_deg=self.elbow_min_deg,
                elbow_max_deg=self.elbow_max_deg,
                wrist_flex_min_deg=self.wrist_flex_min_deg,
                wrist_flex_max_deg=self.wrist_flex_max_deg,
                wrist_roll_deg=self.wrist_roll_deg,
                hold_gripper_percent=self.hold_gripper_percent,
                release_gripper_percent=self.release_gripper_percent,
                target_radius_margin_m=self.target_radius_margin_m,
                target_search_theta_min_deg=self.target_search_theta_min_deg,
                target_search_theta_max_deg=self.target_search_theta_max_deg,
                search_joint2_deg=self.search_joint2_deg,
                search_joint3_deg=self.search_joint3_deg,
                search_joint4_deg=self.search_joint4_deg,
                return_joint2_deg=self.return_joint2_deg,
                return_joint3_deg=self.return_joint3_deg,
                return_search_prep_final_offset_deg=(
                    self.return_search_prep_final_offset_deg
                ),
            ),
            self.kinematics,
        )
        self.carry_planner = MoveCupCarryPlanner(
            MoveCupCarryPlannerConfig(
                lift_z_offset=self.lift_z_offset,
                pre_pan_radius=self.pre_pan_radius,
                post_release_clearance_retract_m=(
                    self.post_release_clearance_retract_m
                ),
                base_min_deg=self.base_min_deg,
                base_max_deg=self.base_max_deg,
            ),
            self.kinematics,
        )
        self.search_prep_planner = MoveCupSearchPrepPlanner(
            SearchPrepConfig(
                min_duration_sec=self.search_prep_min_duration_sec,
                head_down_duration_sec=self.search_prep_head_down_duration_sec,
                duration_scale=self.search_prep_duration_scale,
                trajectory_samples=self.search_prep_trajectory_samples,
                max_vel_deg=tuple(self.search_prep_max_vel_deg),
                max_acc_deg=tuple(self.search_prep_max_acc_deg),
                final_wrist_offset_deg=self.search_joint4_deg,
                wrist_flex_min_deg=self.wrist_flex_min_deg,
                wrist_flex_max_deg=self.wrist_flex_max_deg,
            ),
            self.wrist_flex_deg,
        )
        self.target_scan_planner = TargetScanPlanner(
            TargetScanConfig(
                theta_min_deg=self.target_search_theta_min_deg,
                theta_max_deg=self.target_search_theta_max_deg,
                theta_step_deg=self.target_search_step_deg,
                lock_samples=self.target_lock_samples,
                lock_max_std_m=self.target_lock_max_std_m,
            )
        )
        self.path_runtime = MoveCupPathRuntime(self)
        self.target_runtime = MoveCupTargetRuntime(self)
        self.controller = MoveCupController(
            MoveCupControllerConfig(settle_cycles=self.settle_cycles)
        )

        self.target_pub = self.create_publisher(JointState, self.target_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.target_base_pub = self.create_publisher(
            PointStamped,
            self.target_base_topic,
            10,
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
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
        self.path_start_gripper_percent = self.hold_gripper_percent
        self.path_goal_gripper_percent = self.hold_gripper_percent
        self.active_carry_z = None
        self.cap = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.detect_markers = None
        self.camera_ok = False
        self.debug_window_failed = False
        self.camera_source = normalize_camera_source(
            self.camera_device.strip() if self.camera_device.strip() else self.camera_index
        )
        self.target_samples = []
        self.release_cylindrical_pose = None
        self.locked_target_base_xyz = None
        self.locked_marker_base_xyz = None
        self.locked_marker_camera_xyz = None
        self.target_locking = False
        self.search_theta_deg = 0.0
        self.search_direction = 1.0
        self.search_prep_trajectory = None
        self.return_search_prep_trajectory = None
        self.last_target_error = ""

        self.timer = self.create_timer(1.0 / self.control_rate, self.loop)

        self.publish_status("IDLE")
        self.get_logger().info(
            "move_cup_node started; waiting for /move_cup_command=start "
            "or scan"
        )

    def clamp(self, value, lower, upper):
        return self.pose_planner.clamp(value, lower, upper)

    def in_range(self, value, lower, upper):
        return self.pose_planner.in_range(value, lower, upper)

    def clamp_wrist_flex_deg(self, joint4_deg, context=""):
        clamped = self.pose_planner.clamp_wrist_flex_deg(joint4_deg)
        if abs(clamped - joint4_deg) > 1e-6 and context:
            self.log_progress(
                f"move cup: {context} joint4 clamped to {clamped:.1f} deg "
                f"(horizontal={joint4_deg:.1f} deg)"
            )
        return clamped

    def gripper_percent_to_rad(self, percent):
        return self.pose_planner.gripper_percent_to_rad(percent)

    def wrist_flex_deg(self, joint2_deg, joint3_deg):
        return self.pose_planner.wrist_flex_deg(joint2_deg, joint3_deg)

    def current_pan_deg(self):
        return self.pose_planner.current_pan_deg(
            self.latest_joint_positions,
            self.target_theta_deg,
        )

    def current_theta_deg(self):
        return self.pose_planner.current_theta_deg(
            self.latest_joint_positions,
            self.target_theta_deg,
        )

    def current_cylindrical_pose(self):
        return self.pose_planner.current_cylindrical_pose(
            self.latest_joint_positions,
        )

    def max_reachable_radius_for_z(self, z):
        return self.pose_planner.max_reachable_radius_for_z(z)

    def clamp_radius_for_reach_margin(self, radius, z, context):
        plan = self.pose_planner.clamp_radius_for_reach_margin(radius, z, context)
        if plan.warning_text:
            self.get_logger().warn(plan.warning_text)
        return plan.radius

    def make_pose(self, pan_deg, joint2_deg, joint3_deg, gripper_percent):
        plan = self.pose_planner.make_pose(
            pan_deg,
            joint2_deg,
            joint3_deg,
            gripper_percent,
        )
        if plan.error_text:
            self.get_logger().warn(plan.error_text)
        return plan.positions

    def make_joint_pose(
        self,
        pan_deg,
        joint2_deg,
        joint3_deg,
        joint4_deg,
        gripper_percent,
    ):
        plan = self.pose_planner.make_joint_pose(
            pan_deg,
            joint2_deg,
            joint3_deg,
            joint4_deg,
            gripper_percent,
        )
        if plan.error_text:
            self.get_logger().warn(plan.error_text)
        return plan.positions

    def cylindrical_ik(self, radius, theta_deg, z):
        plan = self.pose_planner.cylindrical_ik(radius, theta_deg, z)
        if plan.error_text:
            self.get_logger().warn(plan.error_text)
            return None

        return plan.base_deg, plan.joint2_deg, plan.joint3_deg

    def make_cylindrical_pose(
        self,
        gripper_percent,
        radius=None,
        theta_deg=None,
        z=None,
    ):
        plan = self.pose_planner.make_cylindrical_pose(
            self.target_radius if radius is None else radius,
            self.target_theta_deg if theta_deg is None else theta_deg,
            self.target_z if z is None else z,
            gripper_percent,
        )
        if plan.error_text:
            self.get_logger().warn(plan.error_text)
        return plan.positions

    def make_pose_from_cylindrical(self, pose, gripper_percent):
        plan = self.pose_planner.make_pose_from_cylindrical(
            pose,
            gripper_percent,
        )
        if plan.error_text:
            self.get_logger().warn(plan.error_text)
        return plan.positions

    def make_grip_pose_from_current(self):
        plan = self.pose_planner.make_grip_pose_from_current(
            self.latest_joint_positions,
        )
        if plan.error_text:
            self.get_logger().warn(plan.error_text)
        return plan.positions

    def make_return_pose(self):
        plan = self.pose_planner.make_return_pose(self.current_pan_deg())
        if plan.error_text:
            self.get_logger().warn(plan.error_text)
        return plan.positions

    def clear_path(self):
        self.path_runtime.clear_path()

    def pose_or_path_goal(self):
        return self.path_runtime.pose_or_path_goal()

    def set_cylindrical_path_phase(
        self,
        phase,
        start_pose,
        goal_pose,
        duration_sec,
        gripper_percent,
        goal_gripper_percent=None,
    ):
        self.path_runtime.set_cylindrical_path_phase(
            phase,
            start_pose,
            goal_pose,
            duration_sec,
            gripper_percent,
            goal_gripper_percent,
        )

    def path_complete(self):
        return self.path_runtime.path_complete()

    def update_path_target(self):
        self.path_runtime.update_path_target()

    def start_lifting_phase(self):
        self.path_runtime.start_lifting_phase()

    def start_retract_radius_phase(self):
        self.path_runtime.start_retract_radius_phase()

    def start_horizontal_move_phase(self):
        self.path_runtime.start_horizontal_move_phase()

    def start_extend_radius_phase(self):
        self.path_runtime.start_extend_radius_phase()

    def start_lowering_phase(self):
        self.path_runtime.start_lowering_phase()

    def pose_with_base_pan_offset(self, start_pose, base_pan_offset_deg):
        return self.path_runtime.pose_with_base_pan_offset(
            start_pose,
            base_pan_offset_deg,
        )

    def start_releasing_phase(self, start_pose):
        self.path_runtime.start_releasing_phase(start_pose)

    def start_post_release_clearance_phase(self):
        self.path_runtime.start_post_release_clearance_phase()

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
        joint4_deg = self.clamp_wrist_flex_deg(joint4_deg, "realtime")

        self.current_target[3] = math.radians(joint4_deg)
        return True

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

    def release_z(self):
        return float(self.carry_z - self.lift_z_offset)

    def format_cm(self, xyz):
        return format_xyz_cm(xyz)

    def clamp_target_search_theta(self, theta_deg):
        return self.target_scan_planner.clamp_theta(theta_deg)

    def target_search_goal_joint_degs(self, theta_deg):
        return self.pose_planner.target_search_goal_joint_degs(theta_deg)

    def clamp_target_joint_degs(self, joint_degs, context=""):
        plan = self.pose_planner.clamp_target_joint_degs(joint_degs, context)
        if plan.warning_text:
            self.log_progress(plan.warning_text)
        return plan.joint_degs

    def make_target_search_pose_from_joint_degs(
        self,
        joint_degs,
        clamp_to_limits=False,
        context="",
    ):
        plan = self.pose_planner.make_target_search_pose_from_joint_degs(
            joint_degs,
            clamp_to_limits=clamp_to_limits,
            context=context,
        )
        if plan.warning_text:
            self.log_progress(plan.warning_text)
        if plan.error_text:
            self.get_logger().warn(plan.error_text)
        return plan.positions

    def make_target_search_pose(self, theta_deg):
        plan = self.pose_planner.make_target_search_pose(theta_deg)
        if plan.error_text:
            self.get_logger().warn(plan.error_text)
        return plan.positions

    def start_target_search_prep_trajectory(self):
        if self.latest_joint_positions is None:
            self.current_target = None
            self.search_prep_trajectory = None
            return False

        self.search_theta_deg = self.clamp_target_search_theta(
            self.current_theta_deg()
        )
        start_deg = [
            math.degrees(position)
            for position in self.latest_joint_positions[:4]
        ]
        goal_deg = self.target_search_goal_joint_degs(self.search_theta_deg)
        goal_target = self.make_target_search_pose_from_joint_degs(goal_deg)

        if goal_target is None:
            self.phase = "FAILED"
            self.publish_status(self.phase)
            self.get_logger().warn("move cup: invalid ArUco search-prep pose")
            return False

        start_horizontal_wrist_deg = self.wrist_flex_deg(start_deg[1], start_deg[2])
        start_wrist_offset_deg = start_deg[3] - start_horizontal_wrist_deg
        trajectory = self.search_prep_planner.make_trajectory(
            start_deg,
            goal_deg,
            start_wrist_offset_deg,
            target_theta_deg=self.search_theta_deg,
        )
        self.search_prep_trajectory = trajectory
        self.phase_started_at = time.monotonic()
        self.current_target = self.make_target_search_pose_from_joint_degs(
            self.planned_search_prep_joint_degs_at(
                0.0,
                self.search_prep_trajectory,
            ),
            clamp_to_limits=True,
            context="SEARCH_TARGET_PREP",
        )
        self.get_logger().info(
            "move cup: SEARCH_TARGET_PREP trajectory "
            f"theta={self.search_theta_deg:.1f}deg "
            f"duration={trajectory.duration_sec:.2f}s "
            f"head_down_start={trajectory.head_down_start_sec:.2f}s "
            f"start_offset={start_wrist_offset_deg:+.1f}deg"
        )
        return True

    def current_joint_degs_for_trajectory(self):
        if self.latest_joint_positions is not None and len(self.latest_joint_positions) >= 4:
            return [
                math.degrees(position)
                for position in self.latest_joint_positions[:4]
            ]

        if self.current_target is not None and len(self.current_target) >= 4:
            return [
                math.degrees(position)
                for position in self.current_target[:4]
            ]

        return None

    def return_search_prep_goal_joint_degs(self, start_deg):
        base_deg = self.clamp(start_deg[0], self.base_min_deg, self.base_max_deg)
        if abs(base_deg - start_deg[0]) > 1e-6:
            self.get_logger().warn(
                "move cup: return search-prep base clamped "
                f"from {start_deg[0]:.1f}deg to {base_deg:.1f}deg"
            )

        joint4_deg = self.clamp_wrist_flex_deg(
            self.wrist_flex_deg(self.return_joint2_deg, self.return_joint3_deg)
            + self.return_search_prep_final_offset_deg,
            "return search prep",
        )
        return [
            base_deg,
            self.return_joint2_deg,
            self.return_joint3_deg,
            joint4_deg,
        ]

    def start_return_search_prep_trajectory(self):
        start_deg = self.current_joint_degs_for_trajectory()
        if start_deg is None:
            self.set_phase(
                "RETURNING_SEARCH_PREP",
                self.make_return_pose(),
            )
            self.return_search_prep_trajectory = None
            return

        goal_deg = self.return_search_prep_goal_joint_degs(start_deg)
        goal_target = self.make_target_search_pose_from_joint_degs(goal_deg)
        if goal_target is None:
            self.phase = "FAILED"
            self.current_target = None
            self.publish_status(self.phase)
            self.get_logger().warn("move cup failed: invalid return search-prep pose")
            return

        start_horizontal_wrist_deg = self.wrist_flex_deg(start_deg[1], start_deg[2])
        start_wrist_offset_deg = start_deg[3] - start_horizontal_wrist_deg
        target_theta_deg = self.kinematics.theta_deg_from_base(
            goal_deg[0],
            fallback_deg=self.current_theta_deg(),
        )
        trajectory = self.search_prep_planner.make_trajectory(
            start_deg,
            goal_deg,
            start_wrist_offset_deg,
            target_theta_deg=target_theta_deg,
            final_wrist_offset_deg=self.return_search_prep_final_offset_deg,
        )
        self.return_search_prep_trajectory = trajectory
        self.phase = "RETURNING_SEARCH_PREP"
        self.current_target = self.make_target_search_pose_from_joint_degs(
            self.planned_search_prep_joint_degs_at(
                0.0,
                self.return_search_prep_trajectory,
            )
        )
        self.clear_path()
        self.settle_count = 0
        self.phase_started_at = time.monotonic()
        self.last_progress_log_time = 0.0
        self.publish_status(self.phase)
        self.get_logger().info(
            "move cup: RETURNING_SEARCH_PREP trajectory "
            f"theta={target_theta_deg:.1f}deg "
            f"duration={trajectory.duration_sec:.2f}s "
            f"head_down_start={trajectory.head_down_start_sec:.2f}s "
            f"start_offset={start_wrist_offset_deg:+.1f}deg "
            f"final_offset={self.return_search_prep_final_offset_deg:+.1f}deg"
        )

    def search_prep_duration_sec(
        self,
        start_deg,
        goal_deg,
        start_wrist_offset_deg,
        final_wrist_offset_deg=None,
    ):
        return self.search_prep_planner.duration_sec(
            start_deg,
            goal_deg,
            start_wrist_offset_deg,
            final_wrist_offset_deg,
        )

    def sample_search_prep_limits(
        self,
        start_deg,
        goal_deg,
        start_wrist_offset_deg,
        final_wrist_offset_deg,
        duration_sec,
    ):
        return self.search_prep_planner.sample_limits(
            start_deg,
            goal_deg,
            start_wrist_offset_deg,
            duration_sec,
            final_wrist_offset_deg,
        )

    def planned_search_prep_joint_degs_at(self, elapsed_sec, trajectory):
        return self.search_prep_planner.joint_degs_at(elapsed_sec, trajectory)

    def search_prep_wrist_offset_at(self, elapsed_sec, trajectory):
        return self.search_prep_planner.wrist_offset_at(elapsed_sec, trajectory)

    def ensure_aruco_ready(self):
        return self.target_runtime.ensure_aruco_ready()

    def open_camera(self):
        return self.target_runtime.open_camera()

    def release_camera(self):
        self.target_runtime.release_camera()

    def publish_target_base(self, target_base_xyz):
        self.target_runtime.publish_target_base(target_base_xyz)

    def apply_locked_target(self, target_base_xyz):
        return self.target_runtime.apply_locked_target(target_base_xyz)

    def clear_locked_target(self):
        self.target_runtime.clear_locked_target()

    def lock_target(self, target_base_xyz, marker_base_xyz, marker_camera_xyz):
        self.target_runtime.lock_target(
            target_base_xyz,
            marker_base_xyz,
            marker_camera_xyz,
        )

    def draw_marker_offset_target(self, frame, rvec, tvec):
        return self.target_runtime.draw_marker_offset_target(frame, rvec, tvec)

    def show_aruco_debug(self, frame, corners=None, ids=None, rvec=None, tvec=None, status=""):
        self.target_runtime.show_aruco_debug(
            frame,
            corners=corners,
            ids=ids,
            rvec=rvec,
            tvec=tvec,
            status=status,
        )

    def detect_target(self):
        return self.target_runtime.detect_target()

    def update_target_search_pan(self):
        self.target_runtime.update_target_search_pan()

    def update_target_search(self):
        self.target_runtime.update_target_search()

    def command_callback(self, msg):
        command = parse_move_cup_command(msg.data)

        if command == COMMAND_START_MOVE:
            self.start_move()
        elif command == COMMAND_START_SCAN:
            self.start_target_scan()
        elif command == COMMAND_START_LOCKED:
            self.start_move_to_locked_target()
        elif command == COMMAND_CANCEL:
            self.cancel_move()
        elif command == COMMAND_RESET:
            self.reset_to_idle()
        else:
            self.get_logger().warn(f"Unknown move cup command: {msg.data}")

    def start_target_scan(self):
        if not can_start_from_phase(self.phase):
            self.get_logger().warn(f"Move cup already active: {self.phase}")
            return

        if not self.ensure_aruco_ready():
            self.phase = "FAILED"
            self.current_target = None
            self.publish_status(self.phase)
            return

        self.clear_locked_target()
        self.done_reported = False
        self.settle_count = 0
        self.last_progress_log_time = 0.0
        self.active_carry_z = None
        self.clear_path()
        self.search_direction = 1.0
        self.phase = SEARCH_TARGET_PREP
        self.phase_started_at = time.monotonic()
        self.current_target = None
        self.search_prep_trajectory = None
        self.return_search_prep_trajectory = None
        if self.latest_joint_positions is not None:
            if not self.start_target_search_prep_trajectory():
                return
        self.publish_status(self.phase)
        theta_text = (
            "pending"
            if self.current_target is None
            else f"{self.search_theta_deg:.1f}deg"
        )
        self.get_logger().info(
            "move cup: SEARCH_TARGET_PREP "
            f"id={self.target_marker_id}, "
            f"theta={theta_text}, "
            f"offset_cm={self.format_cm(self.target_marker_offset_xyz)}"
        )

    def start_move_to_locked_target(self):
        if self.locked_target_base_xyz is None:
            reason = self.last_target_error or "scan ArUco target first"
            self.get_logger().warn(f"move cup: no locked target; {reason}")
            self.phase = "FAILED"
            self.publish_status(self.phase)
            return

        if not self.apply_locked_target(self.locked_target_base_xyz):
            self.phase = "FAILED"
            self.publish_status(self.phase)
            return

        self.start_move()

    def start_move(self):
        if not can_start_from_phase(self.phase):
            self.get_logger().warn(f"Move cup already active: {self.phase}")
            return

        self.phase = "GRIPPING"
        self.done_reported = False
        self.settle_count = 0
        self.last_progress_log_time = 0.0
        self.active_carry_z = None
        self.release_cylindrical_pose = None
        self.return_search_prep_trajectory = None
        self.clear_path()
        self.current_target = self.make_grip_pose_from_current()
        self.phase_started_at = (
            time.monotonic() if self.current_target is not None else None
        )

        if self.current_target is None:
            self.get_logger().warn("move cup: waiting for real joint feedback before gripping")

        self.publish_status(self.phase)
        self.get_logger().info("move cup: GRIPPING")

    def cancel_move(self):
        if self.phase in ("IDLE", "DONE", "CANCELED"):
            return

        self.release_camera()
        self.phase = "CANCELED"
        self.current_target = None
        self.search_prep_trajectory = None
        self.return_search_prep_trajectory = None
        self.release_cylindrical_pose = None
        self.clear_path()
        self.phase_started_at = time.monotonic()
        self.publish_status(self.phase)
        self.get_logger().info("move cup: CANCELED")

    def reset_to_idle(self):
        self.release_camera()
        previous_phase = self.phase
        self.phase = "IDLE"
        self.current_target = None
        self.done_reported = False
        self.settle_count = 0
        self.phase_started_at = None
        self.last_progress_log_time = 0.0
        self.active_carry_z = None
        self.release_cylindrical_pose = None
        self.search_prep_trajectory = None
        self.return_search_prep_trajectory = None
        self.clear_locked_target()
        self.clear_path()
        self.publish_status(self.phase)
        if previous_phase != "IDLE":
            self.get_logger().info("move cup: IDLE")

    def joint_errors_deg(self):
        if self.current_target is None or self.latest_joint_positions is None:
            return []

        return [
            abs(math.degrees(current - target_rad))
            for current, target_rad in zip(
                self.latest_joint_positions[:self.target_check_joint_count],
                self.current_target[:self.target_check_joint_count],
            )
        ]

    def at_target(self):
        errors_deg = self.joint_errors_deg()

        if not errors_deg:
            return False

        tolerance_deg = math.degrees(self.joint_goal_tolerance)

        for error_deg in errors_deg:
            if error_deg > tolerance_deg:
                return False

        return True

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
                f"move cup: {self.phase} waiting "
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
            self.log_progress(f"move cup: {self.phase} waiting for target")
            return

        if self.latest_joint_positions is None:
            self.log_progress(f"move cup: {self.phase} waiting for real joint feedback")
            return

        errors_deg = self.joint_errors_deg()

        if not errors_deg:
            return

        max_index = max(range(len(errors_deg)), key=errors_deg.__getitem__)
        self.log_progress(
            f"move cup: {self.phase} waiting for joint {max_index + 1}, "
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
                f"move cup: {self.phase} timeout advance after {elapsed:.1f}s; "
                f"joint {max_index + 1} error={max_error:.1f} deg, "
                f"soft_tolerance={self.timeout_advance_tolerance_deg:.1f} deg"
            )
            return True

        self.log_progress(
            f"move cup: {self.phase} timeout reached, but joint {max_index + 1} "
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

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def set_phase(self, phase, target):
        if target is None:
            self.phase = "FAILED"
            self.publish_status(self.phase)
            self.get_logger().warn("move cup failed: invalid target")
            return

        self.phase = phase
        self.current_target = target
        self.clear_path()
        self.settle_count = 0
        self.phase_started_at = time.monotonic()
        self.last_progress_log_time = 0.0
        self.publish_status(self.phase)
        self.get_logger().info(f"move cup: {self.phase}")

    def advance_if_ready(self):
        path_complete = True
        gripper_waiting = False
        target_ready = False
        timeout_advance_ready = False

        if self.phase == GRIPPING:
            gripper_waiting = self.gripper_phase_waiting(self.grip_settle_sec)
        elif self.phase in PATH_PHASES:
            path_complete = self.path_complete()
            if path_complete:
                target_ready = self.at_target()
                if not target_ready and self.phase != LIFTING:
                    timeout_advance_ready = self.timeout_advance_ready()
        else:
            target_ready = self.at_target()
            if not target_ready and self.phase != LIFTING:
                timeout_advance_ready = self.timeout_advance_ready()

        observation = MoveCupObservation(
            target_ready=target_ready,
            path_complete=path_complete,
            gripper_waiting=gripper_waiting,
            timeout_advance_ready=timeout_advance_ready,
        )
        step = self.controller.advance_if_ready(
            MoveCupState(self.phase, self.settle_count),
            observation,
        )
        self.settle_count = step.state.settle_count

        if step.log_target_wait:
            self.log_target_wait()

        self.execute_controller_action(step.action)

    def execute_controller_action(self, action):
        if action == ACTION_NONE:
            return
        if action == ACTION_START_LIFTING:
            self.start_lifting_phase()
        elif action == ACTION_START_RETRACT_RADIUS:
            self.start_retract_radius_phase()
        elif action == ACTION_START_HORIZONTAL_MOVE:
            self.start_horizontal_move_phase()
        elif action == ACTION_START_EXTEND_RADIUS:
            self.start_extend_radius_phase()
        elif action == ACTION_START_LOWERING:
            self.start_lowering_phase()
        elif action == ACTION_START_RELEASING:
            release_pose = self.pose_or_path_goal()
            self.start_releasing_phase(release_pose)
        elif action == ACTION_START_POST_RELEASE_CLEARANCE:
            self.start_post_release_clearance_phase()
        elif action == ACTION_START_RETURN_SEARCH_PREP:
            self.start_return_search_prep_trajectory()
        elif action == ACTION_DONE:
            self.phase = "DONE"
            self.phase_started_at = time.monotonic()
            self.publish_status(self.phase)
            self.get_logger().info("move cup: DONE")

    def handle_search_target_prep(self):
        self.publish_status(self.phase)
        if self.search_prep_trajectory is None:
            if self.latest_joint_positions is None:
                self.log_progress(
                    "move cup: SEARCH_TARGET_PREP waiting for real joint "
                    "feedback before planning trajectory"
                )
                return
            if not self.start_target_search_prep_trajectory():
                return

        elapsed_sec = self.phase_elapsed_sec()
        trajectory = self.search_prep_trajectory
        if elapsed_sec < trajectory.duration_sec:
            planned_degs = self.planned_search_prep_joint_degs_at(
                elapsed_sec,
                trajectory,
            )
            refreshed_target = self.make_target_search_pose_from_joint_degs(
                planned_degs,
                clamp_to_limits=True,
                context="SEARCH_TARGET_PREP",
            )
            if refreshed_target is None:
                self.phase = "FAILED"
                self.publish_status(self.phase)
                self.get_logger().warn("move cup: invalid ArUco search-prep pose")
                return
            self.current_target = refreshed_target
            self.publish_target()
            return

        final_target = self.make_target_search_pose_from_joint_degs(
            self.planned_search_prep_joint_degs_at(
                trajectory.duration_sec,
                trajectory,
            )
        )
        if final_target is None:
            self.phase = "FAILED"
            self.publish_status(self.phase)
            self.get_logger().warn("move cup: invalid ArUco search-prep pose")
            return
        self.current_target = final_target

        self.publish_target()
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

        self.phase = SEARCH_TARGET
        self.settle_count = 0
        self.phase_started_at = time.monotonic()
        self.last_progress_log_time = 0.0
        self.publish_status(self.phase)
        self.get_logger().info(
            "move cup: SEARCH_TARGET "
            f"theta_range=[{self.target_search_theta_min_deg:.1f}, "
            f"{self.target_search_theta_max_deg:.1f}]deg"
        )

    def handle_search_target(self):
        self.publish_status(self.phase)
        self.update_target_search()
        if self.phase == SEARCH_TARGET:
            self.publish_target()

    def handle_target_locked(self):
        self.publish_status(self.phase)

    def handle_returning_search_prep(self):
        self.publish_status(self.phase)
        trajectory = self.return_search_prep_trajectory
        if trajectory is None:
            if self.current_target is None:
                self.start_return_search_prep_trajectory()
                return

            self.publish_target()
            self.advance_if_ready()
            return

        elapsed_sec = self.phase_elapsed_sec()
        if elapsed_sec < trajectory.duration_sec:
            planned_degs = self.planned_search_prep_joint_degs_at(
                elapsed_sec,
                trajectory,
            )
            refreshed_target = self.make_target_search_pose_from_joint_degs(
                planned_degs
            )
            if refreshed_target is None:
                self.phase = "FAILED"
                self.publish_status(self.phase)
                self.get_logger().warn(
                    "move cup failed: invalid return search-prep target"
                )
                return

            self.current_target = refreshed_target
            self.publish_target()
            return

        final_target = self.make_target_search_pose_from_joint_degs(
            self.planned_search_prep_joint_degs_at(
                trajectory.duration_sec,
                trajectory,
            )
        )
        if final_target is None:
            self.phase = "FAILED"
            self.publish_status(self.phase)
            self.get_logger().warn(
                "move cup failed: invalid return search-prep final target"
            )
            return

        self.current_target = final_target
        self.publish_target()
        self.advance_if_ready()

    def handle_gripping_wait(self):
        self.current_target = self.make_grip_pose_from_current()

        if self.current_target is not None:
            self.phase_started_at = time.monotonic()
            self.last_progress_log_time = 0.0
            self.get_logger().info("move cup: GRIPPING target ready")
        else:
            self.log_progress("move cup: GRIPPING waiting for real joint feedback")

    def handle_active_motion(self):
        self.publish_status(self.phase)
        self.update_path_target()

        if self.phase not in ACTIVE_PHASES:
            return

        self.apply_realtime_joint4_target()
        self.publish_target()
        self.advance_if_ready()

    def handle_done(self):
        if self.done_reported:
            return

        self.publish_status("DONE")
        self.done_reported = True

    def loop(self):
        if self.phase == SEARCH_TARGET_PREP:
            self.handle_search_target_prep()
            return

        if self.phase == SEARCH_TARGET:
            self.handle_search_target()
            return

        if self.phase == TARGET_LOCKED:
            self.handle_target_locked()
            return

        if self.phase == "RETURNING_SEARCH_PREP":
            self.handle_returning_search_prep()
            return

        if self.phase == "GRIPPING" and self.current_target is None:
            self.handle_gripping_wait()

        if self.phase in ACTIVE_PHASES:
            self.handle_active_motion()
        elif self.phase == "DONE":
            self.handle_done()

    def destroy_node(self):
        self.release_camera()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MoveCupNode()

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

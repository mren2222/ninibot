# flake8: noqa
from dataclasses import replace
import math
import time

import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from arm_controller.kinematics.cylindrical import CylindricalArmKinematics
from ninibot.skills.grasp_cup.approach_servo import ApproachServo
from ninibot.skills.grasp_cup.approach_servo import ApproachServoConfig
from ninibot.skills.grasp_cup.config import GraspCupParams
from ninibot.skills.grasp_cup.controller import GraspCupController
from ninibot.skills.grasp_cup.controller import GraspCupControllerConfig
from ninibot.skills.grasp_cup.controller import SEARCH_STATUSES
from ninibot.skills.grasp_cup.debug_view import DebugTelemetry
from ninibot.skills.grasp_cup.debug_view import draw_overlay as draw_debug_overlay
from ninibot.skills.grasp_cup.debug_view import make_debug_text as format_debug_text
from ninibot.skills.grasp_cup.detector_manager import CupDetectorManager
from ninibot.skills.grasp_cup.fine_align_prep import FineAlignPrepConfig
from ninibot.skills.grasp_cup.fine_align_prep import FineAlignPrepPlanner
from ninibot.skills.grasp_cup.fine_align_prep import load_template
from ninibot.skills.grasp_cup.pose_planner import PosePlanner
from ninibot.skills.grasp_cup.pose_planner import PosePlannerConfig
from ninibot.skills.grasp_cup.ros_io import GraspCupRosIO
from ninibot.skills.grasp_cup.runtime import GraspCupRuntimeAdapter
from ninibot.skills.grasp_cup.search_prep import SearchPrepConfig
from ninibot.skills.grasp_cup.search_prep import SearchPrepPlanner
from ninibot.task_params import CYLINDRICAL_VISION_PARAMS
from ninibot.skills.grasp_cup.visual_servo import DetectionCenterSmoother
from ninibot.skills.grasp_cup.visual_servo import PanServo
from ninibot.skills.grasp_cup.visual_servo import PanServoConfig


class GraspCupNode(Node):
    def __init__(self):
        super().__init__("grasp_cup_node")

        self.declare_parameters(
            namespace="",
            parameters=list(CYLINDRICAL_VISION_PARAMS.items()),
        )
        self.params = GraspCupParams.from_node(self)
        self.params.apply_to(self)
        self.io = GraspCupRosIO(self, self.params)

        # Initialize callback-visible state before subscriptions or optional
        # detector setup, so a startup error cannot leave the node half-built.
        self.smoothed_center_x = None
        self.last_warn_times = {}
        self.last_status = "STARTING"
        self.stage = "SEARCH_PREP"
        self.search_anchor_theta = self.theta
        self.search_anchor_captured = False
        self.latest_joint_theta = None
        self.latest_joint_positions = None
        self.frame_detection = None
        self.frame_error_x = None
        self.frame_bbox_error = None
        self.frame_bbox_w = 0.0
        self.window_failed = False
        self.search_prep_trajectory = None
        self.search_prep_started_at = 0.0
        self.tracking_lost_started_at = None
        self.command_sub = self.create_subscription(
            String,
            self.command_topic,
            self.command_callback,
            10,
        )
        self.real_joint_state_sub = self.create_subscription(
            JointState,
            self.real_joint_states_topic,
            self.joint_state_callback,
            10,
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
        self.pose_planner = PosePlanner(
            PosePlannerConfig(
                search_joint2_deg=self.search_joint2_deg,
                search_joint3_deg=self.search_joint3_deg,
                search_joint4_deg=self.search_joint4_deg,
                wrist_flex_min_deg=self.wrist_flex_min_deg,
                wrist_flex_max_deg=self.wrist_flex_max_deg,
                wrist_roll_deg=self.wrist_roll_deg,
                gripper_rad=self.gripper_rad,
                base_min_deg=self.base_min_deg,
                base_max_deg=self.base_max_deg,
            ),
            self.kinematics,
        )
        self.search_prep_planner = SearchPrepPlanner(
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
            self.kinematics.wrist_flex_deg,
        )
        self.approach_servo = ApproachServo(
            ApproachServoConfig(
                bbox_tolerance_px=self.bbox_tolerance_px,
                bbox_resume_tolerance_px=self.bbox_resume_tolerance_px,
                radius_kp_m_per_px=self.approach_radius_kp,
                max_radius_step_m=self.approach_max_radius_step,
            )
        )
        self.pan_servo = PanServo(
            PanServoConfig(
                theta_kp=self.theta_kp,
                theta_sign=self.theta_sign,
                theta_max_step=self.theta_max_step,
                tracking_theta_max_step=self.tracking_theta_max_step,
                tracking_settle_tolerance_px=self.tracking_settle_tolerance_px,
                target_bbox_width=self.target_bbox_width,
            )
        )
        self.center_smoother = DetectionCenterSmoother(self.center_filter_alpha)

        self.search_direction = 1.0
        self.theta = self.clamp(self.theta, self.theta_min, self.theta_max)
        self.approach_servo_started = False
        self.approach_locked = False
        self.tracking_centered_count = 0
        self.fine_align_centered_count = 0
        self.servo_radius_m = None
        self.servo_z_m = None
        self.approach_joint2_deg = 0.0
        self.approach_joint3_deg = 0.0
        self.fine_align_prep_template = None
        self.fine_align_prep_planner = None
        self.fine_align_prep_trajectory = None
        self.prep_trajectory_last_j234 = None
        self.controller = GraspCupController(
            GraspCupControllerConfig(
                search_enabled=self.search_enabled,
                search_step=self.search_step,
                theta_min=self.theta_min,
                theta_max=self.theta_max,
                tracking_lost_grace_sec=self.tracking_lost_grace_sec,
                tracking_settle_cycles=self.tracking_settle_cycles,
                fine_align_centered_cycles=self.fine_align_centered_cycles,
                publish_while_lost=self.publish_while_lost,
            )
        )
        self.runtime = GraspCupRuntimeAdapter(self)
        self.detectors = CupDetectorManager(
            self.params,
            self.get_logger(),
            self.warn_throttled,
        )
        self.detector_backend = self.detectors.detector_backend
        self.yolo_error_message = self.detectors.yolo_error_message

        self.load_fine_align_prep_template()

        self.runtime.apply_state(
            self.controller.reset_tracking_state(
                self.runtime.capture_state(),
                *self.runtime.reset_tracking_joint_degs(),
            )
        )
        self.runtime.reset_tracking_runtime()
        self.search_anchor_theta = self.theta
        self.search_anchor_captured = False
        self.stage = "IDLE"
        self.state = self.runtime.capture_state()

        timer_period = 1.0 / self.control_rate
        self.timer = self.create_timer(timer_period, self.loop)

        if self.yolo_error_message:
            self.stage = "YOLO_ERROR"
            self.state = self.runtime.capture_state()
            self.io.publish_status(self.stage)
        elif self.auto_start:
            self.start_servo()
        else:
            self.io.publish_status("IDLE")

        self.get_logger().info("grasp_cup_node started")
        self.get_logger().info(f"camera_source={self.camera_source}")
        self.get_logger().info(
            f"auto_start={self.auto_start}, command_topic={self.command_topic}"
        )
        self.get_logger().info(self.detectors.summary())
        self.get_logger().info(
            f"Search pose j2={self.search_joint2_deg:.1f}, "
            f"j3={self.search_joint3_deg:.1f}, j4_offset={self.search_joint4_deg:.1f}"
        )
        end_j234 = self.fine_align_prep_end_j234()
        self.get_logger().info(
            "Fine align prep "
            "end_source=trajectory_processed_end, "
            f"end_j234_deg={[round(math.degrees(value), 1) for value in end_j234]}, "
            f"trajectory={self.fine_align_prep_trajectory_path}, "
            f"radius_kp={self.approach_radius_kp:.6f}, "
            f"max_radius_step={self.approach_max_radius_step:.4f}"
        )

    def load_fine_align_prep_template(self):
        try:
            self.fine_align_prep_template = load_template(
                self.fine_align_prep_trajectory_path
            )
            self.fine_align_prep_planner = FineAlignPrepPlanner(
                self.fine_align_prep_template,
                self.kinematics,
                FineAlignPrepConfig(
                    duration_sec=self.fine_align_prep_trajectory_duration_sec,
                    shape_scale=self.fine_align_prep_shape_scale,
                    z_offset_m=self.fine_align_prep_z_offset_m,
                    wrist_flex_min_deg=self.wrist_flex_min_deg,
                    wrist_flex_max_deg=self.wrist_flex_max_deg,
                ),
            )
            self.get_logger().info(
                "Loaded fine-align-prep trajectory: "
                f"{self.fine_align_prep_trajectory_path}"
            )
        except Exception as exc:
            self.fine_align_prep_template = None
            self.fine_align_prep_planner = None
            raise RuntimeError(
                "Cannot load fine-align-prep trajectory: "
                f"{self.fine_align_prep_trajectory_path}"
            ) from exc

    def command_callback(self, msg):
        command = msg.data.strip().lower()

        if command in ("start", "grasp"):
            self.start_servo()
        elif command in ("cancel", "stop"):
            self.cancel_servo()
        elif command == "reset":
            self.reset_to_idle()
        else:
            self.get_logger().warn(f"Unknown cup servo command: {msg.data}")

    def start_servo(self):
        if self.stage not in ("IDLE", "CANCELED", "FINISHED", "CAMERA_ERROR", "YOLO_ERROR"):
            self.get_logger().warn(f"cup servo already active: {self.stage}")
            return

        detector_ready, reason = self.detectors.ready_to_start()
        if not detector_ready:
            self.stage = "YOLO_ERROR"
            self.state = self.runtime.capture_state()
            self.get_logger().error(f"cup servo cannot start: {reason}")
            self.io.publish_status(self.stage)
            return

        self.detectors.reset()

        if not self.io.open_camera():
            self.stage = "CAMERA_ERROR"
            self.state = self.runtime.capture_state()
            self.io.publish_status(self.stage)
            return

        self.last_status = "STARTING"
        observation = self.runtime.make_observation(None, 0.0)
        step = self.controller.reset_to_search(
            self.runtime.capture_state(),
            observation,
        )
        self.runtime.execute_step(step)
        if self.skip_search_prep_on_start:
            self.runtime.apply_state(
                replace(
                    self.runtime.capture_state(),
                    search_prep_trajectory=None,
                    stage="SEARCHING",
                )
            )
            self.get_logger().info("cup servo: skipping SEARCH_PREP on start")
        self.get_logger().info("cup servo: START")

    def cancel_servo(self):
        if self.stage in ("IDLE", "CANCELED"):
            return

        self.stage = "CANCELED"
        self.search_prep_trajectory = None
        self.tracking_lost_started_at = None
        self.state = self.runtime.capture_state()
        self.io.release_camera()
        self.io.publish_status(self.stage)
        self.get_logger().info("cup servo: CANCELED")

    def reset_to_idle(self):
        previous_stage = self.stage
        self.stage = "IDLE"
        self.last_status = "STARTING"
        self.search_prep_trajectory = None
        self.tracking_lost_started_at = None
        state = self.controller.reset_tracking_state(
            self.runtime.capture_state(),
            *self.runtime.reset_tracking_joint_degs(),
        )
        self.runtime.apply_state(
            replace(
                state,
                stage="IDLE",
                search_prep_trajectory=None,
                tracking_lost_started_at=None,
            )
        )
        self.runtime.reset_tracking_runtime()
        self.io.release_camera()
        self.io.publish_status(self.stage)
        if previous_stage != "IDLE":
            self.get_logger().info("cup servo: IDLE")

    def clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def warn_throttled(self, key, text, period_sec=1.0):
        now = time.monotonic()
        last = self.last_warn_times.get(key, 0.0)

        if now - last >= period_sec:
            self.get_logger().warn(text)
            self.last_warn_times[key] = now

    def joint_state_callback(self, msg):
        if "1" not in msg.name:
            return

        name_to_position = {
            name: position
            for name, position in zip(msg.name, msg.position)
        }

        if all(str(i) in name_to_position for i in range(1, 7)):
            self.latest_joint_positions = [
                name_to_position[str(i)]
                for i in range(1, 7)
            ]

        index = msg.name.index("1")

        if index >= len(msg.position):
            return

        base_deg = math.degrees(msg.position[index])

        theta_deg = self.kinematics.theta_deg_from_base(
            base_deg,
            fallback_deg=math.degrees(self.theta),
        )
        self.latest_joint_theta = self.clamp(
            math.radians(theta_deg),
            self.theta_min,
            self.theta_max,
        )

        if self.stage == "SEARCH_PREP" and not self.search_anchor_captured:
            self.search_anchor_theta = self.latest_joint_theta
            self.theta = self.search_anchor_theta
            self.search_anchor_captured = True
            self.state = self.runtime.capture_state()

    def publish_target(self, status):
        self.theta = self.clamp(self.theta, self.theta_min, self.theta_max)
        self.state = self.runtime.capture_state()

        positions = self.compute_joint_positions(
            self.theta,
            status,
        )

        if positions is None:
            return

        self.io.publish_target_positions(self.joint_names, positions)

    def joint_feedback_ready(self):
        return (
            self.latest_joint_positions is not None
            and len(self.latest_joint_positions) >= 4
        )

    def current_cylindrical_estimate(self):
        if not self.joint_feedback_ready():
            base_deg = self.pose_planner.base_deg_from_theta(self.theta)
            return self.kinematics.cylindrical_from_joint_degs(
                base_deg,
                self.approach_joint2_deg,
                self.approach_joint3_deg,
            )

        base_deg = math.degrees(self.latest_joint_positions[0])
        shoulder_deg = math.degrees(self.latest_joint_positions[1])
        elbow_deg = math.degrees(self.latest_joint_positions[2])
        return self.kinematics.cylindrical_from_joint_degs(
            base_deg,
            shoulder_deg,
            elbow_deg,
        )

    def current_z_estimate(self):
        return self.current_cylindrical_estimate().z

    def current_radius_estimate(self):
        return self.current_cylindrical_estimate().radius

    def desired_wrist_pitch_rad(self, status):
        fine_align_prep_joint4_rad = None
        if status == "FINE_ALIGN_PREP" and self.prep_trajectory_last_j234 is not None:
            fine_align_prep_joint4_rad = self.prep_trajectory_last_j234[2]

        return self.pose_planner.wrist_pitch_rad(
            status,
            self.current_joint_or_deg(2, self.search_joint2_deg),
            self.current_joint_or_deg(3, self.search_joint3_deg),
            fine_align_prep_joint4_rad=fine_align_prep_joint4_rad,
        )

    def search_goal_joint_degs(self, base_deg):
        return self.pose_planner.search_goal_joint_degs(base_deg)

    def desired_search_joint4_rad(self):
        if self.stage == "SEARCH_PREP" and self.search_prep_trajectory is not None:
            return math.radians(
                self.planned_search_prep_joint_degs_at(
                    self.search_prep_elapsed_sec(),
                    self.search_prep_trajectory,
                )[3]
            )

        base_deg = self.pose_planner.base_deg_from_theta(self.theta)
        return math.radians(self.search_goal_joint_degs(base_deg)[3])

    def search_pose(self, base_deg):
        return self.pose_planner.search_pose(base_deg)

    def current_joint_or_deg(self, joint_number, fallback_deg):
        if self.joint_feedback_ready():
            return self.latest_joint_positions[joint_number - 1]

        return math.radians(fallback_deg)

    def search_prep_elapsed_sec(self):
        if self.search_prep_trajectory is None:
            return 0.0
        return time.monotonic() - self.search_prep_started_at

    def start_search_prep_trajectory(self):
        if self.search_prep_trajectory is not None:
            return True

        if not self.joint_feedback_ready():
            self.warn_throttled(
                "search_prep_wait_feedback",
                "cup servo: SEARCH_PREP waiting for real joint feedback",
                1.0,
            )
            return False

        start_deg = [
            math.degrees(position)
            for position in self.latest_joint_positions[:4]
        ]
        base_deg = self.pose_planner.base_deg_from_theta(self.search_anchor_theta)
        goal_deg = self.search_goal_joint_degs(base_deg)
        start_horizontal_wrist_deg = self.kinematics.wrist_flex_deg(
            start_deg[1],
            start_deg[2],
        )
        start_wrist_offset_deg = start_deg[3] - start_horizontal_wrist_deg
        self.search_prep_trajectory = self.search_prep_planner.make_trajectory(
            start_deg,
            goal_deg,
            start_wrist_offset_deg,
        )
        self.search_prep_started_at = time.monotonic()
        self.get_logger().info(
            "cup servo: SEARCH_PREP trajectory "
            f"duration={self.search_prep_trajectory.duration_sec:.2f}s "
            f"head_down_start={self.search_prep_trajectory.head_down_start_sec:.2f}s "
            f"start_offset={start_wrist_offset_deg:+.1f}deg"
        )
        return True

    def planned_search_prep_joint_degs_at(self, elapsed_sec, trajectory):
        return self.search_prep_planner.joint_degs_at(elapsed_sec, trajectory)

    def search_prep_pose(self):
        if not self.start_search_prep_trajectory():
            return None

        joint_degs = self.planned_search_prep_joint_degs_at(
            self.search_prep_elapsed_sec(),
            self.search_prep_trajectory,
        )
        return self.pose_planner.pose_from_joint_degs(joint_degs)

    def search_prep_pose_ready(self):
        if not self.start_search_prep_trajectory():
            return False

        if self.search_prep_elapsed_sec() < self.search_prep_trajectory.duration_sec:
            return False

        if not self.joint_feedback_ready():
            return False

        goal_deg = self.planned_search_prep_joint_degs_at(
            self.search_prep_trajectory.duration_sec,
            self.search_prep_trajectory,
        )
        for joint_index, target_deg in enumerate(goal_deg):
            current_rad = self.latest_joint_positions[joint_index]
            if abs(current_rad - math.radians(target_deg)) > self.joint_goal_tolerance:
                return False

        return True

    def fine_align_prep_end_j234(self):
        if self.fine_align_prep_planner is None:
            raise RuntimeError("Fine-align-prep trajectory template is not loaded")

        end = self.fine_align_prep_planner.end_j234()
        if end.warning_key:
            self.warn_throttled(
                end.warning_key,
                end.warning_text,
                5.0,
            )

        return end.j234_rad

    def reset_fine_align_prep_trajectory(self):
        self.fine_align_prep_trajectory = None
        self.prep_trajectory_last_j234 = None

    def start_fine_align_prep_trajectory(self):
        if self.fine_align_prep_trajectory is not None:
            return

        if self.joint_feedback_ready():
            start = [
                self.latest_joint_positions[1],
                self.latest_joint_positions[2],
                self.latest_joint_positions[3],
            ]
        else:
            search_pose = self.search_pose(
                self.pose_planner.base_deg_from_theta(self.theta)
            )
            start = [search_pose[1], search_pose[2], search_pose[3]]

        end_j234 = self.fine_align_prep_end_j234()
        self.fine_align_prep_trajectory = self.fine_align_prep_planner.start(
            start,
            time.monotonic(),
            end_j234=end_j234,
        )
        self.prep_trajectory_last_j234 = list(start)
        self.get_logger().info(
            "Started FINE_ALIGN_PREP trajectory: "
            f"start_j234_deg={[round(math.degrees(value), 1) for value in start]}, "
            f"end_j234_deg={[round(math.degrees(value), 1) for value in end_j234]}"
        )

    def fine_align_prep_trajectory_duration(self):
        return self.fine_align_prep_planner.duration_sec()

    def fine_align_prep_trajectory_progress(self):
        if self.fine_align_prep_trajectory is None:
            return 0.0

        return self.fine_align_prep_planner.progress(
            self.fine_align_prep_trajectory,
            time.monotonic(),
        )

    def fine_align_prep_trajectory_pose(self, base_deg):
        self.start_fine_align_prep_trajectory()
        self.prep_trajectory_last_j234 = self.fine_align_prep_planner.sample_j234(
            self.fine_align_prep_trajectory,
            time.monotonic(),
        )
        self.approach_joint2_deg = math.degrees(self.prep_trajectory_last_j234[0])
        self.approach_joint3_deg = math.degrees(self.prep_trajectory_last_j234[1])
        return self.pose_planner.fine_align_prep_pose(
            base_deg,
            self.prep_trajectory_last_j234,
        )

    def fine_align_prep_trajectory_pose_ready(self):
        if (
            self.fine_align_prep_trajectory is None
            or not self.joint_feedback_ready()
        ):
            return False

        return self.fine_align_prep_planner.ready(
            self.fine_align_prep_trajectory,
            self.latest_joint_positions,
            self.joint_goal_tolerance,
            time.monotonic(),
        )

    def cylindrical_servo_pose(self):
        if self.servo_radius_m is None or self.servo_z_m is None:
            self.capture_servo_pose()

        plan = self.pose_planner.cylindrical_servo_pose(
            self.servo_radius_m,
            self.theta,
            self.servo_z_m,
        )

        if plan.positions is None:
            self.warn_throttled(plan.warning_key, plan.warning_text)
            return None

        self.approach_joint2_deg = plan.joint2_deg
        self.approach_joint3_deg = plan.joint3_deg

        return plan.positions

    def compute_joint_positions(self, theta_rad, status):
        base_deg = self.pose_planner.base_deg_from_theta(theta_rad)

        if not self.pose_planner.base_in_range(base_deg):
            self.warn_throttled("base_range", f"Base out of range: {base_deg:.1f} deg")
            return None

        if status in SEARCH_STATUSES:
            if status == "SEARCH_PREP":
                return self.search_prep_pose()
            return self.search_pose(base_deg)

        if status == "TRACKING_CENTER":
            return self.search_pose(base_deg)

        if status == "FINE_ALIGN_PREP":
            return self.fine_align_prep_trajectory_pose(base_deg)

        return self.cylindrical_servo_pose()

    def smooth_detection(self, detection):
        detection = self.center_smoother.smooth(detection)
        self.smoothed_center_x = self.center_smoother.smoothed_center_x
        return detection

    def tracking_errors(self, detection, image_target_x):
        return self.pan_servo.tracking_errors(detection, image_target_x)

    def pan_centered(self, error_x, tolerance=None):
        return self.pan_servo.pan_centered(error_x, tolerance)

    def capture_servo_pose(self):
        pose = self.current_cylindrical_estimate()
        self.servo_radius_m = pose.radius
        self.servo_z_m = pose.z
        end_j234 = self.fine_align_prep_end_j234()
        self.approach_joint2_deg = math.degrees(
            self.current_joint_or_deg(2, math.degrees(end_j234[0]))
        )
        self.approach_joint3_deg = math.degrees(
            self.current_joint_or_deg(3, math.degrees(end_j234[1]))
        )

    def step_cylindrical_approach(self, bbox_error):
        if self.servo_radius_m is None or self.servo_z_m is None:
            self.capture_servo_pose()

        step = self.approach_servo.step_cylindrical(
            self.servo_radius_m,
            self.theta,
            self.servo_z_m,
            self.kinematics,
            bbox_error,
        )

        if not step.success:
            self.warn_throttled(step.warning_key, step.warning_text)
            return False

        self.servo_radius_m = step.radius_m
        self.approach_joint2_deg = step.joint2_deg
        self.approach_joint3_deg = step.joint3_deg
        return True

    def reset_frame_debug(self, detection):
        self.frame_detection = detection
        self.frame_error_x = None
        self.frame_bbox_error = None
        self.frame_bbox_w = 0.0

    def set_tracking_debug(self, detection, image_target_x):
        detection = self.smooth_detection(detection)
        self.frame_detection = detection
        (
            self.frame_error_x,
            self.frame_bbox_error,
            self.frame_bbox_w,
        ) = self.tracking_errors(detection, image_target_x)

    def make_debug_text(self, status, error_x, bbox_error, bbox_w):
        if status in ("SEARCH_PREP", "SEARCHING", "STARTING"):
            joint4_target_rad = self.desired_search_joint4_rad()
        else:
            joint4_target_rad = self.desired_wrist_pitch_rad(status)

        return format_debug_text(
            DebugTelemetry(
                status=status,
                radius_now=self.current_radius_estimate(),
                theta_rad=self.theta,
                z_now=self.current_z_estimate(),
                approach_servo_started=self.approach_servo_started,
                approach_locked=self.approach_locked,
                tracking_settle_tolerance_px=self.tracking_settle_tolerance_px,
                tracking_centered_count=self.tracking_centered_count,
                tracking_settle_cycles=self.tracking_settle_cycles,
                fine_align_centered_count=self.fine_align_centered_count,
                fine_align_centered_cycles=self.fine_align_centered_cycles,
                servo_radius_m=self.servo_radius_m,
                servo_z_m=self.servo_z_m,
                approach_joint2_deg=self.approach_joint2_deg,
                approach_joint3_deg=self.approach_joint3_deg,
                joint4_target_rad=joint4_target_rad,
                prep_progress=self.fine_align_prep_trajectory_progress(),
                error_x=error_x,
                bbox_w=bbox_w,
                bbox_error=bbox_error,
                target_bbox_width=self.target_bbox_width,
            )
        )

    def draw_overlay(
        self,
        frame,
        detection,
        status,
        target_x,
        error_x=None,
        bbox_error=None,
        bbox_w=None,
    ):
        draw_debug_overlay(
            frame,
            detection,
            status,
            target_x,
            self.target_bbox_width,
            self.current_radius_estimate(),
            self.theta,
            self.current_z_estimate(),
            error_x=error_x,
            bbox_error=bbox_error,
            bbox_w=bbox_w,
        )

    def show_debug_windows(self, frame, mask):
        if not self.show_window or self.window_failed:
            return

        try:
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.show_window = False
                cv2.destroyAllWindows()
        except cv2.error as exc:
            self.window_failed = True
            self.get_logger().warn(f"OpenCV window disabled: {exc}")

    def process_frame(self, frame):
        width = frame.shape[1]
        image_target_x = width * self.target_center_x_ratio

        detection, mask = self.detectors.detect(frame, mode=self.stage)
        self.reset_frame_debug(detection)
        observation = self.runtime.make_observation(detection, image_target_x)
        step = self.controller.update_state(
            self.runtime.capture_state(),
            observation,
        )
        self.runtime.execute_step(step)
        return step, detection, mask, image_target_x

    def publish_frame_result(
        self,
        frame,
        detection,
        mask,
        step,
        image_target_x,
    ):
        status = step.status

        if not step.publish_debug:
            self.io.publish_status(status)
            self.draw_overlay(frame, None, status, image_target_x)
            self.io.publish_debug_image(frame)
            self.show_debug_windows(frame, mask)
            return

        if step.publish_target:
            self.publish_target(status)

        self.io.publish_status(status)
        if step.release_camera:
            self.io.release_camera()

        self.io.publish_debug(
            self.make_debug_text(
                status,
                self.frame_error_x,
                self.frame_bbox_error,
                self.frame_bbox_w,
            )
        )

        if status != self.last_status:
            self.get_logger().info(f"status: {self.last_status} -> {status}")

        self.last_status = status

        self.draw_overlay(
            frame,
            self.frame_detection,
            status,
            image_target_x,
            self.frame_error_x,
            self.frame_bbox_error,
            self.frame_bbox_w,
        )
        self.io.publish_debug_image(frame)
        self.show_debug_windows(frame, mask)

    def loop(self):
        if self.stage in ("IDLE", "CANCELED", "CAMERA_ERROR", "YOLO_ERROR"):
            self.io.publish_status(self.stage)
            return

        if self.stage == "FINISHED":
            self.io.publish_status("FINISHED")
            self.io.release_camera()
            return

        if not self.io.camera_ok:
            return

        ret, frame = self.io.read_frame()

        if not ret:
            self.stage = "CAMERA_ERROR"
            self.state = self.runtime.capture_state()
            self.io.publish_status("CAMERA_ERROR")
            self.warn_throttled("camera_read", "Cannot read frame", 2.0)
            return

        step, detection, mask, image_target_x = self.process_frame(frame)
        self.publish_frame_result(frame, detection, mask, step, image_target_x)

    def destroy_node(self):
        self.io.release_camera()

        if self.show_window:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GraspCupNode()

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

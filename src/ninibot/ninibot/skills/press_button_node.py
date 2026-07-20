# flake8: noqa
import math
import time

import cv2
import numpy as np

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import tf2_ros

from vision_controller.aruco_detection import load_camera_info
from vision_controller.aruco_detection import make_detector
from vision_controller.aruco_detection import normalize_camera_source
from ninibot.task_params import PRESS_BUTTON_PARAMS
from ninibot.skills.press_button.joint_ramp import ramp_joint_goal as ramp_joint_goal_step
from ninibot.skills.press_button.camera_io import PressButtonCameraIo
from ninibot.skills.press_button import debug_view
from ninibot.skills.press_button.controller import CMD_DETECT_MARKER
from ninibot.skills.press_button.controller import CMD_FINISH_FINE_ALIGN_IK_WAIT
from ninibot.skills.press_button.controller import CMD_FINISH_FINE_ALIGN_SAMPLE
from ninibot.skills.press_button.controller import CMD_INCREMENT_PRESS_CYCLE
from ninibot.skills.press_button.controller import CMD_INCREMENT_SETTLE
from ninibot.skills.press_button.controller import CMD_LOG_DONE_SUMMARY
from ninibot.skills.press_button.controller import CMD_LOG_PROGRESS
from ninibot.skills.press_button.controller import CMD_PUBLISH_TARGET
from ninibot.skills.press_button.controller import CMD_RESET_LOCK_WAIT_TIMER
from ninibot.skills.press_button.controller import CMD_RESET_SETTLE
from ninibot.skills.press_button.controller import CMD_SET_CURRENT_TARGET
from ninibot.skills.press_button.controller import CMD_SET_PHASE
from ninibot.skills.press_button.controller import CMD_SHOW_LOCKED_MARKER_DEBUG
from ninibot.skills.press_button.controller import CMD_START_FINE_ALIGN_SAMPLE
from ninibot.skills.press_button.controller import CMD_START_LOCK_WAIT
from ninibot.skills.press_button.controller import CMD_START_MOVE_TO_MARKER
from ninibot.skills.press_button.controller import CMD_START_PRESS_LIFT_OR_FAIL
from ninibot.skills.press_button.controller import CMD_START_PRESS_NUDGE_OR_FAIL
from ninibot.skills.press_button.controller import CMD_START_PRESS_REPEAT_WAIT
from ninibot.skills.press_button.controller import CMD_START_PRESS_WAIT
from ninibot.skills.press_button.controller import CMD_START_SEARCH
from ninibot.skills.press_button.controller import PressButtonController
from ninibot.skills.press_button.controller import PressButtonObservation
from ninibot.skills.press_button.debug_renderer import DebugRenderer
from ninibot.skills.press_button.fine_align import FineAlignCommandAhead
from ninibot.skills.press_button.fine_align_controller import FineAlignController
from ninibot.skills.press_button.ik_service_client import FineAlignIkServiceClient
from ninibot.skills.press_button.marker_tracker import MarkerTracker
from ninibot.skills.press_button.planner_factory import create_local_ik
from ninibot.skills.press_button.planner_factory import create_prep_planner
from ninibot.skills.press_button.planner_factory import create_press_motion_planner
from ninibot.skills.press_button.planner_factory import create_search_scanner
from ninibot.skills.press_button.prep_trajectory import joint_pitch_deg as prep_joint_pitch_deg
from ninibot.skills.press_button.press_sequence import PressSequence
from ninibot.skills.press_button.ros_params import load_press_button_config
from ninibot.skills.press_button.search_scan import SearchScanState
from ninibot.skills.press_button.state_machine import Phase
from ninibot.skills.press_button.state_machine import TERMINAL_PHASES
from ninibot.skills.press_button.target_planner import joint4_rad_from_joint23 as planned_joint4_rad_from_joint23
from ninibot.skills.press_button import tf_helpers


class PressButtonNode(Node):
    PARAMETER_DEFAULTS = PRESS_BUTTON_PARAMS

    def __init__(self):
        super().__init__("press_button_node")
        self._initialize_pre_config_state()
        self._declare_ros_parameters()
        self._load_config()
        if not self._setup_ik_and_planners():
            return
        if not self._setup_camera_and_marker_tracker():
            return
        self._setup_ros_interfaces()
        self._setup_tf()
        self._setup_runtime_state()
        self._setup_phase_handlers()
        self._start_timer_and_autostart()
        self._log_startup_summary()

    def _initialize_pre_config_state(self):
        self.camera_ok = False
        self.debug_window = False
        self.debug_window_failed = False
        self.cap = None

    def _declare_ros_parameters(self):
        self.declare_parameters(
            namespace="",
            parameters=list(self.PARAMETER_DEFAULTS.items()),
        )

    def _load_config(self):
        self.config = load_press_button_config(
            self,
            normalize_camera_source,
            self.resolve_urdf_path,
        )
        self.config.apply_to(self)

    def _setup_ik_and_planners(self):
        try:
            self.ik_planner = create_local_ik(
                self.config,
                log_progress=self.log_progress,
                warn=self.get_logger().warn,
            )
            self.ik_lower_bounds = self.ik_planner.lower_bounds
            self.ik_upper_bounds = self.ik_planner.upper_bounds
            self.joint4_lower = self.ik_planner.joint4_lower
            self.joint4_upper = self.ik_planner.joint4_upper
            self.prep_planner = create_prep_planner(
                self.config,
                self.joint4_lower,
                self.joint4_upper,
            )
            self.search_scanner = create_search_scanner(self.config)
            self.press_motion_planner = create_press_motion_planner(
                self.config,
                self.ik_lower_bounds,
                self.ik_upper_bounds,
                self.joint4_lower,
                self.joint4_upper,
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to load URDF IK model: {exc}")
            return False
        return True

    def _setup_camera_and_marker_tracker(self):
        try:
            (
                calibration_width,
                calibration_height,
                self.camera_matrix,
                self.dist_coeffs,
            ) = load_camera_info(self.calibration_path)
        except Exception as exc:
            self.get_logger().error(f"Failed to load camera calibration: {exc}")
            self.camera_ok = False
            return False

        self.capture_width = self.frame_width or calibration_width
        self.capture_height = self.frame_height or calibration_height
        self.detect_markers = make_detector(self.aruco_dictionary)
        self.marker_tracker = MarkerTracker(
            self.target_marker_id,
            self.marker_length,
            self.camera_matrix,
            self.dist_coeffs,
            self.target_marker_offset_xyz,
        )
        self.camera_io = PressButtonCameraIo(self)
        return True

    def _setup_ros_interfaces(self):
        self.target_pub = self.create_publisher(JointState, self.target_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.debug_image_pub = None
        if self.publish_debug_image_enabled:
            self.debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 10)
        self.real_joint_sub = self.create_subscription(
            JointState,
            self.real_joint_states_topic,
            self.real_joint_callback,
            10,
        )
        self.command_sub = self.create_subscription(
            String,
            self.command_topic,
            self.command_callback,
            10,
        )
        ik_service_client = None
        if self.fine_align_use_ik_service:
            ik_service_client = FineAlignIkServiceClient.create_ros_client(
                self,
                self.fine_align_ik_service_name,
            )
        self.ik_service = FineAlignIkServiceClient(
            ik_service_client,
            enabled=self.fine_align_use_ik_service,
            service_name=self.fine_align_ik_service_name,
            timeout_sec=self.fine_align_ik_service_timeout_sec,
            fallback_enabled=self.fine_align_ik_service_fallback,
            joint6_rad=self.joint6_rad,
            log_progress=self.log_progress,
        )

    def _setup_tf(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def _setup_runtime_state(self):
        self.joint_names = ["1", "2", "3", "4", "6"]
        self.latest_joint_positions = None
        self.fine_align_command_planner = FineAlignCommandAhead(
            self.fine_align_integral_enable,
            self.fine_align_max_command_ahead,
        )
        self.reset_runtime_state_for_new_run()
        self.phase = Phase.IDLE
        self.phase_start_time = time.monotonic()
        self.last_log_time = 0.0
        self.camera_ok = True
        self.last_status = None

    def reset_runtime_state_for_new_run(self):
        self.current_target = None
        self.marker_camera_xyz = None
        self.marker_base_xyz = None
        self.marker_target_base_xyz = None
        self.gripper_base_xyz = None
        self.locked_marker_camera_xyz = None
        self.locked_marker_base_xyz = None
        self.locked_marker_target_base_xyz = None
        self.search_pan_deg = self.prep_joint1_deg
        self.search_direction = 1.0
        self.last_search_update_time = None
        self.move_goal = None
        self.press_base_goal = None
        self.press_goal = None
        self.lift_goal = None
        self.move_marker_base_xyz = None
        self.move_target_base_xyz = None
        self.last_move_goal_update_time = 0.0
        self.fine_align_goal = None
        self.fine_align_target_base_xyz = None
        self.fine_align_cycle_index = 0
        self.fine_align_samples = []
        self.fine_align_sample_start_time = None
        self.ik_service.reset()
        self.fine_align_command_planner.reset()
        self.fine_align_command_ahead = 0.0
        self.fine_align_command_direction = None
        self.prep_trajectory = None
        self.prep_trajectory_start = None
        self.prep_trajectory_goal = None
        self.prep_trajectory_started_at = 0.0
        self.press_wait_start_time = None
        self.press_repeat_wait_start_time = None
        self.press_cycle_index = 0
        self.done_summary_logged = False
        self.last_key = -1
        self.continue_requested = False
        self.lock_wait_start_time = None
        self.settle_count = 0
        self.ik_planner.reset_last_result()

    def _setup_phase_handlers(self):
        self.debug_renderer = DebugRenderer(self)
        self.fine_align_controller = FineAlignController(self)
        self.press_sequence = PressSequence(self)
        self.controller = PressButtonController()

    def _start_timer_and_autostart(self):
        self.timer = self.create_timer(1.0 / self.control_rate, self.loop)
        self.publish_status(Phase.IDLE)
        if self.auto_start:
            self.start_test()

    def _log_startup_summary(self):
        self.get_logger().info(
            "press_button_node started; "
            f"prep_j234=[{self.prep_joint2_deg:.1f}, "
            f"{self.prep_joint3_deg:.1f}, {self.prep_joint4_deg:.1f}] deg, "
            f"prep_trajectory={self.prep_trajectory_duration_sec:.1f}s, "
            f"prep_j2_transition={self.prep_j2_transition_sec:.1f}s, "
            f"search_pan=[{self.search_pan_min_deg:.1f}, {self.search_pan_max_deg:.1f}] deg, "
            f"move_speed_scale={self.move_to_marker_speed_scale:.2f}, "
            f"align_wrist_pitch={self.align_wrist_pitch_deg:.1f} deg, "
            f"ik_joint4=[{math.degrees(self.ik_lower_bounds[3]):.1f}, "
            f"{math.degrees(self.ik_upper_bounds[3]):.1f}] deg, "
            f"track_marker_during_move={self.track_marker_during_move}, "
            f"move_goal_update_period={self.move_goal_update_period_sec:.2f}s, "
            f"fine_align_enable={self.fine_align_enable}, "
            f"fine_align_start={self.fine_align_start_distance * 100.0:.1f}cm, "
            f"fine_align_tol={self.fine_align_tolerance * 100.0:.1f}cm, "
            f"fine_align_max_step={self.fine_align_max_step * 100.0:.1f}cm, "
            f"fine_align_speed_scale={self.fine_align_speed_scale:.2f}, "
            f"fine_align_max_joint_step={math.degrees(self.fine_align_max_joint_step_rad):.1f}deg, "
            f"fine_align_lock_joint4={self.fine_align_lock_joint4}, "
            f"fine_align_use_ik_service={self.fine_align_use_ik_service}, "
            f"fine_align_integral={self.fine_align_integral_enable}, "
            f"fine_align_max_command_ahead={self.fine_align_max_command_ahead * 100.0:.1f}cm, "
            f"fine_align_move_timeout={self.fine_align_move_timeout_sec:.1f}s, "
            f"fine_align_joint_goal_tol={math.degrees(self.fine_align_joint_goal_tolerance):.2f}deg, "
            f"target_marker_offset_cm={self.format_cm(self.target_marker_offset_xyz)}, "
            f"press_wait={self.press_wait_sec:.1f}s, "
            f"press_joint2_delta={self.press_joint2_delta_deg:+.1f} deg, "
            f"press_joint3_delta={self.press_joint3_delta_deg:+.1f} deg, "
            f"press_joint4_delta={self.press_joint4_delta_deg:+.1f} deg, "
            "press_lift_delta_is_negative_press_delta=True, "
            f"press_motion_speed_scale={self.press_motion_speed_scale:.2f}, "
            f"press_motion_settle_cycles={self.press_motion_settle_cycles}, "
            f"press_repeat_wait={self.press_repeat_wait_sec:.1f}s, "
            f"press_cycles={self.press_cycles}, "
            f"joint6={self.joint6_percent:.1f}% ({self.joint6_rad:.3f} rad), "
            f"TF={self.base_frame}<-{self.camera_frame}, "
            f"gripper_frame={self.gripper_frame}, auto_start={self.auto_start}, "
            f"debug_window={self.debug_window}, "
            f"lock_wait={self.lock_wait_sec:.1f}s"
        )

    def publish_status(self, status):
        if status == self.last_status:
            return
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.last_status = status

    def set_phase(self, phase, log_text=None):
        self.phase = phase
        self.phase_start_time = time.monotonic()
        self.publish_status(phase)
        if log_text:
            self.get_logger().info(log_text)

    def phase_elapsed_sec(self):
        if not hasattr(self, "phase_start_time"):
            return 0.0
        return time.monotonic() - self.phase_start_time

    def command_callback(self, msg):
        command = msg.data.strip().lower()
        if command in ("start", "press", "press_button"):
            self.start_test()
        elif command in ("continue", "move", "space", "go"):
            self.request_locked_marker_move("command")
        elif command in ("stop", "cancel"):
            self.current_target = None
            self.set_phase(Phase.IDLE, "press button: IDLE")
        else:
            self.get_logger().warn(f"Unknown press button command: {msg.data!r}")

    def start_test(self):
        if not self.camera_ok:
            self.publish_status(Phase.FAILED)
            return
        if self.phase not in TERMINAL_PHASES:
            self.get_logger().info(f"press button: already running phase={self.phase}")
            return
        self.reset_runtime_state_for_new_run()
        self.set_phase(
            Phase.PREP,
            "press button: PREP scheme=B trajectory to search pose",
        )

    def open_camera(self):
        return self.camera_io.open_camera()

    def release_camera(self):
        self.camera_io.release_camera()

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

    def log_progress(self, text, period_sec=1.0):
        now = time.monotonic()
        if now - self.last_log_time >= period_sec:
            self.get_logger().info(text)
            self.last_log_time = now

    def request_locked_marker_move(self, source="command"):
        if self.phase != Phase.LOCK_WAIT:
            self.get_logger().info(
                f"press button: continue from {source} ignored because phase={self.phase}"
            )
            return
        self.continue_requested = True
        self.get_logger().info(f"press button: LOCK_WAIT continue requested from {source}")

    def resolve_urdf_path(self, urdf_path):
        if urdf_path:
            return urdf_path
        share_dir = get_package_share_directory("arm_controller")
        return f"{share_dir}/urdf/so101.urdf"

    def make_joint_pose(self, joint1_deg, joint2_deg, joint3_deg, joint4_deg):
        return [
            math.radians(joint1_deg),
            math.radians(joint2_deg),
            math.radians(joint3_deg),
            math.radians(joint4_deg),
            self.joint6_rad,
        ]

    def prep_pan_deg_from_current(self):
        pan = self.prep_planner.pan_from_current(self.latest_joint_positions)
        if pan is None:
            return None
        if pan.clamped_from_deg is not None:
            self.get_logger().warn(
                "press button: PREP pan clamped "
                f"from {pan.clamped_from_deg:.1f}deg to {pan.value_deg:.1f}deg"
            )
        return pan.value_deg

    def make_prep_pose_from_current_pan(self):
        self.prep_pan_deg_from_current()
        return self.prep_planner.goal_from_current_pan(self.latest_joint_positions)

    def current_prep_start_pose(self):
        return self.prep_planner.current_start_pose(self.latest_joint_positions)

    def joint_pitch_deg(self, pose):
        return prep_joint_pitch_deg(pose)

    def ensure_prep_trajectory(self):
        if self.prep_trajectory is not None:
            return True

        self.prep_pan_deg_from_current()
        trajectory = self.prep_planner.make_trajectory(self.latest_joint_positions)
        if trajectory is None:
            self.log_progress(
                "press button: PREP waiting for first /real_joint_states"
            )
            return False

        self.prep_trajectory = trajectory
        self.prep_trajectory_start = trajectory.start
        self.prep_trajectory_goal = trajectory.goal
        self.prep_trajectory_started_at = time.monotonic()
        self.current_target = list(trajectory.start)
        duration_sec = trajectory.duration_sec
        transition_sec = trajectory.j2_transition_sec
        self.get_logger().info(
            "press button: PREP trajectory scheme=B "
            f"duration={duration_sec:.2f}s "
            f"pitch={self.joint_pitch_deg(trajectory.start):+.1f}->"
            f"{self.joint_pitch_deg(trajectory.goal):+.1f}deg "
            f"j2_hold={max(0.0, duration_sec - transition_sec):.2f}s "
            f"j2_transition={transition_sec:.2f}s "
            f"goal={self.format_joint_target_deg(trajectory.goal)}"
        )
        return True

    def prep_trajectory_elapsed_sec(self):
        return time.monotonic() - self.prep_trajectory_started_at

    def sample_prep_trajectory(self):
        if self.prep_trajectory is None:
            return None
        return self.prep_planner.sample(
            self.prep_trajectory,
            self.prep_trajectory_elapsed_sec(),
        )

    def prep_trajectory_ready(self):
        if self.prep_trajectory_goal is None:
            return False
        if self.prep_trajectory_elapsed_sec() < self.prep_trajectory.duration_sec:
            return False
        self.current_target = list(self.prep_trajectory_goal)
        return self.at_target()

    def horizontal_plus_offset_joint4_deg(self, joint2_deg, joint3_deg):
        return -joint2_deg - joint3_deg + self.align_wrist_pitch_deg

    def joint4_rad_from_joint23(self, joint2_rad, joint3_rad):
        return planned_joint4_rad_from_joint23(
            joint2_rad,
            joint3_rad,
            self.align_wrist_pitch_deg,
        )

    def apply_realtime_joint4_target(self):
        if self.current_target is None or len(self.current_target) < 4:
            return False

        joint4_rad = self.joint4_rad_from_joint23(
            self.current_target[1],
            self.current_target[2],
        )
        if joint4_rad < self.joint4_lower[0] or joint4_rad > self.joint4_upper[0]:
            self.log_progress(
                f"press button: realtime joint4 out of range: {math.degrees(joint4_rad):.1f} deg"
            )
            return False

        self.current_target[3] = joint4_rad
        return True

    def ik_initial_guess(self):
        return self.ik_planner.initial_guess(
            self.latest_joint_positions,
            self.current_target,
        )

    def ik_seed_guesses(self, q0):
        return self.ik_planner.seed_guesses(q0)

    def is_joint4_in_range(self, joint4):
        return self.ik_planner.is_joint4_in_range(joint4)

    def format_degrees(self, values):
        return debug_view.format_degrees(values)

    def format_cm_delta(self, values):
        return debug_view.format_cm_delta(values)

    def format_bound_flags(self, q):
        return self.ik_planner.format_bound_flags(q)

    def solve_grippertip_ik(self, target_xyz):
        return self.ik_planner.solve_grippertip_ik(
            target_xyz,
            self.latest_joint_positions,
            self.current_target,
        )

    def make_target_pose(self, target_base_xyz):
        return self.ik_planner.make_target_pose(
            target_base_xyz,
            self.latest_joint_positions,
            self.current_target,
        )

    def local_ik_reference(self):
        return self.ik_planner.local_reference(
            self.latest_joint_positions,
            self.current_target,
        )

    def fk_grippertip_xyz_from_q1234(self, q1234):
        return self.ik_planner.forward_grippertip_xyz(q1234)

    def make_fine_align_pose(self, target_base_xyz):
        return self.ik_planner.make_fine_align_pose(
            target_base_xyz,
            self.latest_joint_positions,
            self.current_target,
        )

    def fine_align_local_tolerance(self):
        return self.ik_planner.fine_align_local_tolerance()

    def fine_align_near_miss_tolerance(self):
        return self.ik_planner.fine_align_near_miss_tolerance()

    def make_fine_align_context(
        self,
        visual_step_target,
        ik_step_target,
        distance,
        max_step,
        command_ahead,
    ):
        return self.fine_align_controller.make_fine_align_context(
            visual_step_target,
            ik_step_target,
            distance,
            max_step,
            command_ahead,
        )

    def start_fine_align_move_from_pose(self, pose, context, source):
        return self.fine_align_controller.start_fine_align_move_from_pose(
            pose,
            context,
            source,
        )

    def fallback_fine_align_ik(self, context, reason):
        return self.fine_align_controller.fallback_fine_align_ik(context, reason)

    def start_fine_align_ik_service_request(self, context):
        return self.fine_align_controller.start_fine_align_ik_service_request(context)

    def finish_fine_align_ik_service_wait(self):
        return self.fine_align_controller.finish_fine_align_ik_service_wait()

    def at_target(self):
        return self.at_joint_pose(self.current_target)

    def at_joint_pose(self, target, tolerance=None):
        if target is None or self.latest_joint_positions is None:
            return False
        if tolerance is None:
            tolerance = self.joint_goal_tolerance
        for current, goal in zip(
            self.latest_joint_positions[:4],
            target[:4],
        ):
            if abs(current - goal) > tolerance:
                return False
        return True

    def start_search(self):
        state = self.search_scanner.initial_state(self.latest_joint_positions)
        self.search_pan_deg = state.pan_deg
        self.search_direction = state.direction
        self.last_search_update_time = time.monotonic()
        self.current_target = self.search_scanner.pose(self.search_pan_deg)
        self.set_phase(Phase.SEARCH, "press button: SEARCH")

    def update_search_target(self):
        now = time.monotonic()
        if self.last_search_update_time is None:
            self.last_search_update_time = now
        dt = max(0.0, now - self.last_search_update_time)
        self.last_search_update_time = now

        state = self.search_scanner.update(
            SearchScanState(self.search_pan_deg, self.search_direction),
            dt,
        )
        self.search_pan_deg = state.pan_deg
        self.search_direction = state.direction
        self.current_target = self.search_scanner.pose(self.search_pan_deg)

    def start_lock_wait(self, marker_base_xyz):
        if self.marker_camera_xyz is not None:
            self.locked_marker_camera_xyz = np.asarray(
                self.marker_camera_xyz,
                dtype=float,
            )
        self.locked_marker_base_xyz = np.asarray(marker_base_xyz, dtype=float)
        if self.marker_target_base_xyz is not None:
            self.locked_marker_target_base_xyz = np.asarray(
                self.marker_target_base_xyz,
                dtype=float,
            )
        self.marker_base_xyz = self.locked_marker_base_xyz
        self.last_key = -1
        self.continue_requested = False
        self.lock_wait_start_time = time.monotonic()
        self.set_phase(
            Phase.LOCK_WAIT,
            f"press button: LOCK_WAIT auto move in {self.lock_wait_sec:.1f}s; marker={self.format_cm(marker_base_xyz)}",
        )

    def start_move_to_marker(self, pose, marker_base_xyz, target_base_xyz=None):
        self.move_goal = list(pose)
        self.move_marker_base_xyz = np.asarray(marker_base_xyz, dtype=float)
        if target_base_xyz is not None:
            self.move_target_base_xyz = np.asarray(target_base_xyz, dtype=float)
        elif self.marker_target_base_xyz is not None:
            self.move_target_base_xyz = np.asarray(
                self.marker_target_base_xyz,
                dtype=float,
            )
        self.done_summary_logged = False
        self.fine_align_goal = None
        self.fine_align_target_base_xyz = None
        self.fine_align_cycle_index = 0
        self.fine_align_samples = []
        self.fine_align_sample_start_time = None
        self.ik_service.reset()
        self.fine_align_command_planner.reset()
        self.fine_align_command_ahead = 0.0
        self.fine_align_command_direction = None
        if self.latest_joint_positions is not None:
            self.current_target = list(self.latest_joint_positions[:4]) + [
                self.move_goal[4],
            ]
        elif self.current_target is None:
            self.current_target = list(self.move_goal)
        self.settle_count = 0
        self.set_phase(
            Phase.MOVE_TO_MARKER,
            f"press button: MOVE_TO_MARKER marker={self.format_cm(marker_base_xyz)}",
        )
        self.log_move_plan()
        self.last_move_goal_update_time = time.monotonic()

    def refresh_move_goal_from_marker(self):
        marker_base_xyz = self.detect_marker()
        if not self.track_marker_during_move:
            return False

        if marker_base_xyz is None:
            return False

        now = time.monotonic()
        if (
            self.last_move_goal_update_time > 0.0
            and now - self.last_move_goal_update_time < self.move_goal_update_period_sec
        ):
            return False
        self.last_move_goal_update_time = now

        if self.marker_target_base_xyz is None:
            return False

        pose = self.make_target_pose(self.marker_target_base_xyz)
        if pose is None:
            self.log_progress(
                "press button: MOVE_TO_MARKER keeping previous goal; "
                f"latest target unreachable target_base_cm={self.format_cm(self.marker_target_base_xyz)} "
                f"marker_base_cm={self.format_cm(marker_base_xyz)}"
            )
            return False

        old_goal = self.move_goal
        self.move_goal = list(pose)
        self.move_marker_base_xyz = np.asarray(marker_base_xyz, dtype=float)
        self.move_target_base_xyz = np.asarray(self.marker_target_base_xyz, dtype=float)
        if old_goal is not None:
            delta = np.asarray(self.move_goal[:4]) - np.asarray(old_goal[:4])
            self.log_progress(
                "press button: MOVE_TO_MARKER refreshed goal "
                f"marker={self.format_cm(marker_base_xyz)} "
                f"target={self.format_cm(self.move_target_base_xyz)} "
                f"dq1234_deg={self.format_degrees(delta)} "
                f"ik={self.format_ik_residual_log()}",
                period_sec=1.0,
            )
        return True

    def current_tip_to_target_distance(self):
        target = self.move_target_base_xyz
        if target is None:
            target = self.marker_target_base_xyz
        if target is None:
            return None
        tip = self.lookup_gripper_base_xyz()
        if tip is None:
            return None
        self.gripper_base_xyz = tip
        return float(np.linalg.norm(np.asarray(target, dtype=float) - tip))

    def should_start_fine_align(self):
        if not self.fine_align_enable:
            return False
        distance = self.current_tip_to_target_distance()
        if distance is None:
            return False
        if distance <= self.fine_align_start_distance:
            self.log_progress(
                "press button: entering FINE_ALIGN "
                f"tip-target={distance * 100.0:.1f}cm",
                period_sec=0.5,
            )
            return True
        return False

    def freeze_current_target(self):
        if self.latest_joint_positions is not None:
            self.current_target = list(self.latest_joint_positions[:4]) + [
                self.joint6_rad,
            ]
        elif self.current_target is None and self.move_goal is not None:
            self.current_target = list(self.move_goal)
        elif self.current_target is not None and len(self.current_target) > 4:
            self.current_target[4] = self.joint6_rad

    def start_fine_align_sample(self, reason=""):
        return self.fine_align_controller.start_fine_align_sample(reason)

    def collect_fine_align_sample(self):
        return self.fine_align_controller.collect_fine_align_sample()

    def fine_align_sample_ready(self):
        return self.fine_align_controller.fine_align_sample_ready()

    def make_fine_align_ik_step_target(self, fk_tip_xyz, step_direction, max_step):
        return self.fine_align_controller.make_fine_align_ik_step_target(
            fk_tip_xyz,
            step_direction,
            max_step,
        )

    def finish_fine_align_sample(self):
        return self.fine_align_controller.finish_fine_align_sample()

    def ramp_fine_align_target(self):
        return self.fine_align_controller.ramp_fine_align_target()

    def ramp_joint_goal(self, goal, joint_indices, speed_scale=None):
        if speed_scale is None:
            speed_scale = self.move_to_marker_speed_scale
        result = ramp_joint_goal_step(
            self.current_target,
            goal,
            joint_indices,
            self.control_rate,
            speed_scale,
        )
        if result.target is not None:
            self.current_target = result.target
        else:
            return False
        return result.done

    def ramp_move_target(self):
        done = self.ramp_joint_goal(self.move_goal, range(4))
        if self.current_target is None or self.move_goal is None:
            return False
        self.current_target[4] = self.move_goal[4]
        return done

    def start_press_wait(self):
        return self.press_sequence.start_press_wait()

    def press_wait_ready(self):
        return self.press_sequence.press_wait_ready()

    def start_press_nudge(self):
        return self.press_sequence.start_press_nudge()

    def ramp_press_target(self):
        return self.press_sequence.ramp_press_target()

    def start_press_lift(self):
        return self.press_sequence.start_press_lift()

    def ramp_lift_target(self):
        return self.press_sequence.ramp_lift_target()

    def start_press_repeat_wait(self):
        return self.press_sequence.start_press_repeat_wait()

    def press_repeat_wait_ready(self):
        return self.press_sequence.press_repeat_wait_ready()

    def consume_continue_input(self):
        if self.continue_requested:
            self.continue_requested = False
            return True
        if self.last_key in (10, 13, 32):
            self.last_key = -1
            return True
        return False

    def lock_wait_ready(self):
        if self.consume_continue_input():
            return True
        if self.lock_wait_start_time is None:
            return self.lock_wait_sec <= 0.0
        return (time.monotonic() - self.lock_wait_start_time) >= self.lock_wait_sec

    def publish_target(self):
        if self.current_target is None:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.current_target
        self.target_pub.publish(msg)

    def lookup_gripper_base_xyz(self):
        return tf_helpers.lookup_gripper_base_xyz(
            self.tf_buffer,
            self.base_frame,
            self.gripper_frame,
        )

    def lookup_gripper_camera_xyz(self):
        return tf_helpers.lookup_gripper_camera_xyz(
            self.tf_buffer,
            self.camera_frame,
            self.gripper_frame,
        )

    def publish_debug_frame(self, frame):
        return self.debug_renderer.publish_debug_frame(frame)

    def draw_text(self, frame, text, y, color=(0, 255, 255)):
        return self.debug_renderer.draw_text(frame, text, y, color)

    def format_cm(self, xyz):
        return debug_view.format_cm(xyz)

    def current_button_target_base_xyz(self):
        if self.move_target_base_xyz is not None:
            return self.move_target_base_xyz
        if self.marker_target_base_xyz is not None:
            return self.marker_target_base_xyz
        if self.locked_marker_target_base_xyz is not None:
            return self.locked_marker_target_base_xyz
        return None

    def format_tip_target_distance(self):
        target = self.current_button_target_base_xyz()
        if target is None or self.gripper_base_xyz is None:
            return "tip-target=(n/a)"
        delta = np.asarray(target) - np.asarray(self.gripper_base_xyz)
        distance_cm = float(np.linalg.norm(delta) * 100.0)
        return f"tip-target={distance_cm:.1f}cm"

    def distance_cm(self, a, b):
        return debug_view.distance_cm(a, b)

    def format_distance_cm(self, a, b):
        return debug_view.format_distance_cm(a, b)

    def format_joint_error_to_target_deg(self, target):
        if target is None or self.latest_joint_positions is None:
            return "dq_real_to_goal=(n/a)"
        errors = [
            math.degrees(goal - current)
            for current, goal in zip(self.latest_joint_positions[:4], target[:4])
        ]
        max_error = max(abs(value) for value in errors) if errors else 0.0
        values = ", ".join(f"{value:+.2f}" for value in errors)
        return f"dq_real_to_goal_deg=[{values}] max={max_error:.2f}"

    def format_joint_target_deg(self, target):
        return debug_view.format_joint_target_deg(target)

    def format_ik_residual_log(self):
        if self.ik_planner.last_error is None or self.ik_planner.last_residual is None:
            return "n/a"
        return (
            f"{self.ik_planner.last_error * 100.0:.1f}cm "
            f"residual_cm={self.format_cm_delta(self.ik_planner.last_residual)}"
        )

    def log_move_plan(self):
        self.get_logger().info(
            "press button result plan: "
            f"marker_base={self.format_cm(self.move_marker_base_xyz)} "
            f"marker_target_base={self.format_cm(self.move_target_base_xyz)} "
            f"target_base={self.format_cm(self.ik_planner.last_target_xyz)} "
            f"q1234_deg={self.format_joint_target_deg(self.move_goal)} "
            f"ik={self.format_ik_residual_log()}"
        )

    def log_done_summary(self):
        if self.done_summary_logged:
            return
        final_grippertip = self.lookup_gripper_base_xyz()
        self.gripper_base_xyz = final_grippertip
        self.get_logger().info(
            "press button result final: "
            f"marker_base={self.format_cm(self.move_marker_base_xyz)} "
            f"marker_target_base={self.format_cm(self.move_target_base_xyz)} "
            f"final_{self.gripper_frame}={self.format_cm(final_grippertip)} "
            f"tip_target_distance={self.format_distance_cm(self.move_target_base_xyz, final_grippertip)} "
            f"q1234_target_deg={self.format_joint_target_deg(self.move_goal)} "
            f"ik={self.format_ik_residual_log()}"
        )
        self.done_summary_logged = True

    def draw_marker_offset_target(self, frame, rvec, tvec):
        return self.debug_renderer.draw_marker_offset_target(frame, rvec, tvec)

    def draw_grippertip_camera_point(self, frame):
        return self.debug_renderer.draw_grippertip_camera_point(frame)

    def show_debug(self, frame, corners=None, ids=None, rvec=None, tvec=None, status=""):
        return self.debug_renderer.show_debug(
            frame,
            corners=corners,
            ids=ids,
            rvec=rvec,
            tvec=tvec,
            status=status,
        )

    def show_locked_marker_debug(self):
        self.camera_io.show_locked_marker_debug()

    def detect_marker(self):
        return self.camera_io.detect_marker()

    def build_controller_observation(self):
        kwargs = {
            "phase": self.phase,
            "camera_ok": self.camera_ok,
            "settle_count": self.settle_count,
            "settle_cycles": self.settle_cycles,
        }
        if not self.camera_ok or self.phase == Phase.IDLE:
            return PressButtonObservation(**kwargs)

        if self.phase == Phase.PREP:
            self._observe_prep(kwargs)
        elif self.phase == Phase.SEARCH:
            self._observe_search(kwargs)
        elif self.phase == Phase.LOCK_WAIT:
            self._observe_lock_wait(kwargs)
        elif self.phase == Phase.MOVE_TO_MARKER:
            self._observe_move_to_marker(kwargs)
        elif self.phase == Phase.FINE_ALIGN_SAMPLE:
            self._observe_fine_align_sample(kwargs)
        elif self.phase == Phase.FINE_ALIGN_MOVE:
            self._observe_fine_align_move(kwargs)
        elif self.phase == Phase.PRESS_WAIT:
            kwargs["press_wait_ready"] = self.press_wait_ready()
        elif self.phase == Phase.PRESS_NUDGE:
            kwargs.update(
                {
                    "press_nudge_ready": self.ramp_press_target(),
                    "press_motion_settle_cycles": self.press_motion_settle_cycles,
                }
            )
        elif self.phase == Phase.PRESS_LIFT:
            kwargs.update(
                {
                    "press_lift_ready": self.ramp_lift_target(),
                    "press_motion_settle_cycles": self.press_motion_settle_cycles,
                    "press_cycle_index": self.press_cycle_index,
                    "press_cycles": self.press_cycles,
                }
            )
        elif self.phase == Phase.PRESS_REPEAT_WAIT:
            kwargs["press_repeat_wait_ready"] = self.press_repeat_wait_ready()

        return PressButtonObservation(**kwargs)

    def _observe_prep(self, kwargs):
        if not self.ensure_prep_trajectory():
            return

        sampled_target = self.sample_prep_trajectory()
        kwargs["prep_sampled_target"] = sampled_target
        kwargs["prep_goal_target"] = self.prep_trajectory_goal
        if sampled_target is None or self.prep_trajectory_goal is None:
            return

        trajectory_ready = (
            self.prep_trajectory is not None
            and self.prep_trajectory_elapsed_sec()
            >= self.prep_trajectory.duration_sec
        )
        kwargs["prep_trajectory_ready"] = trajectory_ready
        kwargs["prep_at_goal"] = (
            self.at_joint_pose(self.prep_trajectory_goal)
            if trajectory_ready
            else False
        )

    def _observe_search(self, kwargs):
        self.update_search_target()
        kwargs["marker_base_xyz"] = self.detect_marker()

    def _observe_lock_wait(self, kwargs):
        lock_wait_ready = self.lock_wait_ready()
        marker_base_xyz = self.locked_marker_base_xyz
        target_base_xyz = self.locked_marker_target_base_xyz
        target_pose = None
        pose_failed = False
        if lock_wait_ready and target_base_xyz is not None:
            target_pose = self.make_target_pose(target_base_xyz)
            pose_failed = target_pose is None

        kwargs.update(
            {
                "lock_wait_ready": lock_wait_ready,
                "locked_marker_base_xyz": marker_base_xyz,
                "locked_marker_target_base_xyz": target_base_xyz,
                "lock_wait_target_pose": target_pose,
                "lock_wait_pose_failed": pose_failed,
            }
        )

    def _observe_move_to_marker(self, kwargs):
        self.refresh_move_goal_from_marker()
        should_start_fine_align = self.should_start_fine_align()
        move_ramp_ready = False
        move_at_goal = False
        if not should_start_fine_align:
            move_ramp_ready = self.ramp_move_target()
            move_at_goal = self.at_joint_pose(self.move_goal)

        kwargs.update(
            {
                "fine_align_enable": self.fine_align_enable,
                "should_start_fine_align": should_start_fine_align,
                "move_ramp_ready": move_ramp_ready,
                "move_at_goal": move_at_goal,
            }
        )

    def _observe_fine_align_sample(self, kwargs):
        self.collect_fine_align_sample()
        kwargs.update(
            {
                "fine_align_sample_count": len(self.fine_align_samples),
                "fine_align_marker_samples": self.fine_align_marker_samples,
                "fine_align_sample_ready": self.fine_align_sample_ready(),
            }
        )

    def _observe_fine_align_move(self, kwargs):
        align_ready = self.ramp_fine_align_target()
        timed_out = (
            not align_ready
            and self.fine_align_move_timeout_sec > 0.0
            and self.phase_elapsed_sec() >= self.fine_align_move_timeout_sec
        )
        timeout_text = (
            "press button: FAILED fine align command ramp not complete "
            f"after {self.phase_elapsed_sec():.1f}s "
            f"{self.format_joint_error_to_target_deg(self.fine_align_goal)}"
        )
        settle_text = (
            "press button: FINE_ALIGN_MOVE settling before visual resample "
            f"{self.settle_count + 1}/{self.fine_align_settle_cycles} "
            f"{self.format_joint_error_to_target_deg(self.fine_align_goal)}"
        )
        kwargs.update(
            {
                "fine_align_move_ready": align_ready,
                "fine_align_move_timed_out": timed_out,
                "fine_align_move_timeout_text": timeout_text,
                "fine_align_settle_cycles": self.fine_align_settle_cycles,
                "fine_align_move_settle_text": settle_text,
            }
        )

    def execute_controller_step(self, step):
        for command in step.commands:
            self.execute_controller_command(command)

    def execute_controller_command(self, command):
        payload = command.payload
        name = command.name

        if name == CMD_SET_CURRENT_TARGET:
            target = payload.get("target")
            self.current_target = None if target is None else list(target)
        elif name == CMD_PUBLISH_TARGET:
            self.publish_target()
        elif name == CMD_DETECT_MARKER:
            self.detect_marker()
        elif name == CMD_SHOW_LOCKED_MARKER_DEBUG:
            self.show_locked_marker_debug()
        elif name == CMD_LOG_PROGRESS:
            self.log_progress(
                self.format_controller_log_text(payload.get("text", "")),
                period_sec=float(payload.get("period_sec", 1.0)),
            )
        elif name == CMD_RESET_SETTLE:
            self.settle_count = 0
        elif name == CMD_INCREMENT_SETTLE:
            self.settle_count += 1
        elif name == CMD_RESET_LOCK_WAIT_TIMER:
            self.lock_wait_start_time = time.monotonic()
        elif name == CMD_START_SEARCH:
            self.start_search()
        elif name == CMD_START_LOCK_WAIT:
            self.start_lock_wait(payload["marker_base_xyz"])
        elif name == CMD_START_MOVE_TO_MARKER:
            self.start_move_to_marker(
                payload["pose"],
                payload["marker_base_xyz"],
                payload.get("target_base_xyz"),
            )
        elif name == CMD_START_FINE_ALIGN_SAMPLE:
            self.start_fine_align_sample(payload.get("reason", ""))
        elif name == CMD_FINISH_FINE_ALIGN_SAMPLE:
            self.finish_fine_align_sample()
        elif name == CMD_FINISH_FINE_ALIGN_IK_WAIT:
            self.finish_fine_align_ik_service_wait()
        elif name == CMD_START_PRESS_WAIT:
            self.start_press_wait()
        elif name == CMD_START_PRESS_NUDGE_OR_FAIL:
            if not self.start_press_nudge():
                reason = (
                    "press button: FAILED missing repeat target"
                    if payload.get("repeat")
                    else "press button: FAILED missing press target"
                )
                self.set_phase(Phase.FAILED, reason)
        elif name == CMD_START_PRESS_LIFT_OR_FAIL:
            if not self.start_press_lift():
                self.set_phase(Phase.FAILED, "press button: FAILED missing lift target")
        elif name == CMD_START_PRESS_REPEAT_WAIT:
            self.start_press_repeat_wait()
        elif name == CMD_INCREMENT_PRESS_CYCLE:
            self.press_cycle_index += 1
        elif name == CMD_LOG_DONE_SUMMARY:
            self.log_done_summary()
        elif name == CMD_SET_PHASE:
            phase = payload["phase"]
            log_text = payload.get("log_text")
            if phase == Phase.DONE and not log_text:
                log_text = (
                    f"press button: DONE {self.gripper_frame}_base_cm="
                    f"{self.format_cm(self.gripper_base_xyz)}"
                )
            self.set_phase(phase, log_text)
        else:
            self.get_logger().warn(f"Unknown press button command: {name}")

    def format_controller_log_text(self, text):
        values = {
            "lock_wait_sec": self.lock_wait_sec,
            "press_wait_sec": self.press_wait_sec,
            "press_repeat_wait_sec": self.press_repeat_wait_sec,
            "target_base_cm": self.format_cm(self.locked_marker_target_base_xyz),
            "marker_base_cm": self.format_cm(self.locked_marker_base_xyz),
        }
        try:
            return text.format(**values)
        except (KeyError, ValueError):
            return text

    def loop(self):
        observation = self.build_controller_observation()
        step = self.controller.step(observation)
        self.execute_controller_step(step)

    def destroy_node(self):
        self.release_camera()
        if self.debug_window and not self.debug_window_failed:
            try:
                cv2.destroyWindow(self.debug_window_name)
            except cv2.error:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PressButtonNode()
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

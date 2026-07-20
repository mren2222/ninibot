import math
import time

from ninibot.skills.grasp_cup.controller import GraspCupObservation
from ninibot.skills.grasp_cup.controller import GraspCupState


class GraspCupRuntimeAdapter:
    def __init__(self, host):
        self.host = host

    def reset_tracking_joint_degs(self):
        end_j234 = self.host.fine_align_prep_end_j234()
        return math.degrees(end_j234[0]), math.degrees(end_j234[1])

    def capture_state(self):
        host = self.host
        return GraspCupState(
            stage=host.stage,
            theta=host.theta,
            search_direction=host.search_direction,
            search_anchor_theta=host.search_anchor_theta,
            search_anchor_captured=host.search_anchor_captured,
            tracking_lost_started_at=host.tracking_lost_started_at,
            approach_servo_started=host.approach_servo_started,
            approach_locked=host.approach_locked,
            tracking_centered_count=host.tracking_centered_count,
            fine_align_centered_count=host.fine_align_centered_count,
            servo_radius_m=host.servo_radius_m,
            servo_z_m=host.servo_z_m,
            smoothed_center_x=host.smoothed_center_x,
            approach_joint2_deg=host.approach_joint2_deg,
            approach_joint3_deg=host.approach_joint3_deg,
            search_prep_trajectory=host.search_prep_trajectory,
            search_prep_started_at=host.search_prep_started_at,
        )

    def apply_state(self, state):
        host = self.host
        host.state = state
        host.stage = state.stage
        host.theta = state.theta
        host.search_direction = state.search_direction
        host.search_anchor_theta = state.search_anchor_theta
        host.search_anchor_captured = state.search_anchor_captured
        host.tracking_lost_started_at = state.tracking_lost_started_at
        host.approach_servo_started = state.approach_servo_started
        host.approach_locked = state.approach_locked
        host.tracking_centered_count = state.tracking_centered_count
        host.fine_align_centered_count = state.fine_align_centered_count
        host.servo_radius_m = state.servo_radius_m
        host.servo_z_m = state.servo_z_m
        host.smoothed_center_x = state.smoothed_center_x
        host.approach_joint2_deg = state.approach_joint2_deg
        host.approach_joint3_deg = state.approach_joint3_deg
        host.search_prep_trajectory = state.search_prep_trajectory
        host.search_prep_started_at = state.search_prep_started_at

    def reset_tracking_runtime(self):
        self.host.center_smoother.reset()
        self.host.reset_fine_align_prep_trajectory()

    def capture_fine_align_start_values(self):
        host = self.host
        pose = host.current_cylindrical_estimate()
        end_j234 = host.fine_align_prep_end_j234()
        return {
            "capture_servo_radius_m": pose.radius,
            "capture_servo_z_m": pose.z,
            "capture_joint2_deg": math.degrees(
                host.current_joint_or_deg(2, math.degrees(end_j234[0]))
            ),
            "capture_joint3_deg": math.degrees(
                host.current_joint_or_deg(3, math.degrees(end_j234[1]))
            ),
        }

    def make_observation(self, detection, image_target_x):
        host = self.host
        search_prep_ready = False
        fine_align_prep_ready = False
        capture_values = {}

        if host.stage == "SEARCH_PREP":
            search_prep_ready = host.search_prep_pose_ready()
        elif host.stage == "FINE_ALIGN_PREP":
            host.start_fine_align_prep_trajectory()
            fine_align_prep_ready = host.fine_align_prep_trajectory_pose_ready()
            if fine_align_prep_ready:
                capture_values = self.capture_fine_align_start_values()

        tracking_error_x = None
        bbox_error = None
        bbox_w = 0.0
        tracking_centered = False
        fine_align_centered = False
        tracking_theta_delta = 0.0
        fine_align_theta_delta = 0.0
        approach_locked_after_current = None
        approach_locked_after_reset = None

        if detection is not None and host.stage in ("TRACKING_CENTER", "FINE_ALIGN"):
            host.set_tracking_debug(detection, image_target_x)
            tracking_error_x = host.frame_error_x
            bbox_error = host.frame_bbox_error
            bbox_w = host.frame_bbox_w
            tracking_centered = host.pan_centered(tracking_error_x)
            fine_align_centered = host.pan_centered(tracking_error_x)
            tracking_theta_delta = host.pan_servo.tracking_theta_delta(
                tracking_error_x
            )
            fine_align_theta_delta = host.pan_servo.theta_delta(
                tracking_error_x,
                tolerance=host.center_tolerance_px,
            )
            approach_locked_after_current = host.approach_servo.update_lock(
                host.approach_locked,
                bbox_error,
            )
            approach_locked_after_reset = host.approach_servo.update_lock(
                False,
                bbox_error,
            )

        reset_joint2_deg, reset_joint3_deg = self.reset_tracking_joint_degs()
        return GraspCupObservation(
            detection_present=detection is not None,
            image_target_x=image_target_x,
            now=time.monotonic(),
            latest_joint_theta=host.latest_joint_theta,
            reset_joint2_deg=reset_joint2_deg,
            reset_joint3_deg=reset_joint3_deg,
            search_prep_ready=search_prep_ready,
            fine_align_prep_ready=fine_align_prep_ready,
            tracking_error_x=tracking_error_x,
            bbox_error=bbox_error,
            bbox_w=bbox_w,
            tracking_centered=tracking_centered,
            fine_align_centered=fine_align_centered,
            tracking_theta_delta=tracking_theta_delta,
            fine_align_theta_delta=fine_align_theta_delta,
            approach_locked_after_current=approach_locked_after_current,
            approach_locked_after_reset=approach_locked_after_reset,
            **capture_values,
        )

    def execute_step(self, step):
        host = self.host
        for warning in step.warnings:
            host.warn_throttled(
                warning.key,
                warning.text,
                warning.period_sec,
            )

        self.apply_state(step.state)

        if step.reset_tracking_runtime:
            self.reset_tracking_runtime()

        if step.step_approach and step.approach_bbox_error is not None:
            if host.approach_servo.should_step(
                host.approach_servo_started,
                host.approach_locked,
                step.approach_bbox_error,
            ):
                host.step_cylindrical_approach(step.approach_bbox_error)
                host.state = self.capture_state()

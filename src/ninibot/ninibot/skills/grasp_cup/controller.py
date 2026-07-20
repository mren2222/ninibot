from dataclasses import dataclass
from dataclasses import replace
from typing import Any


SEARCH_STATUSES = frozenset({"STARTING", "SEARCH_PREP", "SEARCHING", "LOST"})


@dataclass(frozen=True)
class GraspCupControllerConfig:
    search_enabled: bool
    search_step: float
    theta_min: float
    theta_max: float
    tracking_lost_grace_sec: float
    tracking_settle_cycles: int
    fine_align_centered_cycles: int
    publish_while_lost: bool


@dataclass(frozen=True)
class GraspCupState:
    stage: str = "SEARCH_PREP"
    theta: float = 0.0
    search_direction: float = 1.0
    search_anchor_theta: float = 0.0
    search_anchor_captured: bool = False
    tracking_lost_started_at: float | None = None
    approach_servo_started: bool = False
    approach_locked: bool = False
    tracking_centered_count: int = 0
    fine_align_centered_count: int = 0
    servo_radius_m: float | None = None
    servo_z_m: float | None = None
    smoothed_center_x: float | None = None
    approach_joint2_deg: float = 0.0
    approach_joint3_deg: float = 0.0
    search_prep_trajectory: Any = None
    search_prep_started_at: float = 0.0


@dataclass(frozen=True)
class GraspCupObservation:
    detection_present: bool
    image_target_x: float
    now: float
    latest_joint_theta: float | None = None
    reset_joint2_deg: float = 0.0
    reset_joint3_deg: float = 0.0
    search_prep_ready: bool = False
    fine_align_prep_ready: bool = False
    tracking_error_x: float | None = None
    bbox_error: float | None = None
    bbox_w: float = 0.0
    tracking_centered: bool = False
    fine_align_centered: bool = False
    tracking_theta_delta: float = 0.0
    fine_align_theta_delta: float = 0.0
    approach_locked_after_current: bool | None = None
    approach_locked_after_reset: bool | None = None
    capture_servo_radius_m: float | None = None
    capture_servo_z_m: float | None = None
    capture_joint2_deg: float | None = None
    capture_joint3_deg: float | None = None


@dataclass(frozen=True)
class GraspCupWarning:
    key: str
    text: str
    period_sec: float = 1.0


@dataclass(frozen=True)
class GraspCupStep:
    state: GraspCupState
    status: str
    publish_target: bool = True
    publish_debug: bool = True
    release_camera: bool = False
    reset_tracking_runtime: bool = False
    step_approach: bool = False
    approach_bbox_error: float | None = None
    warnings: tuple[GraspCupWarning, ...] = ()


class GraspCupController:
    def __init__(self, config: GraspCupControllerConfig):
        self.config = config

    def clamp_theta(self, theta):
        return max(self.config.theta_min, min(self.config.theta_max, theta))

    def reset_tracking_state(
        self,
        state,
        reset_joint2_deg,
        reset_joint3_deg,
    ):
        return replace(
            state,
            approach_servo_started=False,
            approach_locked=False,
            tracking_centered_count=0,
            fine_align_centered_count=0,
            servo_radius_m=None,
            servo_z_m=None,
            smoothed_center_x=None,
            tracking_lost_started_at=None,
            approach_joint2_deg=reset_joint2_deg,
            approach_joint3_deg=reset_joint3_deg,
        )

    def reset_to_search_state(self, state, observation):
        anchor_theta = self.clamp_theta(
            observation.latest_joint_theta
            if observation.latest_joint_theta is not None
            else state.theta
        )
        state = self.reset_tracking_state(
            state,
            observation.reset_joint2_deg,
            observation.reset_joint3_deg,
        )
        return replace(
            state,
            stage="SEARCH_PREP",
            theta=anchor_theta,
            search_direction=1.0,
            search_anchor_theta=anchor_theta,
            search_anchor_captured=observation.latest_joint_theta is not None,
            search_prep_trajectory=None,
            search_prep_started_at=0.0,
            tracking_lost_started_at=None,
        )

    def reset_to_search(self, state, observation, warnings=()):
        return self.make_step(
            self.reset_to_search_state(state, observation),
            observation,
            "SEARCH_PREP",
            reset_tracking_runtime=True,
            warnings=warnings,
        )

    def update_search(self, state):
        if not self.config.search_enabled:
            return state, "LOST"

        theta = state.theta + state.search_direction * self.config.search_step
        search_direction = state.search_direction

        if theta > self.config.theta_max:
            theta = self.config.theta_max
            search_direction = -1.0
        elif theta < self.config.theta_min:
            theta = self.config.theta_min
            search_direction = 1.0

        return (
            replace(
                state,
                theta=theta,
                search_direction=search_direction,
            ),
            "SEARCHING",
        )

    def hold_tracking_after_missing_detection(self, state, status, observation):
        if self.config.tracking_lost_grace_sec <= 0.0:
            return False, state, ()

        started_at = (
            observation.now
            if state.tracking_lost_started_at is None
            else state.tracking_lost_started_at
        )
        elapsed = observation.now - started_at
        state = replace(state, tracking_lost_started_at=started_at)

        if elapsed >= self.config.tracking_lost_grace_sec:
            return False, state, ()

        warning = GraspCupWarning(
            "tracking_lost_grace",
            f"cup temporarily lost in {status}; holding target "
            f"{elapsed:.2f}/{self.config.tracking_lost_grace_sec:.2f}s",
            0.5,
        )
        return True, state, (warning,)

    def handle_search_prep(self, state, observation):
        if not observation.search_prep_ready:
            state = replace(
                state,
                theta=self.clamp_theta(state.search_anchor_theta),
            )
            return self.make_step(state, observation, "SEARCH_PREP")

        state = replace(state, stage="SEARCHING")
        return self.make_step(state, observation, "SEARCHING")

    def handle_searching(self, state, observation):
        if not observation.detection_present:
            state, status = self.update_search(state)
            return self.make_step(state, observation, status)

        state = self.reset_tracking_state(
            replace(state, stage="TRACKING_CENTER"),
            observation.reset_joint2_deg,
            observation.reset_joint3_deg,
        )
        return self.make_step(
            state,
            observation,
            "TRACKING_CENTER",
            reset_tracking_runtime=True,
        )

    def handle_tracking_center(self, state, observation):
        if not observation.detection_present:
            held, state, warnings = self.hold_tracking_after_missing_detection(
                state,
                "TRACKING_CENTER",
                observation,
            )
            if held:
                return self.make_step(
                    state,
                    observation,
                    "TRACKING_CENTER",
                    warnings=warnings,
                )
            return self.reset_to_search(state, observation, warnings=warnings)

        if observation.tracking_centered:
            centered_count = state.tracking_centered_count + 1
            if centered_count >= self.config.tracking_settle_cycles:
                state = replace(
                    state,
                    stage="FINE_ALIGN_PREP",
                    tracking_lost_started_at=None,
                    tracking_centered_count=centered_count,
                )
                return self.make_step(state, observation, "FINE_ALIGN_PREP")

            state = replace(
                state,
                tracking_lost_started_at=None,
                tracking_centered_count=centered_count,
            )
            return self.make_step(state, observation, "TRACKING_CENTER")

        state = replace(
            state,
            tracking_lost_started_at=None,
            tracking_centered_count=0,
            theta=state.theta + observation.tracking_theta_delta,
        )
        return self.make_step(state, observation, "TRACKING_CENTER")

    def handle_fine_align_prep(self, state, observation):
        if not observation.fine_align_prep_ready:
            return self.make_step(state, observation, "FINE_ALIGN_PREP")

        state = replace(
            state,
            stage="FINE_ALIGN",
            fine_align_centered_count=0,
            servo_radius_m=(
                observation.capture_servo_radius_m
                if observation.capture_servo_radius_m is not None
                else state.servo_radius_m
            ),
            servo_z_m=(
                observation.capture_servo_z_m
                if observation.capture_servo_z_m is not None
                else state.servo_z_m
            ),
            approach_joint2_deg=(
                observation.capture_joint2_deg
                if observation.capture_joint2_deg is not None
                else state.approach_joint2_deg
            ),
            approach_joint3_deg=(
                observation.capture_joint3_deg
                if observation.capture_joint3_deg is not None
                else state.approach_joint3_deg
            ),
        )
        return self.make_step(state, observation, "FINE_ALIGN")

    def handle_fine_align(self, state, observation):
        if not observation.detection_present:
            held, state, warnings = self.hold_tracking_after_missing_detection(
                state,
                "FINE_ALIGN",
                observation,
            )
            if held:
                return self.make_step(
                    state,
                    observation,
                    "FINE_ALIGN",
                    warnings=warnings,
                )
            return self.reset_to_search(state, observation, warnings=warnings)

        fine_align_centered_count = state.fine_align_centered_count
        approach_servo_started = state.approach_servo_started
        approach_locked = state.approach_locked
        started_this_step = False

        if observation.fine_align_centered:
            fine_align_centered_count += 1
            if (
                not approach_servo_started
                and fine_align_centered_count
                >= self.config.fine_align_centered_cycles
            ):
                approach_servo_started = True
                approach_locked = False
                started_this_step = True
        else:
            fine_align_centered_count = 0

        theta = state.theta + observation.fine_align_theta_delta
        step_approach = False

        if approach_servo_started:
            if started_this_step:
                approach_locked = (
                    observation.approach_locked_after_reset
                    if observation.approach_locked_after_reset is not None
                    else approach_locked
                )
            else:
                approach_locked = (
                    observation.approach_locked_after_current
                    if observation.approach_locked_after_current is not None
                    else approach_locked
                )
            step_approach = True

        ready_to_grab = (
            approach_servo_started
            and observation.fine_align_centered
            and approach_locked
        )
        status = "FINISHED" if ready_to_grab else "FINE_ALIGN"
        state = replace(
            state,
            stage=status,
            tracking_lost_started_at=None,
            fine_align_centered_count=fine_align_centered_count,
            approach_servo_started=approach_servo_started,
            approach_locked=approach_locked,
            theta=theta,
        )
        return self.make_step(
            state,
            observation,
            status,
            step_approach=step_approach,
            approach_bbox_error=observation.bbox_error,
        )

    def update_state(self, state, observation):
        if state.stage == "SEARCH_PREP":
            return self.handle_search_prep(state, observation)

        if state.stage == "SEARCHING":
            return self.handle_searching(state, observation)

        if state.stage == "TRACKING_CENTER":
            return self.handle_tracking_center(state, observation)

        if state.stage == "FINE_ALIGN_PREP":
            return self.handle_fine_align_prep(state, observation)

        if state.stage == "FINE_ALIGN":
            return self.handle_fine_align(state, observation)

        if state.stage == "FINISHED":
            return self.make_step(
                state,
                observation,
                "FINISHED",
                release_camera=True,
            )

        warning = GraspCupWarning(
            "unknown_stage",
            f"Unknown cup vision stage: {state.stage}",
        )
        return self.reset_to_search(state, observation, warnings=(warning,))

    def make_step(
        self,
        state,
        observation,
        status,
        reset_tracking_runtime=False,
        step_approach=False,
        approach_bbox_error=None,
        warnings=(),
        release_camera=None,
    ):
        lost_without_publish = (
            not observation.detection_present
            and status in SEARCH_STATUSES
            and not self.config.publish_while_lost
        )
        should_release_camera = (
            status == "FINISHED"
            if release_camera is None
            else release_camera
        )
        return GraspCupStep(
            state=state,
            status=status,
            publish_target=status != "FINISHED" and not lost_without_publish,
            publish_debug=not lost_without_publish,
            release_camera=should_release_camera,
            reset_tracking_runtime=reset_tracking_runtime,
            step_approach=step_approach,
            approach_bbox_error=approach_bbox_error,
            warnings=tuple(warnings),
        )

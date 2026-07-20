import math

from ninibot.skills.grasp_cup.controller import GraspCupController
from ninibot.skills.grasp_cup.controller import GraspCupControllerConfig
from ninibot.skills.grasp_cup.controller import GraspCupObservation
from ninibot.skills.grasp_cup.controller import GraspCupState


def make_controller(**overrides):
    values = {
        "search_enabled": True,
        "search_step": 0.2,
        "theta_min": -1.0,
        "theta_max": 1.0,
        "tracking_lost_grace_sec": 0.5,
        "tracking_settle_cycles": 2,
        "fine_align_centered_cycles": 2,
        "publish_while_lost": False,
    }
    values.update(overrides)
    return GraspCupController(GraspCupControllerConfig(**values))


def make_observation(**overrides):
    values = {
        "detection_present": False,
        "image_target_x": 100.0,
        "now": 0.0,
        "reset_joint2_deg": 21.0,
        "reset_joint3_deg": 32.0,
    }
    values.update(overrides)
    return GraspCupObservation(**values)


def test_update_search_sweeps_and_flips_at_limit():
    controller = make_controller()
    state = GraspCupState(stage="SEARCHING", theta=0.95)

    step = controller.update_state(state, make_observation())

    assert step.status == "SEARCHING"
    assert step.state.theta == 1.0
    assert step.state.search_direction == -1.0


def test_search_prep_ready_transitions_to_searching():
    controller = make_controller()
    state = GraspCupState(stage="SEARCH_PREP")

    step = controller.update_state(
        state,
        make_observation(search_prep_ready=True),
    )

    assert step.status == "SEARCHING"
    assert step.state.stage == "SEARCHING"


def test_tracking_center_requires_settle_cycles_before_fine_align_prep():
    controller = make_controller()
    state = GraspCupState(stage="TRACKING_CENTER")
    observation = make_observation(
        detection_present=True,
        tracking_centered=True,
        tracking_error_x=0.0,
    )

    first = controller.update_state(state, observation)
    second = controller.update_state(first.state, observation)

    assert first.status == "TRACKING_CENTER"
    assert second.status == "FINE_ALIGN_PREP"
    assert second.state.stage == "FINE_ALIGN_PREP"


def test_tracking_center_missing_detection_holds_until_grace_expires():
    controller = make_controller()
    state = GraspCupState(stage="TRACKING_CENTER")

    held = controller.update_state(state, make_observation(now=10.0))
    reset = controller.update_state(held.state, make_observation(now=10.6))

    assert held.status == "TRACKING_CENTER"
    assert held.state.tracking_lost_started_at == 10.0
    assert held.warnings[0].key == "tracking_lost_grace"
    assert reset.status == "SEARCH_PREP"
    assert reset.state.stage == "SEARCH_PREP"
    assert reset.state.search_prep_trajectory is None
    assert reset.reset_tracking_runtime


def test_fine_align_prep_ready_starts_fine_align_state():
    controller = make_controller()
    state = GraspCupState(stage="FINE_ALIGN_PREP")

    step = controller.update_state(
        state,
        make_observation(
            fine_align_prep_ready=True,
            capture_servo_radius_m=0.25,
            capture_servo_z_m=0.12,
            capture_joint2_deg=33.0,
            capture_joint3_deg=44.0,
        ),
    )

    assert step.status == "FINE_ALIGN"
    assert step.state.stage == "FINE_ALIGN"
    assert step.state.fine_align_centered_count == 0
    assert step.state.servo_radius_m == 0.25
    assert step.state.servo_z_m == 0.12
    assert step.state.approach_joint2_deg == 33.0
    assert step.state.approach_joint3_deg == 44.0


def test_fine_align_centered_and_locked_finishes():
    controller = make_controller()
    state = GraspCupState(
        stage="FINE_ALIGN",
        fine_align_centered_count=1,
    )

    step = controller.update_state(
        state,
        make_observation(
            detection_present=True,
            fine_align_centered=True,
            tracking_error_x=0.0,
            bbox_error=1.0,
            approach_locked_after_reset=True,
        ),
    )

    assert step.status == "FINISHED"
    assert step.state.stage == "FINISHED"
    assert step.state.approach_servo_started
    assert step.state.approach_locked
    assert step.step_approach
    assert step.release_camera


def test_unknown_stage_warns_and_resets_to_search_prep():
    controller = make_controller(theta_max=math.radians(10.0))
    state = GraspCupState(stage="SURPRISE", theta=0.0)

    step = controller.update_state(
        state,
        make_observation(latest_joint_theta=math.radians(20.0)),
    )

    assert step.status == "SEARCH_PREP"
    assert step.state.stage == "SEARCH_PREP"
    assert math.isclose(step.state.search_anchor_theta, math.radians(10.0))
    assert step.warnings[0].key == "unknown_stage"

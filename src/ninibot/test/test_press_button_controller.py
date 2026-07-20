from ninibot.skills.press_button.controller import CMD_DETECT_MARKER
from ninibot.skills.press_button.controller import CMD_INCREMENT_PRESS_CYCLE
from ninibot.skills.press_button.controller import CMD_INCREMENT_SETTLE
from ninibot.skills.press_button.controller import CMD_LOG_DONE_SUMMARY
from ninibot.skills.press_button.controller import CMD_LOG_PROGRESS
from ninibot.skills.press_button.controller import CMD_PUBLISH_TARGET
from ninibot.skills.press_button.controller import CMD_RESET_SETTLE
from ninibot.skills.press_button.controller import CMD_SET_CURRENT_TARGET
from ninibot.skills.press_button.controller import CMD_SET_PHASE
from ninibot.skills.press_button.controller import CMD_START_LOCK_WAIT
from ninibot.skills.press_button.controller import CMD_START_PRESS_WAIT
from ninibot.skills.press_button.controller import CMD_START_SEARCH
from ninibot.skills.press_button.controller import PressButtonController
from ninibot.skills.press_button.controller import PressButtonObservation
from ninibot.skills.press_button.state_machine import Phase


def command_names(step):
    return [command.name for command in step.commands]


def test_idle_controller_returns_no_commands():
    controller = PressButtonController()

    step = controller.step(PressButtonObservation(phase=Phase.IDLE))

    assert step.phase == Phase.IDLE
    assert step.commands == ()


def test_prep_following_trajectory_returns_publish_and_wait_commands():
    controller = PressButtonController()

    step = controller.step(
        PressButtonObservation(
            phase=Phase.PREP,
            prep_sampled_target=[1, 2, 3, 4, 5],
            prep_trajectory_ready=False,
        )
    )

    assert command_names(step) == [
        CMD_SET_CURRENT_TARGET,
        CMD_PUBLISH_TARGET,
        CMD_RESET_SETTLE,
        CMD_LOG_PROGRESS,
    ]
    assert step.commands[0].payload["target"] == [1, 2, 3, 4, 5]


def test_prep_settled_starts_search_after_required_cycles():
    controller = PressButtonController()

    step = controller.step(
        PressButtonObservation(
            phase=Phase.PREP,
            settle_count=7,
            settle_cycles=8,
            prep_sampled_target=[1, 2, 3, 4, 5],
            prep_goal_target=[2, 3, 4, 5, 6],
            prep_trajectory_ready=True,
            prep_at_goal=True,
        )
    )

    assert command_names(step) == [
        CMD_SET_CURRENT_TARGET,
        CMD_PUBLISH_TARGET,
        CMD_SET_CURRENT_TARGET,
        CMD_PUBLISH_TARGET,
        CMD_INCREMENT_SETTLE,
        CMD_RESET_SETTLE,
        CMD_START_SEARCH,
    ]


def test_search_starts_lock_wait_when_marker_is_visible():
    controller = PressButtonController()
    marker = object()

    step = controller.step(
        PressButtonObservation(
            phase=Phase.SEARCH,
            marker_base_xyz=marker,
        )
    )

    assert command_names(step) == [CMD_PUBLISH_TARGET, CMD_START_LOCK_WAIT]
    assert step.commands[1].payload["marker_base_xyz"] is marker


def test_move_to_marker_starts_press_wait_without_fine_align():
    controller = PressButtonController()

    step = controller.step(
        PressButtonObservation(
            phase=Phase.MOVE_TO_MARKER,
            settle_count=7,
            settle_cycles=8,
            fine_align_enable=False,
            move_ramp_ready=True,
            move_at_goal=True,
        )
    )

    assert command_names(step) == [
        CMD_PUBLISH_TARGET,
        CMD_INCREMENT_SETTLE,
        CMD_START_PRESS_WAIT,
    ]


def test_press_lift_done_returns_summary_and_done_phase_commands():
    controller = PressButtonController()

    step = controller.step(
        PressButtonObservation(
            phase=Phase.PRESS_LIFT,
            settle_count=2,
            press_motion_settle_cycles=3,
            press_lift_ready=True,
            press_cycle_index=0,
            press_cycles=1,
        )
    )

    assert command_names(step) == [
        CMD_PUBLISH_TARGET,
        CMD_DETECT_MARKER,
        CMD_INCREMENT_SETTLE,
        CMD_INCREMENT_PRESS_CYCLE,
        CMD_LOG_DONE_SUMMARY,
        CMD_SET_PHASE,
    ]
    assert step.commands[-1].payload["phase"] == Phase.DONE

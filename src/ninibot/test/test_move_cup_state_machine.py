from ninibot.skills.move_cup.phases import CANCELED
from ninibot.skills.move_cup.phases import DONE
from ninibot.skills.move_cup.phases import FAILED
from ninibot.skills.move_cup.phases import GRIPPING
from ninibot.skills.move_cup.phases import IDLE
from ninibot.skills.move_cup.phases import LIFTING
from ninibot.skills.move_cup.phases import MOVING_TO_TARGET
from ninibot.skills.move_cup.phases import RETRACTING_RADIUS
from ninibot.skills.move_cup.phases import SEARCH_TARGET
from ninibot.skills.move_cup.phases import SEARCH_TARGET_PREP
from ninibot.skills.move_cup.phases import TARGET_LOCKED
from ninibot.skills.move_cup.state_machine import ACTION_NONE
from ninibot.skills.move_cup.state_machine import ACTION_START_EXTEND_RADIUS
from ninibot.skills.move_cup.state_machine import ACTION_START_HORIZONTAL_MOVE
from ninibot.skills.move_cup.state_machine import ACTION_START_LIFTING
from ninibot.skills.move_cup.state_machine import ACTION_START_RETRACT_RADIUS
from ninibot.skills.move_cup.state_machine import COMMAND_CANCEL
from ninibot.skills.move_cup.state_machine import COMMAND_RESET
from ninibot.skills.move_cup.state_machine import COMMAND_START_LOCKED
from ninibot.skills.move_cup.state_machine import COMMAND_START_MOVE
from ninibot.skills.move_cup.state_machine import COMMAND_START_SCAN
from ninibot.skills.move_cup.state_machine import COMMAND_UNKNOWN
from ninibot.skills.move_cup.state_machine import MOVE_ACTION_ACTIVE_STATUSES
from ninibot.skills.move_cup.state_machine import MoveCupController
from ninibot.skills.move_cup.state_machine import MoveCupControllerConfig
from ninibot.skills.move_cup.state_machine import MoveCupObservation
from ninibot.skills.move_cup.state_machine import MoveCupState
from ninibot.skills.move_cup.state_machine import can_start_from_phase
from ninibot.skills.move_cup.state_machine import parse_move_cup_command


def test_parse_move_cup_command_aliases():
    assert parse_move_cup_command("start") == COMMAND_START_MOVE
    assert parse_move_cup_command("move") == COMMAND_START_MOVE
    assert parse_move_cup_command("scan") == COMMAND_START_SCAN
    assert parse_move_cup_command("lock_target") == COMMAND_START_SCAN
    assert parse_move_cup_command("move_locked") == COMMAND_START_LOCKED
    assert parse_move_cup_command("start_locked") == COMMAND_START_LOCKED
    assert parse_move_cup_command("cancel") == COMMAND_CANCEL
    assert parse_move_cup_command("stop") == COMMAND_CANCEL
    assert parse_move_cup_command("reset") == COMMAND_RESET
    assert parse_move_cup_command("wat") == COMMAND_UNKNOWN


def test_startable_phases_match_node_contract():
    assert can_start_from_phase(IDLE)
    assert can_start_from_phase(DONE)
    assert can_start_from_phase(FAILED)
    assert can_start_from_phase(CANCELED)
    assert can_start_from_phase(TARGET_LOCKED)
    assert not can_start_from_phase(GRIPPING)
    assert not can_start_from_phase(SEARCH_TARGET)


def test_action_active_status_contract_includes_scan_and_move_phases():
    assert SEARCH_TARGET_PREP in MOVE_ACTION_ACTIVE_STATUSES
    assert SEARCH_TARGET in MOVE_ACTION_ACTIVE_STATUSES
    assert TARGET_LOCKED in MOVE_ACTION_ACTIVE_STATUSES
    assert GRIPPING in MOVE_ACTION_ACTIVE_STATUSES


def test_move_cup_controller_waits_for_gripper_settle():
    controller = MoveCupController(MoveCupControllerConfig(settle_cycles=2))

    waiting = controller.advance_if_ready(
        MoveCupState(GRIPPING, settle_count=0),
        MoveCupObservation(gripper_waiting=True),
    )
    ready = controller.advance_if_ready(
        MoveCupState(GRIPPING, settle_count=0),
        MoveCupObservation(gripper_waiting=False),
    )

    assert waiting.action == ACTION_NONE
    assert ready.action == ACTION_START_LIFTING


def test_move_cup_controller_holds_active_path_until_complete():
    controller = MoveCupController(MoveCupControllerConfig(settle_cycles=1))

    step = controller.advance_if_ready(
        MoveCupState(RETRACTING_RADIUS, settle_count=0),
        MoveCupObservation(path_complete=False, target_ready=True),
    )

    assert step.action == ACTION_NONE
    assert step.state.settle_count == 0


def test_move_cup_controller_settles_then_advances_phase():
    controller = MoveCupController(MoveCupControllerConfig(settle_cycles=2))

    first = controller.advance_if_ready(
        MoveCupState(LIFTING, settle_count=0),
        MoveCupObservation(target_ready=True),
    )
    second = controller.advance_if_ready(
        first.state,
        MoveCupObservation(target_ready=True),
    )

    assert first.action == ACTION_NONE
    assert first.state.settle_count == 1
    assert second.action == ACTION_START_RETRACT_RADIUS
    assert second.state.settle_count == 2


def test_move_cup_controller_resets_settle_count_when_not_at_target():
    controller = MoveCupController(MoveCupControllerConfig(settle_cycles=2))

    step = controller.advance_if_ready(
        MoveCupState(LIFTING, settle_count=1),
        MoveCupObservation(target_ready=False),
    )

    assert step.action == ACTION_NONE
    assert step.log_target_wait
    assert step.state.settle_count == 0


def test_move_cup_controller_timeout_can_advance_non_lifting_phase():
    controller = MoveCupController(MoveCupControllerConfig(settle_cycles=3))

    step = controller.advance_if_ready(
        MoveCupState(MOVING_TO_TARGET, settle_count=0),
        MoveCupObservation(
            target_ready=False,
            timeout_advance_ready=True,
        ),
    )

    assert step.action == ACTION_START_EXTEND_RADIUS

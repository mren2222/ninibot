from dataclasses import dataclass

from ninibot.skills.move_cup.phases import ACTIVE_PHASES
from ninibot.skills.move_cup.phases import CANCELED
from ninibot.skills.move_cup.phases import DONE
from ninibot.skills.move_cup.phases import EXTENDING_RADIUS
from ninibot.skills.move_cup.phases import FAILED
from ninibot.skills.move_cup.phases import GRIPPING
from ninibot.skills.move_cup.phases import IDLE
from ninibot.skills.move_cup.phases import LIFTING
from ninibot.skills.move_cup.phases import LOWERING_TO_RELEASE
from ninibot.skills.move_cup.phases import MOVING_TO_TARGET
from ninibot.skills.move_cup.phases import PATH_PHASES
from ninibot.skills.move_cup.phases import POST_RELEASE_CLEARANCE
from ninibot.skills.move_cup.phases import RELEASING
from ninibot.skills.move_cup.phases import RETRACTING_RADIUS
from ninibot.skills.move_cup.phases import RETURNING_SEARCH_PREP
from ninibot.skills.move_cup.phases import SEARCH_TARGET
from ninibot.skills.move_cup.phases import SEARCH_TARGET_PREP
from ninibot.skills.move_cup.phases import TARGET_LOCKED


COMMAND_START_MOVE = "start_move"
COMMAND_START_SCAN = "start_scan"
COMMAND_START_LOCKED = "start_locked"
COMMAND_CANCEL = "cancel"
COMMAND_RESET = "reset"
COMMAND_UNKNOWN = "unknown"

ACTION_NONE = "none"
ACTION_START_LIFTING = "start_lifting"
ACTION_START_RETRACT_RADIUS = "start_retract_radius"
ACTION_START_HORIZONTAL_MOVE = "start_horizontal_move"
ACTION_START_EXTEND_RADIUS = "start_extend_radius"
ACTION_START_LOWERING = "start_lowering"
ACTION_START_RELEASING = "start_releasing"
ACTION_START_POST_RELEASE_CLEARANCE = "start_post_release_clearance"
ACTION_START_RETURN_SEARCH_PREP = "start_return_search_prep"
ACTION_DONE = "done"

START_MOVE_COMMANDS = {"start", "move"}
START_SCAN_COMMANDS = {"scan", "lock", "lock_target"}
START_LOCKED_COMMANDS = {"move_locked", "start_locked"}
CANCEL_COMMANDS = {"cancel", "stop"}
RESET_COMMANDS = {"reset"}

STARTABLE_PHASES = {
    IDLE,
    DONE,
    FAILED,
    CANCELED,
    TARGET_LOCKED,
}

MOVE_ACTION_ACTIVE_STATUSES = ACTIVE_PHASES | {
    TARGET_LOCKED,
    SEARCH_TARGET_PREP,
    SEARCH_TARGET,
}


def parse_move_cup_command(command):
    command = str(command).strip().lower()
    if command in START_MOVE_COMMANDS:
        return COMMAND_START_MOVE
    if command in START_SCAN_COMMANDS:
        return COMMAND_START_SCAN
    if command in START_LOCKED_COMMANDS:
        return COMMAND_START_LOCKED
    if command in CANCEL_COMMANDS:
        return COMMAND_CANCEL
    if command in RESET_COMMANDS:
        return COMMAND_RESET
    return COMMAND_UNKNOWN


def can_start_from_phase(phase):
    return phase in STARTABLE_PHASES


@dataclass(frozen=True)
class MoveCupControllerConfig:
    settle_cycles: int


@dataclass(frozen=True)
class MoveCupState:
    phase: str
    settle_count: int = 0


@dataclass(frozen=True)
class MoveCupObservation:
    target_ready: bool = False
    path_complete: bool = True
    gripper_waiting: bool = False
    timeout_advance_ready: bool = False


@dataclass(frozen=True)
class MoveCupStep:
    state: MoveCupState
    action: str = ACTION_NONE
    log_target_wait: bool = False


class MoveCupController:
    def __init__(self, config):
        self.config = config

    def advance_if_ready(self, state, observation):
        phase = state.phase
        settle_count = state.settle_count

        if phase == GRIPPING:
            if observation.gripper_waiting:
                return MoveCupStep(state)
            return MoveCupStep(state, ACTION_START_LIFTING)

        if phase in PATH_PHASES and not observation.path_complete:
            return MoveCupStep(state)

        target_ready = observation.target_ready
        if (
            not target_ready
            and phase != LIFTING
            and observation.timeout_advance_ready
        ):
            target_ready = True
            settle_count = self.config.settle_cycles

        if not target_ready:
            return MoveCupStep(
                MoveCupState(phase, 0),
                log_target_wait=True,
            )

        settle_count += 1
        next_state = MoveCupState(phase, settle_count)
        if settle_count < self.config.settle_cycles:
            return MoveCupStep(next_state)

        return MoveCupStep(
            next_state,
            self.action_for_settled_phase(phase),
        )

    @staticmethod
    def action_for_settled_phase(phase):
        return {
            LIFTING: ACTION_START_RETRACT_RADIUS,
            RETRACTING_RADIUS: ACTION_START_HORIZONTAL_MOVE,
            MOVING_TO_TARGET: ACTION_START_EXTEND_RADIUS,
            EXTENDING_RADIUS: ACTION_START_LOWERING,
            LOWERING_TO_RELEASE: ACTION_START_RELEASING,
            RELEASING: ACTION_START_POST_RELEASE_CLEARANCE,
            POST_RELEASE_CLEARANCE: ACTION_START_RETURN_SEARCH_PREP,
            RETURNING_SEARCH_PREP: ACTION_DONE,
        }.get(phase, ACTION_NONE)

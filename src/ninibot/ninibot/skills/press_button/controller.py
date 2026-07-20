from dataclasses import dataclass
from dataclasses import field

from ninibot.skills.press_button.state_machine import Phase


CMD_DETECT_MARKER = "detect_marker"
CMD_FINISH_FINE_ALIGN_IK_WAIT = "finish_fine_align_ik_wait"
CMD_FINISH_FINE_ALIGN_SAMPLE = "finish_fine_align_sample"
CMD_INCREMENT_PRESS_CYCLE = "increment_press_cycle"
CMD_INCREMENT_SETTLE = "increment_settle"
CMD_LOG_DONE_SUMMARY = "log_done_summary"
CMD_LOG_PROGRESS = "log_progress"
CMD_PUBLISH_TARGET = "publish_target"
CMD_RESET_LOCK_WAIT_TIMER = "reset_lock_wait_timer"
CMD_RESET_SETTLE = "reset_settle"
CMD_SET_CURRENT_TARGET = "set_current_target"
CMD_SET_PHASE = "set_phase"
CMD_SHOW_LOCKED_MARKER_DEBUG = "show_locked_marker_debug"
CMD_START_FINE_ALIGN_SAMPLE = "start_fine_align_sample"
CMD_START_LOCK_WAIT = "start_lock_wait"
CMD_START_MOVE_TO_MARKER = "start_move_to_marker"
CMD_START_PRESS_LIFT_OR_FAIL = "start_press_lift_or_fail"
CMD_START_PRESS_NUDGE_OR_FAIL = "start_press_nudge_or_fail"
CMD_START_PRESS_REPEAT_WAIT = "start_press_repeat_wait"
CMD_START_PRESS_WAIT = "start_press_wait"
CMD_START_SEARCH = "start_search"


@dataclass(frozen=True)
class PressButtonCommand:
    name: str
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PressButtonObservation:
    phase: str
    camera_ok: bool = True
    settle_count: int = 0
    settle_cycles: int = 0
    prep_sampled_target: object = None
    prep_goal_target: object = None
    prep_trajectory_ready: bool = False
    prep_at_goal: bool = False
    marker_base_xyz: object = None
    lock_wait_ready: bool = False
    locked_marker_base_xyz: object = None
    locked_marker_target_base_xyz: object = None
    lock_wait_target_pose: object = None
    lock_wait_pose_failed: bool = False
    fine_align_enable: bool = False
    should_start_fine_align: bool = False
    move_ramp_ready: bool = False
    move_at_goal: bool = False
    fine_align_sample_count: int = 0
    fine_align_marker_samples: int = 0
    fine_align_sample_ready: bool = False
    fine_align_move_ready: bool = False
    fine_align_move_timed_out: bool = False
    fine_align_move_timeout_text: str = ""
    fine_align_settle_cycles: int = 0
    fine_align_move_settle_text: str = ""
    press_wait_ready: bool = False
    press_nudge_ready: bool = False
    press_lift_ready: bool = False
    press_repeat_wait_ready: bool = False
    press_motion_settle_cycles: int = 0
    press_cycle_index: int = 0
    press_cycles: int = 1
    done_log_text: str = ""


@dataclass(frozen=True)
class PressButtonStep:
    phase: str
    commands: tuple[PressButtonCommand, ...] = ()


def command(name, **payload):
    return PressButtonCommand(name, payload)


class PressButtonController:
    def step(self, observation):
        if not observation.camera_ok or observation.phase == Phase.IDLE:
            return PressButtonStep(observation.phase)

        handlers = {
            Phase.PREP: self.handle_prep,
            Phase.SEARCH: self.handle_search,
            Phase.LOCK_WAIT: self.handle_lock_wait,
            Phase.MOVE_TO_MARKER: self.handle_move_to_marker,
            Phase.FINE_ALIGN_SAMPLE: self.handle_fine_align_sample,
            Phase.FINE_ALIGN_IK_WAIT: self.handle_fine_align_ik_wait,
            Phase.FINE_ALIGN_MOVE: self.handle_fine_align_move,
            Phase.PRESS_WAIT: self.handle_press_wait,
            Phase.PRESS_NUDGE: self.handle_press_nudge,
            Phase.PRESS_LIFT: self.handle_press_lift,
            Phase.PRESS_REPEAT_WAIT: self.handle_press_repeat_wait,
            Phase.DONE: self.handle_done,
        }
        handler = handlers.get(observation.phase)
        if handler is None:
            return PressButtonStep(observation.phase)
        return PressButtonStep(observation.phase, tuple(handler(observation)))

    def handle_prep(self, observation):
        if observation.prep_sampled_target is None:
            return ()

        commands = [
            command(CMD_SET_CURRENT_TARGET, target=observation.prep_sampled_target),
            command(CMD_PUBLISH_TARGET),
        ]
        if not observation.prep_trajectory_ready:
            commands.extend(
                [
                    command(CMD_RESET_SETTLE),
                    command(CMD_LOG_PROGRESS, text="press button: PREP following trajectory"),
                ]
            )
            return commands

        commands.extend(
            [
                command(CMD_SET_CURRENT_TARGET, target=observation.prep_goal_target),
                command(CMD_PUBLISH_TARGET),
            ]
        )
        if observation.prep_at_goal:
            commands.append(command(CMD_INCREMENT_SETTLE))
        else:
            commands.extend(
                [
                    command(CMD_RESET_SETTLE),
                    command(CMD_LOG_PROGRESS, text="press button: PREP waiting for final target"),
                ]
            )
            return commands

        if observation.settle_count + 1 >= observation.settle_cycles:
            commands.extend(
                [
                    command(CMD_RESET_SETTLE),
                    command(CMD_START_SEARCH),
                ]
            )
        return commands

    def handle_search(self, observation):
        commands = [command(CMD_PUBLISH_TARGET)]
        if observation.marker_base_xyz is not None:
            commands.append(
                command(
                    CMD_START_LOCK_WAIT,
                    marker_base_xyz=observation.marker_base_xyz,
                )
            )
        return commands

    def handle_lock_wait(self, observation):
        commands = [
            command(CMD_PUBLISH_TARGET),
            command(CMD_SHOW_LOCKED_MARKER_DEBUG),
        ]
        if not observation.lock_wait_ready:
            commands.append(
                command(
                    CMD_LOG_PROGRESS,
                    text="press button: LOCK_WAIT auto move in {lock_wait_sec:.1f}s",
                )
            )
            return commands

        if observation.locked_marker_target_base_xyz is None:
            commands.extend(
                [
                    command(CMD_RESET_LOCK_WAIT_TIMER),
                    command(CMD_LOG_PROGRESS, text="press button: marker target unavailable"),
                ]
            )
            return commands

        if observation.lock_wait_pose_failed:
            commands.extend(
                [
                    command(CMD_RESET_LOCK_WAIT_TIMER),
                    command(
                        CMD_LOG_PROGRESS,
                        text=(
                            "press button: marker target unreachable "
                            "target_base_cm={target_base_cm} "
                            "marker_base_cm={marker_base_cm}"
                        ),
                    ),
                ]
            )
            return commands

        commands.append(
            command(
                CMD_START_MOVE_TO_MARKER,
                pose=observation.lock_wait_target_pose,
                marker_base_xyz=observation.locked_marker_base_xyz,
                target_base_xyz=observation.locked_marker_target_base_xyz,
            )
        )
        return commands

    def handle_move_to_marker(self, observation):
        if observation.should_start_fine_align:
            return (
                command(CMD_START_FINE_ALIGN_SAMPLE, reason="within start distance"),
            )

        commands = [command(CMD_PUBLISH_TARGET)]
        if not observation.move_ramp_ready:
            commands.extend(
                [
                    command(CMD_RESET_SETTLE),
                    command(CMD_LOG_PROGRESS, text="press button: MOVE_TO_MARKER ramping target"),
                ]
            )
            return commands

        if observation.move_at_goal:
            commands.append(command(CMD_INCREMENT_SETTLE))
        else:
            commands.extend(
                [
                    command(CMD_RESET_SETTLE),
                    command(CMD_LOG_PROGRESS, text="press button: MOVE_TO_MARKER waiting for target"),
                ]
            )
            return commands

        if observation.settle_count + 1 >= observation.settle_cycles:
            if observation.fine_align_enable:
                commands.append(
                    command(CMD_START_FINE_ALIGN_SAMPLE, reason="move target settled")
                )
            else:
                commands.append(command(CMD_START_PRESS_WAIT))
        return commands

    def handle_fine_align_sample(self, observation):
        commands = [command(CMD_PUBLISH_TARGET)]
        if not observation.fine_align_sample_ready:
            commands.append(
                command(
                    CMD_LOG_PROGRESS,
                    text=(
                        "press button: FINE_ALIGN_SAMPLE "
                        f"{observation.fine_align_sample_count}/"
                        f"{observation.fine_align_marker_samples} samples"
                    ),
                )
            )
            return commands
        commands.append(command(CMD_FINISH_FINE_ALIGN_SAMPLE))
        return commands

    def handle_fine_align_ik_wait(self, observation):
        return (
            command(CMD_PUBLISH_TARGET),
            command(CMD_DETECT_MARKER),
            command(CMD_FINISH_FINE_ALIGN_IK_WAIT),
        )

    def handle_fine_align_move(self, observation):
        commands = [
            command(CMD_PUBLISH_TARGET),
            command(CMD_DETECT_MARKER),
        ]
        if not observation.fine_align_move_ready:
            commands.append(command(CMD_RESET_SETTLE))
            if observation.fine_align_move_timed_out:
                commands.append(
                    command(
                        CMD_SET_PHASE,
                        phase=Phase.FAILED,
                        log_text=observation.fine_align_move_timeout_text,
                    )
                )
                return commands
            commands.append(
                command(CMD_LOG_PROGRESS, text="press button: FINE_ALIGN_MOVE ramping slowly")
            )
            return commands

        commands.extend(
            [
                command(CMD_INCREMENT_SETTLE),
                command(
                    CMD_LOG_PROGRESS,
                    text=observation.fine_align_move_settle_text,
                    period_sec=0.25,
                ),
            ]
        )
        if observation.settle_count + 1 >= observation.fine_align_settle_cycles:
            commands.append(
                command(CMD_START_FINE_ALIGN_SAMPLE, reason="step settled")
            )
        return commands

    def handle_press_wait(self, observation):
        commands = [
            command(CMD_PUBLISH_TARGET),
            command(CMD_DETECT_MARKER),
        ]
        if not observation.press_wait_ready:
            commands.append(
                command(CMD_LOG_PROGRESS, text="press button: PRESS_WAIT {press_wait_sec:.1f}s")
            )
            return commands
        commands.append(command(CMD_START_PRESS_NUDGE_OR_FAIL))
        return commands

    def handle_press_nudge(self, observation):
        commands = [
            command(CMD_PUBLISH_TARGET),
            command(CMD_DETECT_MARKER),
        ]
        if not observation.press_nudge_ready:
            commands.extend(
                [
                    command(CMD_RESET_SETTLE),
                    command(CMD_LOG_PROGRESS, text="press button: PRESS_NUDGE ramping joint2/3/4"),
                ]
            )
            return commands

        commands.append(command(CMD_INCREMENT_SETTLE))
        if observation.settle_count + 1 >= observation.press_motion_settle_cycles:
            commands.append(command(CMD_START_PRESS_LIFT_OR_FAIL))
        return commands

    def handle_press_lift(self, observation):
        commands = [
            command(CMD_PUBLISH_TARGET),
            command(CMD_DETECT_MARKER),
        ]
        if not observation.press_lift_ready:
            commands.extend(
                [
                    command(CMD_RESET_SETTLE),
                    command(CMD_LOG_PROGRESS, text="press button: PRESS_LIFT ramping joint2/3/4"),
                ]
            )
            return commands

        commands.append(command(CMD_INCREMENT_SETTLE))
        if observation.settle_count + 1 < observation.press_motion_settle_cycles:
            return commands

        commands.append(command(CMD_INCREMENT_PRESS_CYCLE))
        if observation.press_cycle_index + 1 >= observation.press_cycles:
            commands.extend(
                [
                    command(CMD_LOG_DONE_SUMMARY),
                    command(
                        CMD_SET_PHASE,
                        phase=Phase.DONE,
                        log_text=observation.done_log_text,
                    ),
                ]
            )
        else:
            commands.append(command(CMD_START_PRESS_REPEAT_WAIT))
        return commands

    def handle_press_repeat_wait(self, observation):
        commands = [
            command(CMD_PUBLISH_TARGET),
            command(CMD_DETECT_MARKER),
        ]
        if not observation.press_repeat_wait_ready:
            commands.append(
                command(
                    CMD_LOG_PROGRESS,
                    text="press button: PRESS_REPEAT_WAIT {press_repeat_wait_sec:.1f}s",
                )
            )
            return commands
        commands.append(command(CMD_START_PRESS_NUDGE_OR_FAIL, repeat=True))
        return commands

    def handle_done(self, observation):
        return (command(CMD_DETECT_MARKER),)

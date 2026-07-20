import time

from ninibot.skills.press_button.state_machine import Phase


class PressSequence:
    def __init__(self, node):
        self.node = node

    def start_press_wait(self):
        node = self.node
        base = node.move_goal or node.current_target
        if base is not None:
            node.press_base_goal = list(base)
        node.press_wait_start_time = time.monotonic()
        node.press_goal = None
        node.lift_goal = None
        node.press_repeat_wait_start_time = None
        node.press_cycle_index = 0
        node.settle_count = 0
        node.set_phase(
            Phase.PRESS_WAIT,
            f"press button: PRESS_WAIT {node.press_wait_sec:.1f}s before press nudge",
        )

    def press_wait_ready(self):
        node = self.node
        if node.press_wait_start_time is None:
            return node.press_wait_sec <= 0.0
        return (time.monotonic() - node.press_wait_start_time) >= node.press_wait_sec

    def start_press_nudge(self):
        node = self.node
        base = list(
            node.press_base_goal
            or node.move_goal
            or node.current_target
            or []
        )
        goal = node.press_motion_planner.make_press_nudge(base)
        if goal is None:
            return False
        node.press_goal = goal
        node.settle_count = 0
        node.set_phase(
            Phase.PRESS_NUDGE,
            "press button: PRESS_NUDGE "
            f"cycle={node.press_cycle_index + 1}/{node.press_cycles} "
            f"joint2_delta={node.press_joint2_delta_deg:+.1f} deg "
            f"joint3_delta={node.press_joint3_delta_deg:+.1f} deg "
            f"joint4_delta={node.press_joint4_delta_deg:+.1f} deg "
            f"q1234_goal_deg={node.format_joint_target_deg(goal)}",
        )
        return True

    def ramp_press_target(self):
        node = self.node
        return node.ramp_joint_goal(
            node.press_goal,
            [1, 2, 3],
            node.press_motion_speed_scale,
        )

    def start_press_lift(self):
        node = self.node
        goal = node.press_motion_planner.make_press_lift(node.press_goal)
        if goal is None:
            return False
        node.lift_goal = goal
        node.settle_count = 0
        node.set_phase(
            Phase.PRESS_LIFT,
            "press button: PRESS_LIFT "
            f"cycle={node.press_cycle_index + 1}/{node.press_cycles} "
            f"joint2_delta={-node.press_joint2_delta_deg:+.1f} deg "
            f"joint3_delta={-node.press_joint3_delta_deg:+.1f} deg "
            f"joint4_delta={-node.press_joint4_delta_deg:+.1f} deg "
            f"q1234_goal_deg={node.format_joint_target_deg(goal)}",
        )
        return True

    def ramp_lift_target(self):
        node = self.node
        return node.ramp_joint_goal(
            node.lift_goal,
            [1, 2, 3],
            node.press_motion_speed_scale,
        )

    def start_press_repeat_wait(self):
        node = self.node
        node.press_repeat_wait_start_time = time.monotonic()
        node.settle_count = 0
        node.set_phase(
            Phase.PRESS_REPEAT_WAIT,
            "press button: PRESS_REPEAT_WAIT "
            f"{node.press_repeat_wait_sec:.1f}s before next press "
            f"cycle={node.press_cycle_index + 1}/{node.press_cycles}",
        )

    def press_repeat_wait_ready(self):
        node = self.node
        if node.press_repeat_wait_start_time is None:
            return node.press_repeat_wait_sec <= 0.0
        return (
            time.monotonic() - node.press_repeat_wait_start_time
        ) >= node.press_repeat_wait_sec

import math

from ninibot.skills.press_button.math_utils import clamp
from ninibot.skills.press_button.params import PressMotionConfig


class PressMotionPlanner:
    def __init__(self, config: PressMotionConfig):
        self.config = config

    def make_press_nudge(self, base_pose):
        base = list(base_pose or [])
        if len(base) < 4:
            return None
        if len(base) < 5:
            base.append(self.config.joint6_rad)

        goal = list(base)
        goal[1] = clamp(
            goal[1] + math.radians(self.config.press_joint2_delta_deg),
            self.config.ik_lower_bounds[1],
            self.config.ik_upper_bounds[1],
        )
        goal[2] = clamp(
            goal[2] + math.radians(self.config.press_joint3_delta_deg),
            self.config.ik_lower_bounds[2],
            self.config.ik_upper_bounds[2],
        )
        goal[3] = clamp(
            goal[3] + math.radians(self.config.press_joint4_delta_deg),
            self.config.joint4_lower_rad,
            self.config.joint4_upper_rad,
        )
        goal[4] = self.config.joint6_rad
        return goal

    def make_press_lift(self, press_goal):
        if press_goal is None:
            return None

        goal = list(press_goal)
        goal[1] = clamp(
            goal[1] - math.radians(self.config.press_joint2_delta_deg),
            self.config.ik_lower_bounds[1],
            self.config.ik_upper_bounds[1],
        )
        goal[2] = clamp(
            goal[2] - math.radians(self.config.press_joint3_delta_deg),
            self.config.ik_lower_bounds[2],
            self.config.ik_upper_bounds[2],
        )
        goal[3] = clamp(
            goal[3] - math.radians(self.config.press_joint4_delta_deg),
            self.config.joint4_lower_rad,
            self.config.joint4_upper_rad,
        )
        goal[4] = self.config.joint6_rad
        return goal


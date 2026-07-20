import math
from dataclasses import dataclass

from ninibot.skills.press_button.math_utils import clamp
from ninibot.skills.press_button.params import SearchScanConfig


@dataclass(frozen=True)
class SearchScanState:
    pan_deg: float
    direction: float = 1.0


class SearchScanner:
    def __init__(self, config: SearchScanConfig):
        self.config = config

    def initial_state(self, latest_joint_positions=None):
        if latest_joint_positions is not None:
            pan_deg = math.degrees(latest_joint_positions[0])
        else:
            pan_deg = self.config.prep_joint1_deg
        pan_deg = clamp(
            pan_deg,
            self.config.search_pan_min_deg,
            self.config.search_pan_max_deg,
        )
        return SearchScanState(pan_deg=pan_deg, direction=1.0)

    def pose(self, pan_deg):
        return [
            math.radians(pan_deg),
            math.radians(self.config.prep_joint2_deg),
            math.radians(self.config.prep_joint3_deg),
            math.radians(self.config.prep_joint4_deg),
            self.config.joint6_rad,
        ]

    def update(self, state: SearchScanState, dt_sec):
        span = self.config.search_pan_max_deg - self.config.search_pan_min_deg
        if span <= 1e-6 or self.config.search_pan_speed_deg_s <= 1e-6:
            return SearchScanState(self.config.prep_joint1_deg, state.direction)

        pan_deg = state.pan_deg + (
            state.direction * self.config.search_pan_speed_deg_s * max(0.0, dt_sec)
        )
        direction = state.direction
        if pan_deg >= self.config.search_pan_max_deg:
            pan_deg = self.config.search_pan_max_deg
            direction = -1.0
        elif pan_deg <= self.config.search_pan_min_deg:
            pan_deg = self.config.search_pan_min_deg
            direction = 1.0
        return SearchScanState(pan_deg=pan_deg, direction=direction)


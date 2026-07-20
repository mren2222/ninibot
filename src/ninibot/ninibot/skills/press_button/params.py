from dataclasses import dataclass


@dataclass(frozen=True)
class PrepTrajectoryConfig:
    prep_joint1_deg: float
    prep_joint2_deg: float
    prep_joint3_deg: float
    prep_joint4_deg: float
    duration_sec: float
    j2_transition_sec: float
    search_pan_min_deg: float
    search_pan_max_deg: float
    joint4_lower_rad: float
    joint4_upper_rad: float
    joint6_rad: float


@dataclass(frozen=True)
class SearchScanConfig:
    prep_joint1_deg: float
    prep_joint2_deg: float
    prep_joint3_deg: float
    prep_joint4_deg: float
    search_pan_min_deg: float
    search_pan_max_deg: float
    search_pan_speed_deg_s: float
    joint6_rad: float


@dataclass(frozen=True)
class PressMotionConfig:
    press_joint2_delta_deg: float
    press_joint3_delta_deg: float
    press_joint4_delta_deg: float
    joint6_rad: float
    ik_lower_bounds: tuple
    ik_upper_bounds: tuple
    joint4_lower_rad: float
    joint4_upper_rad: float


@dataclass(frozen=True)
class FineAlignConfig:
    tolerance_m: float
    max_step_m: float
    max_cycles: int
    integral_enable: bool
    max_command_ahead_m: float


import math
from dataclasses import dataclass


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class ApproachServoConfig:
    bbox_tolerance_px: float
    bbox_resume_tolerance_px: float
    radius_kp_m_per_px: float
    max_radius_step_m: float


@dataclass(frozen=True)
class ApproachStep:
    success: bool
    radius_m: float = None
    joint2_deg: float = None
    joint3_deg: float = None
    warning_key: str = ""
    warning_text: str = ""


class ApproachServo:
    def __init__(self, config):
        self.config = config

    def update_lock(self, locked, bbox_error):
        if locked and abs(bbox_error) > self.config.bbox_resume_tolerance_px:
            return False

        if not locked and abs(bbox_error) <= self.config.bbox_tolerance_px:
            return True

        return locked

    def should_step(self, started, locked, bbox_error):
        return (
            started
            and not locked
            and abs(bbox_error) > self.config.bbox_tolerance_px
        )

    def radius_step(self, bbox_error):
        radius_step = self.config.radius_kp_m_per_px * bbox_error
        return clamp(
            radius_step,
            -self.config.max_radius_step_m,
            self.config.max_radius_step_m,
        )

    def step_cylindrical(self, radius_m, theta_rad, z_m, kinematics, bbox_error):
        next_radius = radius_m + self.radius_step(bbox_error)
        solution = kinematics.cylindrical_ik(
            next_radius,
            math.degrees(theta_rad),
            z_m,
        )

        if solution is None:
            return ApproachStep(
                success=False,
                warning_key="cylindrical_approach_ik",
                warning_text=kinematics.last_error,
            )

        return ApproachStep(
            success=True,
            radius_m=next_radius,
            joint2_deg=solution.joint2_deg,
            joint3_deg=solution.joint3_deg,
        )

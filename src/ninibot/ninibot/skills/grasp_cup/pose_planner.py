import math
from dataclasses import dataclass

from ninibot.skills.common.math_utils import GRIPPER_MAX_RAD
from ninibot.skills.common.math_utils import GRIPPER_MIN_RAD
from ninibot.skills.common.math_utils import clamp
from ninibot.skills.common.math_utils import gripper_percent_to_rad

HEAD_DOWN_STATUSES = {"SEARCH_PREP", "TRACKING_CENTER"}


@dataclass(frozen=True)
class PosePlannerConfig:
    search_joint2_deg: float
    search_joint3_deg: float
    search_joint4_deg: float
    wrist_flex_min_deg: float
    wrist_flex_max_deg: float
    wrist_roll_deg: float
    gripper_rad: float
    base_min_deg: float
    base_max_deg: float


@dataclass(frozen=True)
class PosePlan:
    positions: list
    joint2_deg: float = 0.0
    joint3_deg: float = 0.0
    warning_key: str = ""
    warning_text: str = ""


class PosePlanner:
    def __init__(self, config, kinematics):
        self.config = config
        self.kinematics = kinematics

    def base_deg_from_theta(self, theta_rad):
        return self.kinematics.base_deg_from_theta(math.degrees(theta_rad))

    def base_in_range(self, base_deg):
        return self.config.base_min_deg <= base_deg <= self.config.base_max_deg

    def head_down_offset_rad(self, status):
        if status not in HEAD_DOWN_STATUSES:
            return 0.0

        # Positive wrist pitch is the physical "head down" direction on this arm.
        return math.radians(self.config.search_joint4_deg)

    def horizontal_wrist_pitch_rad(self, joint2_rad, joint3_rad):
        return math.radians(
            self.kinematics.wrist_flex_deg(
                math.degrees(joint2_rad),
                math.degrees(joint3_rad),
            )
        )

    def wrist_pitch_rad(
        self,
        status,
        joint2_rad,
        joint3_rad,
        fine_align_prep_joint4_rad=None,
    ):
        if status == "FINE_ALIGN_PREP" and fine_align_prep_joint4_rad is not None:
            return fine_align_prep_joint4_rad

        return self.horizontal_wrist_pitch_rad(
            joint2_rad,
            joint3_rad,
        ) + self.head_down_offset_rad(status)

    def search_goal_joint_degs(self, base_deg):
        joint4_deg = self.kinematics.wrist_flex_deg(
            self.config.search_joint2_deg,
            self.config.search_joint3_deg,
        ) + self.config.search_joint4_deg
        joint4_deg = clamp(
            joint4_deg,
            self.config.wrist_flex_min_deg,
            self.config.wrist_flex_max_deg,
        )
        return [
            base_deg,
            self.config.search_joint2_deg,
            self.config.search_joint3_deg,
            joint4_deg,
        ]

    def pose_from_joint_degs(self, joint_degs):
        return [
            math.radians(joint_degs[0]),
            math.radians(joint_degs[1]),
            math.radians(joint_degs[2]),
            math.radians(joint_degs[3]),
            math.radians(self.config.wrist_roll_deg),
            self.config.gripper_rad,
        ]

    def search_pose(self, base_deg):
        return self.pose_from_joint_degs(self.search_goal_joint_degs(base_deg))

    def fine_align_prep_pose(self, base_deg, j234_rad):
        return [
            math.radians(base_deg),
            j234_rad[0],
            j234_rad[1],
            j234_rad[2],
            math.radians(self.config.wrist_roll_deg),
            self.config.gripper_rad,
        ]

    def cylindrical_servo_pose(self, radius_m, theta_rad, z_m):
        solution = self.kinematics.cylindrical_ik(
            radius_m,
            math.degrees(theta_rad),
            z_m,
        )

        if solution is None:
            return PosePlan(
                positions=None,
                warning_key="cylindrical_servo_ik",
                warning_text=self.kinematics.last_error,
            )

        return PosePlan(
            positions=[
                math.radians(solution.base_deg),
                math.radians(solution.joint2_deg),
                math.radians(solution.joint3_deg),
                math.radians(solution.wrist_flex_deg),
                math.radians(self.config.wrist_roll_deg),
                self.config.gripper_rad,
            ],
            joint2_deg=solution.joint2_deg,
            joint3_deg=solution.joint3_deg,
        )

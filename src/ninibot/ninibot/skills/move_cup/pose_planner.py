import math
from dataclasses import dataclass

import numpy as np

from arm_controller.kinematics.cylindrical import CylindricalPose
from ninibot.skills.common.math_utils import GRIPPER_MAX_RAD
from ninibot.skills.common.math_utils import GRIPPER_MIN_RAD
from ninibot.skills.common.math_utils import clamp
from ninibot.skills.common.math_utils import gripper_percent_to_rad


def in_range(value, lower, upper):
    return lower <= value <= upper


def format_cm(xyz):
    if xyz is None:
        return "(n/a, n/a, n/a)"
    return (
        f"({xyz[0] * 100.0:+.1f}, "
        f"{xyz[1] * 100.0:+.1f}, "
        f"{xyz[2] * 100.0:+.1f})"
    )


@dataclass(frozen=True)
class MoveCupPosePlannerConfig:
    base_min_deg: float
    base_max_deg: float
    shoulder_min_deg: float
    shoulder_max_deg: float
    elbow_min_deg: float
    elbow_max_deg: float
    wrist_flex_min_deg: float
    wrist_flex_max_deg: float
    wrist_roll_deg: float
    hold_gripper_percent: float
    release_gripper_percent: float
    target_radius_margin_m: float
    target_search_theta_min_deg: float
    target_search_theta_max_deg: float
    search_joint2_deg: float
    search_joint3_deg: float
    search_joint4_deg: float
    return_joint2_deg: float
    return_joint3_deg: float
    return_search_prep_final_offset_deg: float


@dataclass(frozen=True)
class JointPosePlan:
    positions: list
    error_text: str = ""
    warning_text: str = ""


@dataclass(frozen=True)
class IkPlan:
    base_deg: float = 0.0
    joint2_deg: float = 0.0
    joint3_deg: float = 0.0
    error_text: str = ""


@dataclass(frozen=True)
class RadiusClampPlan:
    radius: float
    warning_text: str = ""


@dataclass(frozen=True)
class TargetJointClampPlan:
    joint_degs: list
    warning_text: str = ""


@dataclass(frozen=True)
class LockedTargetPlan:
    success: bool
    radius: float = 0.0
    theta_deg: float = 0.0
    z: float = 0.0
    target_base_xyz: tuple = ()
    warning_text: str = ""
    error_text: str = ""


class MoveCupPosePlanner:
    def __init__(self, config, kinematics):
        self.config = config
        self.kinematics = kinematics

    def clamp(self, value, lower, upper):
        return clamp(value, lower, upper)

    def in_range(self, value, lower, upper):
        return in_range(value, lower, upper)

    def clamp_wrist_flex_deg(self, joint4_deg):
        return clamp(
            float(joint4_deg),
            self.config.wrist_flex_min_deg,
            self.config.wrist_flex_max_deg,
        )

    def gripper_percent_to_rad(self, percent):
        return gripper_percent_to_rad(percent)

    def wrist_flex_deg(self, joint2_deg, joint3_deg):
        return self.kinematics.wrist_flex_deg(joint2_deg, joint3_deg)

    def current_pan_deg(self, latest_joint_positions, fallback_theta_deg):
        if latest_joint_positions is None:
            return self.kinematics.base_deg_from_theta(fallback_theta_deg)

        return math.degrees(latest_joint_positions[0])

    def current_theta_deg(self, latest_joint_positions, fallback_theta_deg):
        if latest_joint_positions is None:
            return fallback_theta_deg

        base_deg = math.degrees(latest_joint_positions[0])
        return self.kinematics.theta_deg_from_base(
            base_deg,
            fallback_deg=fallback_theta_deg,
        )

    def current_cylindrical_pose(self, latest_joint_positions):
        if latest_joint_positions is None:
            return None

        shoulder_deg = math.degrees(latest_joint_positions[1])
        elbow_deg = math.degrees(latest_joint_positions[2])
        return self.kinematics.cylindrical_from_joint_degs(
            math.degrees(latest_joint_positions[0]),
            shoulder_deg,
            elbow_deg,
        )

    def max_reachable_radius_for_z(self, z):
        planar_z = float(z) + self.kinematics.z_tool_offset
        max_wrist_radius = self.kinematics.l1 + self.kinematics.l2
        remaining = max_wrist_radius * max_wrist_radius - planar_z * planar_z

        if remaining <= 0.0:
            return None

        return self.kinematics.horizontal_tool_offset + math.sqrt(remaining)

    def clamp_radius_for_reach_margin(self, radius, z, context):
        if self.config.target_radius_margin_m <= 0.0:
            return RadiusClampPlan(radius=float(radius))

        max_radius = self.max_reachable_radius_for_z(z)
        if max_radius is None:
            return RadiusClampPlan(radius=float(radius))

        safe_radius = max(1e-6, max_radius - self.config.target_radius_margin_m)
        if radius <= safe_radius:
            return RadiusClampPlan(radius=float(radius))

        return RadiusClampPlan(
            radius=safe_radius,
            warning_text=(
                f"move cup: {context} radius clamped from {radius:.3f}m "
                f"to {safe_radius:.3f}m for "
                f"{self.config.target_radius_margin_m:.3f}m reach margin "
                f"at z={z:.3f}m"
            ),
        )

    def make_joint_pose(
        self,
        pan_deg,
        joint2_deg,
        joint3_deg,
        joint4_deg,
        gripper_percent,
    ):
        checks = (
            (pan_deg, self.config.base_min_deg, self.config.base_max_deg, "base"),
            (
                joint2_deg,
                self.config.shoulder_min_deg,
                self.config.shoulder_max_deg,
                "shoulder",
            ),
            (
                joint3_deg,
                self.config.elbow_min_deg,
                self.config.elbow_max_deg,
                "elbow",
            ),
            (
                joint4_deg,
                self.config.wrist_flex_min_deg,
                self.config.wrist_flex_max_deg,
                "wrist",
            ),
        )
        for value, lower, upper, name in checks:
            if not in_range(value, lower, upper):
                return JointPosePlan(
                    positions=None,
                    error_text=f"Move cup {name} out of range: {value:.1f} deg",
                )

        return JointPosePlan(
            positions=[
                math.radians(pan_deg),
                math.radians(joint2_deg),
                math.radians(joint3_deg),
                math.radians(joint4_deg),
                math.radians(self.config.wrist_roll_deg),
                gripper_percent_to_rad(gripper_percent),
            ]
        )

    def make_pose(self, pan_deg, joint2_deg, joint3_deg, gripper_percent):
        joint4_deg = self.wrist_flex_deg(joint2_deg, joint3_deg)
        return self.make_joint_pose(
            pan_deg,
            joint2_deg,
            joint3_deg,
            joint4_deg,
            gripper_percent,
        )

    def cylindrical_ik(self, radius, theta_deg, z):
        solution = self.kinematics.cylindrical_ik(radius, theta_deg, z)
        if solution is None:
            return IkPlan(error_text=f"Move cup IK failed: {self.kinematics.last_error}")

        return IkPlan(
            base_deg=solution.base_deg,
            joint2_deg=solution.joint2_deg,
            joint3_deg=solution.joint3_deg,
        )

    def make_cylindrical_pose(self, radius, theta_deg, z, gripper_percent):
        ik_plan = self.cylindrical_ik(radius, theta_deg, z)
        if ik_plan.error_text:
            return JointPosePlan(positions=None, error_text=ik_plan.error_text)

        return self.make_pose(
            ik_plan.base_deg,
            ik_plan.joint2_deg,
            ik_plan.joint3_deg,
            gripper_percent,
        )

    def make_pose_from_cylindrical(self, pose, gripper_percent):
        return self.make_cylindrical_pose(
            pose.radius,
            pose.theta_deg,
            pose.z,
            gripper_percent,
        )

    def make_grip_pose_from_current(self, latest_joint_positions):
        pose = self.current_cylindrical_pose(latest_joint_positions)
        if pose is None:
            return JointPosePlan(positions=None)

        return self.make_pose_from_cylindrical(
            pose,
            self.config.hold_gripper_percent,
        )

    def make_return_pose(self, current_pan_deg):
        joint4_deg = self.clamp_wrist_flex_deg(
            self.wrist_flex_deg(
                self.config.return_joint2_deg,
                self.config.return_joint3_deg,
            )
            + self.config.return_search_prep_final_offset_deg
        )
        return self.make_joint_pose(
            current_pan_deg,
            self.config.return_joint2_deg,
            self.config.return_joint3_deg,
            joint4_deg,
            self.config.release_gripper_percent,
        )

    def clamp_target_search_theta(self, theta_deg):
        return clamp(
            theta_deg,
            self.config.target_search_theta_min_deg,
            self.config.target_search_theta_max_deg,
        )

    def target_search_goal_joint_degs(self, theta_deg):
        base_deg = self.kinematics.base_deg_from_theta(theta_deg)
        joint4_deg = self.clamp_wrist_flex_deg(
            self.wrist_flex_deg(
                self.config.search_joint2_deg,
                self.config.search_joint3_deg,
            )
            + self.config.search_joint4_deg
        )
        return [
            base_deg,
            self.config.search_joint2_deg,
            self.config.search_joint3_deg,
            joint4_deg,
        ]

    def clamp_target_joint_degs(self, joint_degs, context=""):
        limits = [
            (self.config.base_min_deg, self.config.base_max_deg, "base"),
            (self.config.shoulder_min_deg, self.config.shoulder_max_deg, "shoulder"),
            (self.config.elbow_min_deg, self.config.elbow_max_deg, "elbow"),
            (self.config.wrist_flex_min_deg, self.config.wrist_flex_max_deg, "wrist"),
        ]
        clamped = []
        changes = []
        for value, (lower, upper, name) in zip(joint_degs, limits):
            clamped_value = clamp(value, lower, upper)
            clamped.append(clamped_value)
            if abs(clamped_value - value) > 1e-6:
                changes.append(f"{name} {value:.1f}->{clamped_value:.1f}deg")

        warning_text = ""
        if changes and context:
            warning_text = f"move cup: {context} target clamped: " + ", ".join(changes)

        return TargetJointClampPlan(joint_degs=clamped, warning_text=warning_text)

    def make_target_search_pose_from_joint_degs(
        self,
        joint_degs,
        clamp_to_limits=False,
        context="",
    ):
        warning_text = ""
        if clamp_to_limits:
            clamp_plan = self.clamp_target_joint_degs(joint_degs, context)
            joint_degs = clamp_plan.joint_degs
            warning_text = clamp_plan.warning_text

        pose_plan = self.make_joint_pose(
            joint_degs[0],
            joint_degs[1],
            joint_degs[2],
            joint_degs[3],
            self.config.release_gripper_percent,
        )
        return JointPosePlan(
            positions=pose_plan.positions,
            error_text=pose_plan.error_text,
            warning_text=warning_text,
        )

    def make_target_search_pose(self, theta_deg):
        return self.make_target_search_pose_from_joint_degs(
            self.target_search_goal_joint_degs(theta_deg)
        )

    def locked_target_from_base_xyz(self, target_base_xyz, release_z):
        target_base_xyz = np.asarray(target_base_xyz, dtype=float).copy()
        target_base_xyz[2] = float(release_z)

        radius = math.hypot(float(target_base_xyz[0]), float(target_base_xyz[1]))
        if radius <= 1e-6:
            return LockedTargetPlan(
                success=False,
                target_base_xyz=tuple(target_base_xyz),
                error_text=f"invalid target radius from {format_cm(target_base_xyz)}",
            )

        theta_deg = math.degrees(
            math.atan2(float(target_base_xyz[1]), float(target_base_xyz[0]))
        )
        target_z = float(target_base_xyz[2])
        radius_plan = self.clamp_radius_for_reach_margin(
            radius,
            target_z,
            "locked target",
        )
        radius = radius_plan.radius

        if self.kinematics.cylindrical_ik(radius, theta_deg, target_z) is None:
            return LockedTargetPlan(
                success=False,
                target_base_xyz=tuple(target_base_xyz),
                warning_text=radius_plan.warning_text,
                error_text=(
                    "release target unreachable "
                    f"target={format_cm(target_base_xyz)} "
                    f"radius={radius:.3f}, theta={theta_deg:.1f}, "
                    f"z={target_z:.3f}; {self.kinematics.last_error}"
                ),
            )

        target_theta_rad = math.radians(theta_deg)
        effective_target_base_xyz = (
            radius * math.cos(target_theta_rad),
            radius * math.sin(target_theta_rad),
            target_z,
        )
        return LockedTargetPlan(
            success=True,
            radius=radius,
            theta_deg=theta_deg,
            z=target_z,
            target_base_xyz=effective_target_base_xyz,
            warning_text=radius_plan.warning_text,
        )

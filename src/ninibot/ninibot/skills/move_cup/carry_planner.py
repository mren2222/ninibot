from dataclasses import dataclass

from arm_controller.kinematics.cylindrical import CylindricalPose


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def safe_pre_pan_radius(current_radius, pre_pan_radius):
    if pre_pan_radius <= 0.0:
        return current_radius

    return min(current_radius, pre_pan_radius)


def retract_radius_pose(start_pose, pre_pan_radius):
    return CylindricalPose(
        radius=safe_pre_pan_radius(start_pose.radius, pre_pan_radius),
        theta_deg=start_pose.theta_deg,
        z=start_pose.z,
    )


def pan_at_current_radius_pose(start_pose, target_theta_deg, z=None):
    return CylindricalPose(
        radius=start_pose.radius,
        theta_deg=target_theta_deg,
        z=start_pose.z if z is None else z,
    )


def radius_at_current_theta_pose(start_pose, target_radius, z=None):
    return CylindricalPose(
        radius=target_radius,
        theta_deg=start_pose.theta_deg,
        z=start_pose.z if z is None else z,
    )


def lower_at_current_radius_pose(start_pose, target_z):
    return CylindricalPose(
        radius=start_pose.radius,
        theta_deg=start_pose.theta_deg,
        z=target_z,
    )


def interpolate_cylindrical_pose(start_pose, goal_pose, alpha):
    alpha = clamp(float(alpha), 0.0, 1.0)
    return CylindricalPose(
        radius=start_pose.radius + alpha * (goal_pose.radius - start_pose.radius),
        theta_deg=start_pose.theta_deg
        + alpha * (goal_pose.theta_deg - start_pose.theta_deg),
        z=start_pose.z + alpha * (goal_pose.z - start_pose.z),
    )


def interpolate_gripper_percent(start_percent, goal_percent, alpha):
    alpha = clamp(float(alpha), 0.0, 1.0)
    return float(start_percent) + alpha * (float(goal_percent) - float(start_percent))


@dataclass(frozen=True)
class MoveCupCarryPlannerConfig:
    lift_z_offset: float
    pre_pan_radius: float
    post_release_clearance_retract_m: float
    base_min_deg: float
    base_max_deg: float


@dataclass(frozen=True)
class CarryPosePlan:
    pose: CylindricalPose
    warning_text: str = ""


class MoveCupCarryPlanner:
    def __init__(self, config, kinematics):
        self.config = config
        self.kinematics = kinematics

    def lift_goal(self, start_pose):
        return CylindricalPose(
            radius=start_pose.radius,
            theta_deg=start_pose.theta_deg,
            z=start_pose.z + self.config.lift_z_offset,
        )

    def retract_radius_goal(self, start_pose):
        return retract_radius_pose(start_pose, self.config.pre_pan_radius)

    def horizontal_move_goal(self, start_pose, target_theta_deg, carry_z):
        return pan_at_current_radius_pose(
            start_pose,
            target_theta_deg,
            carry_z,
        )

    def extend_radius_goal(self, start_pose, target_radius, carry_z):
        return radius_at_current_theta_pose(
            start_pose,
            target_radius,
            carry_z,
        )

    def lower_goal(self, start_pose):
        return lower_at_current_radius_pose(
            start_pose,
            start_pose.z - self.config.lift_z_offset,
        )

    def release_goal(self, start_pose, base_pan_offset_deg):
        start_base_deg = self.kinematics.base_deg_from_theta(start_pose.theta_deg)
        requested_base_deg = start_base_deg + base_pan_offset_deg
        goal_base_deg = clamp(
            requested_base_deg,
            self.config.base_min_deg,
            self.config.base_max_deg,
        )
        warning_text = ""
        if abs(goal_base_deg - requested_base_deg) > 1e-6:
            warning_text = (
                "move cup: base pan clearance clamped "
                f"from {requested_base_deg:.1f}deg to {goal_base_deg:.1f}deg"
            )

        return CarryPosePlan(
            pose=CylindricalPose(
                radius=start_pose.radius,
                theta_deg=self.kinematics.theta_deg_from_base(
                    goal_base_deg,
                    fallback_deg=start_pose.theta_deg,
                ),
                z=start_pose.z,
            ),
            warning_text=warning_text,
        )

    def post_release_clearance_goal(self, start_pose):
        return CylindricalPose(
            radius=max(
                0.0,
                start_pose.radius - self.config.post_release_clearance_retract_m,
            ),
            theta_deg=start_pose.theta_deg,
            z=start_pose.z,
        )

    def path_pose_at(self, start_pose, goal_pose, alpha):
        return interpolate_cylindrical_pose(start_pose, goal_pose, alpha)

    def gripper_percent_at(self, start_percent, goal_percent, alpha):
        return interpolate_gripper_percent(start_percent, goal_percent, alpha)

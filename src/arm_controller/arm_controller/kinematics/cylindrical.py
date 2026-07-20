# flake8: noqa
import math
from dataclasses import dataclass


@dataclass
class CylindricalPose:
    radius: float
    theta_deg: float
    z: float


@dataclass
class JointSolution:
    base_deg: float
    joint2_deg: float
    joint3_deg: float
    wrist_flex_deg: float


class CylindricalArmKinematics:
    def __init__(
        self,
        l1,
        l2,
        horizontal_tool_offset,
        z_tool_offset,
        theta_joint_sign,
        theta_joint_offset_deg,
        base_min_deg,
        base_max_deg,
        shoulder_min_deg,
        shoulder_max_deg,
        elbow_min_deg,
        elbow_max_deg,
        wrist_flex_min_deg,
        wrist_flex_max_deg,
        horizontal_wrist_pitch_offset_deg,
    ):
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.horizontal_tool_offset = float(horizontal_tool_offset)
        self.z_tool_offset = float(z_tool_offset)
        self.theta_joint_sign = float(theta_joint_sign)
        self.theta_joint_offset_deg = float(theta_joint_offset_deg)
        self.base_min_deg = float(base_min_deg)
        self.base_max_deg = float(base_max_deg)
        self.shoulder_min_deg = float(shoulder_min_deg)
        self.shoulder_max_deg = float(shoulder_max_deg)
        self.elbow_min_deg = float(elbow_min_deg)
        self.elbow_max_deg = float(elbow_max_deg)
        self.wrist_flex_min_deg = float(wrist_flex_min_deg)
        self.wrist_flex_max_deg = float(wrist_flex_max_deg)
        self.horizontal_wrist_pitch_offset_deg = float(
            horizontal_wrist_pitch_offset_deg
        )
        self.last_error = ""

    def fail(self, text):
        self.last_error = text
        return None

    def in_range(self, value, lower, upper):
        return lower <= value <= upper

    def joint_angle_offsets(self):
        theta1_offset = math.atan2(0.028, 0.11257)
        theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset
        return theta1_offset, theta2_offset

    def base_deg_from_theta(self, theta_deg):
        return self.theta_joint_sign * theta_deg + self.theta_joint_offset_deg

    def theta_deg_from_base(self, base_deg, fallback_deg=0.0):
        if abs(self.theta_joint_sign) < 1e-6:
            return fallback_deg

        return (base_deg - self.theta_joint_offset_deg) / self.theta_joint_sign

    def wrist_flex_deg(self, joint2_deg, joint3_deg):
        return -joint2_deg - joint3_deg + self.horizontal_wrist_pitch_offset_deg

    def wrist_planar_from_joint_degs(self, shoulder_deg, elbow_deg):
        theta1_offset, theta2_offset = self.joint_angle_offsets()

        joint2_internal = math.radians(90.0 - shoulder_deg)
        joint3_internal = math.radians(elbow_deg + 90.0)
        theta1 = joint2_internal - theta1_offset
        theta2 = joint3_internal - theta2_offset

        wrist_z = (
            self.l1 * math.sin(theta1)
            + self.l2 * math.sin(theta1 - theta2)
        )
        wrist_radius = (
            self.l1 * math.cos(theta1)
            + self.l2 * math.cos(theta1 - theta2)
        )
        return wrist_radius, wrist_z

    def cylindrical_from_joint_degs(self, base_deg, shoulder_deg, elbow_deg):
        wrist_radius, wrist_z = self.wrist_planar_from_joint_degs(
            shoulder_deg,
            elbow_deg,
        )
        return CylindricalPose(
            radius=wrist_radius + self.horizontal_tool_offset,
            theta_deg=self.theta_deg_from_base(base_deg),
            z=wrist_z - self.z_tool_offset,
        )

    def joint23_ik(self, radius, z):
        planar_x = radius - self.horizontal_tool_offset
        planar_z = z + self.z_tool_offset

        if planar_x <= 0.0:
            return self.fail(
                f"cylindrical target invalid: radius={radius:.3f}, "
                f"tool_offset={self.horizontal_tool_offset:.3f}"
            )

        theta1_offset, theta2_offset = self.joint_angle_offsets()
        wrist_r = math.sqrt(planar_x * planar_x + planar_z * planar_z)
        r_max = self.l1 + self.l2
        r_min = abs(self.l1 - self.l2)

        if wrist_r > r_max or wrist_r < r_min:
            return self.fail(
                f"cylindrical target unreachable: wrist_r={wrist_r:.3f}, "
                f"allowed=[{r_min:.3f}, {r_max:.3f}]"
            )

        cos_theta2 = -(
            wrist_r * wrist_r
            - self.l1 * self.l1
            - self.l2 * self.l2
        ) / (2.0 * self.l1 * self.l2)
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))

        theta2 = math.pi - math.acos(cos_theta2)
        beta = math.atan2(planar_z, planar_x)
        gamma = math.atan2(
            self.l2 * math.sin(theta2),
            self.l1 + self.l2 * math.cos(theta2),
        )
        theta1 = beta + gamma

        joint2 = theta1 + theta1_offset
        joint3 = theta2 + theta2_offset
        joint2_deg = 90.0 - math.degrees(joint2)
        joint3_deg = math.degrees(joint3) - 90.0

        if not self.in_range(joint2_deg, self.shoulder_min_deg, self.shoulder_max_deg):
            return self.fail(
                f"shoulder out of range: {joint2_deg:.1f} deg"
            )

        if not self.in_range(joint3_deg, self.elbow_min_deg, self.elbow_max_deg):
            return self.fail(
                f"elbow out of range: {joint3_deg:.1f} deg"
            )

        self.last_error = ""
        return joint2_deg, joint3_deg

    def cylindrical_ik(self, radius, theta_deg, z):
        base_deg = self.base_deg_from_theta(theta_deg)

        if not self.in_range(base_deg, self.base_min_deg, self.base_max_deg):
            return self.fail(f"base out of range: {base_deg:.1f} deg")

        joint23 = self.joint23_ik(radius, z)

        if joint23 is None:
            return None

        joint2_deg, joint3_deg = joint23
        wrist_flex_deg = self.wrist_flex_deg(joint2_deg, joint3_deg)

        if not self.in_range(
            wrist_flex_deg,
            self.wrist_flex_min_deg,
            self.wrist_flex_max_deg,
        ):
            return self.fail(
                f"wrist flex out of range: {wrist_flex_deg:.1f} deg"
            )

        self.last_error = ""
        return JointSolution(
            base_deg=base_deg,
            joint2_deg=joint2_deg,
            joint3_deg=joint3_deg,
            wrist_flex_deg=wrist_flex_deg,
        )

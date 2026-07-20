import math
from dataclasses import dataclass

from ninibot.trajectories.prep_trajectory import load_processed_template
from ninibot.trajectories.prep_trajectory import retarget_sample


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def load_template(path):
    return load_processed_template(path)


@dataclass(frozen=True)
class FineAlignPrepConfig:
    duration_sec: float
    shape_scale: float
    z_offset_m: float
    wrist_flex_min_deg: float
    wrist_flex_max_deg: float


@dataclass(frozen=True)
class FineAlignPrepEnd:
    j234_rad: list
    warning_key: str = ""
    warning_text: str = ""


@dataclass(frozen=True)
class FineAlignPrepTrajectory:
    start_j234: list
    end_j234: list
    started_at: float


class FineAlignPrepPlanner:
    def __init__(self, template, kinematics, config):
        self.template = template
        self.kinematics = kinematics
        self.config = config

    def end_j234(self):
        if self.template is None:
            raise RuntimeError("Fine-align-prep trajectory template is not loaded")

        values = self.template.get("processed_end_rad")
        if not isinstance(values, list) or len(values) < 2:
            raise RuntimeError(
                "Fine-align-prep trajectory must contain processed_end_rad"
            )

        joint2_rad = float(values[0])
        joint3_rad = float(values[1])
        warning_key = ""
        warning_text = ""

        if abs(self.config.z_offset_m) > 1e-6:
            original_pose = self.kinematics.cylindrical_from_joint_degs(
                0.0,
                math.degrees(joint2_rad),
                math.degrees(joint3_rad),
            )
            adjusted_joint23 = self.kinematics.joint23_ik(
                original_pose.radius,
                original_pose.z + self.config.z_offset_m,
            )
            if adjusted_joint23 is None:
                warning_key = "fine_align_prep_z_offset_ik"
                warning_text = (
                    "Cannot apply fine-align-prep z offset "
                    f"{self.config.z_offset_m:.3f}m: "
                    f"{self.kinematics.last_error}"
                )
            else:
                candidate_joint4_deg = self.kinematics.wrist_flex_deg(
                    adjusted_joint23[0],
                    adjusted_joint23[1],
                )
                if (
                    self.config.wrist_flex_min_deg
                    <= candidate_joint4_deg
                    <= self.config.wrist_flex_max_deg
                ):
                    joint2_rad = math.radians(adjusted_joint23[0])
                    joint3_rad = math.radians(adjusted_joint23[1])
                else:
                    warning_key = "fine_align_prep_z_offset_wrist"
                    warning_text = (
                        "Cannot apply fine-align-prep z offset "
                        f"{self.config.z_offset_m:.3f}m: "
                        f"wrist flex out of range {candidate_joint4_deg:.1f} deg"
                    )

        joint4_rad = math.radians(
            self.kinematics.wrist_flex_deg(
                math.degrees(joint2_rad),
                math.degrees(joint3_rad),
            )
        )
        return FineAlignPrepEnd(
            j234_rad=[joint2_rad, joint3_rad, joint4_rad],
            warning_key=warning_key,
            warning_text=warning_text,
        )

    def start(self, start_j234, now_sec, end_j234=None):
        if end_j234 is None:
            end_j234 = self.end_j234().j234_rad

        return FineAlignPrepTrajectory(
            start_j234=[float(value) for value in start_j234],
            end_j234=[float(value) for value in end_j234],
            started_at=float(now_sec),
        )

    def duration_sec(self):
        if self.config.duration_sec > 0.0:
            return self.config.duration_sec

        return float(self.template["duration_sec"])

    def progress(self, trajectory, now_sec):
        duration = self.duration_sec()
        elapsed = float(now_sec) - trajectory.started_at
        return clamp(elapsed / max(duration, 1e-9), 0.0, 1.0)

    def sample_j234(self, trajectory, now_sec):
        s = self.progress(trajectory, now_sec)
        j234 = retarget_sample(
            self.template,
            trajectory.start_j234,
            trajectory.end_j234,
            s,
            shape_scale=self.config.shape_scale,
        )
        return [float(value) for value in j234]

    def pose(self, base_deg, trajectory, now_sec, wrist_roll_deg, gripper_rad):
        j234 = self.sample_j234(trajectory, now_sec)
        return [
            math.radians(base_deg),
            j234[0],
            j234[1],
            j234[2],
            math.radians(wrist_roll_deg),
            gripper_rad,
        ]

    def ready(self, trajectory, latest_joint_positions, joint_goal_tolerance, now_sec):
        if trajectory is None or latest_joint_positions is None:
            return False

        if self.progress(trajectory, now_sec) < 1.0:
            return False

        for joint_index, target in zip((1, 2, 3), trajectory.end_j234):
            if abs(latest_joint_positions[joint_index] - target) > joint_goal_tolerance:
                return False

        return True

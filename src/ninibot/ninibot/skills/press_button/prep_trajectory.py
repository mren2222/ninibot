import math
from dataclasses import dataclass

from ninibot.skills.press_button.math_utils import clamp
from ninibot.skills.press_button.math_utils import minimum_jerk
from ninibot.skills.press_button.params import PrepTrajectoryConfig


@dataclass(frozen=True)
class PrepPan:
    value_deg: float
    clamped_from_deg: float | None = None


@dataclass(frozen=True)
class PrepTrajectory:
    start: list
    goal: list
    duration_sec: float
    j2_transition_sec: float


def joint_pitch_deg(pose):
    return sum(math.degrees(value) for value in pose[1:4])


class PrepTrajectoryPlanner:
    def __init__(self, config: PrepTrajectoryConfig):
        self.config = config

    def current_start_pose(self, latest_joint_positions):
        if latest_joint_positions is None:
            return None
        return [
            latest_joint_positions[0],
            latest_joint_positions[1],
            latest_joint_positions[2],
            latest_joint_positions[3],
            latest_joint_positions[5],
        ]

    def pan_from_current(self, latest_joint_positions):
        if latest_joint_positions is None:
            return None
        pan_deg = math.degrees(latest_joint_positions[0])
        clamped = clamp(
            pan_deg,
            self.config.search_pan_min_deg,
            self.config.search_pan_max_deg,
        )
        clamped_from = pan_deg if abs(clamped - pan_deg) > 1e-6 else None
        return PrepPan(clamped, clamped_from)

    def goal_from_current_pan(self, latest_joint_positions):
        pan = self.pan_from_current(latest_joint_positions)
        if pan is None:
            return None
        return [
            math.radians(pan.value_deg),
            math.radians(self.config.prep_joint2_deg),
            math.radians(self.config.prep_joint3_deg),
            math.radians(self.config.prep_joint4_deg),
            self.config.joint6_rad,
        ]

    def make_trajectory(self, latest_joint_positions):
        start = self.current_start_pose(latest_joint_positions)
        goal = self.goal_from_current_pan(latest_joint_positions)
        if start is None or goal is None:
            return None
        return PrepTrajectory(
            start=list(start),
            goal=list(goal),
            duration_sec=max(0.0, float(self.config.duration_sec)),
            j2_transition_sec=min(
                max(0.0, float(self.config.j2_transition_sec)),
                max(0.0, float(self.config.duration_sec)),
            ),
        )

    def sample(self, trajectory: PrepTrajectory, elapsed_sec):
        if trajectory.duration_sec <= 1e-6:
            return list(trajectory.goal)

        duration_sec = max(1e-6, trajectory.duration_sec)
        t = clamp(float(elapsed_sec), 0.0, duration_sec)
        alpha = minimum_jerk(t / duration_sec)
        start_deg = [math.degrees(value) for value in trajectory.start[:4]]
        goal_deg = [math.degrees(value) for value in trajectory.goal[:4]]
        start_pitch_deg = sum(start_deg[1:4])
        goal_pitch_deg = sum(goal_deg[1:4])

        joint1_deg = start_deg[0] + (goal_deg[0] - start_deg[0]) * alpha
        joint3_deg = start_deg[2] + (goal_deg[2] - start_deg[2]) * alpha
        pitch_deg = start_pitch_deg + (goal_pitch_deg - start_pitch_deg) * alpha

        transition_sec = min(trajectory.j2_transition_sec, duration_sec)
        if transition_sec <= 1e-6:
            joint2_deg = goal_deg[1]
        else:
            j2_start_sec = max(0.0, duration_sec - transition_sec)
            if t < j2_start_sec:
                joint2_deg = start_deg[1]
            else:
                j2_alpha = minimum_jerk((t - j2_start_sec) / transition_sec)
                joint2_deg = start_deg[1] + (goal_deg[1] - start_deg[1]) * j2_alpha

        joint4_deg = pitch_deg - joint2_deg - joint3_deg
        joint4_deg = clamp(
            joint4_deg,
            math.degrees(self.config.joint4_lower_rad),
            math.degrees(self.config.joint4_upper_rad),
        )
        gripper_rad = trajectory.start[4] + (
            trajectory.goal[4] - trajectory.start[4]
        ) * alpha
        return [
            math.radians(joint1_deg),
            math.radians(joint2_deg),
            math.radians(joint3_deg),
            math.radians(joint4_deg),
            gripper_rad,
        ]


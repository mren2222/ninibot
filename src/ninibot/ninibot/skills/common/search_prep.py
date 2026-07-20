from dataclasses import dataclass
import math

from ninibot.skills.common.math_utils import clamp
from ninibot.skills.common.math_utils import minimum_jerk


@dataclass(frozen=True)
class SearchPrepConfig:
    min_duration_sec: float
    head_down_duration_sec: float
    duration_scale: float
    trajectory_samples: int
    max_vel_deg: tuple
    max_acc_deg: tuple
    final_wrist_offset_deg: float
    wrist_flex_min_deg: float
    wrist_flex_max_deg: float

    def __post_init__(self):
        max_vel_deg = tuple(float(value) for value in self.max_vel_deg)
        max_acc_deg = tuple(float(value) for value in self.max_acc_deg)
        if len(max_vel_deg) < 4 or len(max_acc_deg) < 4:
            raise ValueError(
                "SearchPrepConfig requires four velocity and acceleration limits"
            )

        object.__setattr__(self, "max_vel_deg", max_vel_deg)
        object.__setattr__(self, "max_acc_deg", max_acc_deg)
        object.__setattr__(
            self,
            "min_duration_sec",
            max(1e-6, float(self.min_duration_sec)),
        )
        object.__setattr__(
            self,
            "head_down_duration_sec",
            max(1e-6, float(self.head_down_duration_sec)),
        )
        object.__setattr__(
            self,
            "duration_scale",
            max(1.0, float(self.duration_scale)),
        )
        object.__setattr__(
            self,
            "trajectory_samples",
            max(1, int(self.trajectory_samples)),
        )


@dataclass(frozen=True)
class SearchPrepTrajectory:
    start_deg: list
    goal_deg: list
    start_wrist_offset_deg: float
    final_wrist_offset_deg: float
    duration_sec: float
    head_down_duration_sec: float
    head_down_start_sec: float


class SearchPrepPlanner:
    def __init__(self, config, wrist_flex_deg):
        self.config = config
        self.wrist_flex_deg = wrist_flex_deg

    def make_trajectory(self, start_deg, goal_deg, start_wrist_offset_deg):
        start_deg = [float(value) for value in start_deg[:4]]
        goal_deg = [float(value) for value in goal_deg[:4]]
        if len(start_deg) < 4 or len(goal_deg) < 4:
            raise ValueError("Search prep start and goal must contain four joints")

        duration_sec = self.duration_sec(
            start_deg,
            goal_deg,
            start_wrist_offset_deg,
        )
        head_down_duration_sec = min(
            self.config.head_down_duration_sec,
            duration_sec,
        )
        head_down_start_sec = max(0.0, duration_sec - head_down_duration_sec)
        return SearchPrepTrajectory(
            start_deg=start_deg,
            goal_deg=goal_deg,
            start_wrist_offset_deg=float(start_wrist_offset_deg),
            final_wrist_offset_deg=self.config.final_wrist_offset_deg,
            duration_sec=duration_sec,
            head_down_duration_sec=head_down_duration_sec,
            head_down_start_sec=head_down_start_sec,
        )

    @staticmethod
    def minimum_jerk_duration_sec(delta_deg, max_vel_deg, max_acc_deg):
        delta_deg = abs(float(delta_deg))
        if delta_deg <= 1e-6:
            return 0.0

        t_vel = 0.0
        if max_vel_deg > 1e-6:
            t_vel = 1.875 * delta_deg / max_vel_deg

        t_acc = 0.0
        if max_acc_deg > 1e-6:
            t_acc = math.sqrt(5.774 * delta_deg / max_acc_deg)

        return max(t_vel, t_acc)

    def duration_sec(self, start_deg, goal_deg, start_wrist_offset_deg):
        duration_sec = 0.0
        for index in range(3):
            duration_sec = max(
                duration_sec,
                self.minimum_jerk_duration_sec(
                    goal_deg[index] - start_deg[index],
                    self.config.max_vel_deg[index],
                    self.config.max_acc_deg[index],
                ),
            )

        duration_sec = max(
            self.config.min_duration_sec,
            self.config.head_down_duration_sec,
            duration_sec * self.config.duration_scale,
        )

        for _ in range(6):
            max_vel_seen, max_acc_seen = self.sample_limits(
                start_deg,
                goal_deg,
                start_wrist_offset_deg,
                duration_sec,
            )
            vel_ratio = max(
                (
                    max_vel_seen[index] / self.config.max_vel_deg[index]
                    if self.config.max_vel_deg[index] > 1e-6
                    else 0.0
                )
                for index in range(4)
            )
            acc_ratio = max(
                (
                    max_acc_seen[index] / self.config.max_acc_deg[index]
                    if self.config.max_acc_deg[index] > 1e-6
                    else 0.0
                )
                for index in range(4)
            )
            if vel_ratio <= 1.0 and acc_ratio <= 1.0:
                break

            duration_sec *= max(vel_ratio, math.sqrt(acc_ratio)) * 1.05

        return duration_sec

    def sample_limits(
        self,
        start_deg,
        goal_deg,
        start_wrist_offset_deg,
        duration_sec,
    ):
        head_down_duration_sec = min(
            self.config.head_down_duration_sec,
            duration_sec,
        )
        trajectory = SearchPrepTrajectory(
            start_deg=start_deg,
            goal_deg=goal_deg,
            start_wrist_offset_deg=float(start_wrist_offset_deg),
            final_wrist_offset_deg=self.config.final_wrist_offset_deg,
            duration_sec=duration_sec,
            head_down_duration_sec=head_down_duration_sec,
            head_down_start_sec=max(0.0, duration_sec - head_down_duration_sec),
        )
        samples = []
        for index in range(self.config.trajectory_samples + 1):
            t = duration_sec * index / self.config.trajectory_samples
            samples.append(self.joint_degs_at(t, trajectory))

        dt = duration_sec / self.config.trajectory_samples
        max_vel_seen = [0.0, 0.0, 0.0, 0.0]
        velocities = []
        for previous, current in zip(samples[:-1], samples[1:]):
            velocity = [
                abs(current[index] - previous[index]) / dt
                for index in range(4)
            ]
            velocities.append(velocity)
            for index in range(4):
                max_vel_seen[index] = max(max_vel_seen[index], velocity[index])

        max_acc_seen = [0.0, 0.0, 0.0, 0.0]
        for previous, current in zip(velocities[:-1], velocities[1:]):
            for index in range(4):
                accel = abs(current[index] - previous[index]) / dt
                max_acc_seen[index] = max(max_acc_seen[index], accel)

        return max_vel_seen, max_acc_seen

    def joint_degs_at(self, elapsed_sec, trajectory):
        duration_sec = max(1e-6, trajectory.duration_sec)
        t = clamp(elapsed_sec, 0.0, duration_sec)
        s = minimum_jerk(t / duration_sec)

        joint1_deg = trajectory.start_deg[0] + (
            trajectory.goal_deg[0] - trajectory.start_deg[0]
        ) * s
        joint2_deg = trajectory.start_deg[1] + (
            trajectory.goal_deg[1] - trajectory.start_deg[1]
        ) * s
        joint3_deg = trajectory.start_deg[2] + (
            trajectory.goal_deg[2] - trajectory.start_deg[2]
        ) * s
        wrist_offset_deg = self.wrist_offset_at(t, trajectory)
        joint4_deg = self.wrist_flex_deg(joint2_deg, joint3_deg) + wrist_offset_deg
        joint4_deg = clamp(
            joint4_deg,
            self.config.wrist_flex_min_deg,
            self.config.wrist_flex_max_deg,
        )
        return [joint1_deg, joint2_deg, joint3_deg, joint4_deg]

    def wrist_offset_at(self, elapsed_sec, trajectory):
        duration_sec = max(1e-6, trajectory.duration_sec)
        t = clamp(elapsed_sec, 0.0, duration_sec)
        head_down_start_sec = trajectory.head_down_start_sec
        head_down_duration_sec = max(1e-6, trajectory.head_down_duration_sec)

        if head_down_start_sec <= 1e-6:
            u = minimum_jerk(t / duration_sec)
            return trajectory.start_wrist_offset_deg + (
                trajectory.final_wrist_offset_deg - trajectory.start_wrist_offset_deg
            ) * u

        if t < head_down_start_sec:
            u = minimum_jerk(t / head_down_start_sec)
            return trajectory.start_wrist_offset_deg * (1.0 - u)

        u = minimum_jerk((t - head_down_start_sec) / head_down_duration_sec)
        return trajectory.final_wrist_offset_deg * u

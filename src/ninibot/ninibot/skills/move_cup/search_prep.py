from dataclasses import dataclass

from ninibot.skills.common.search_prep import SearchPrepConfig
from ninibot.skills.common.search_prep import SearchPrepPlanner
from ninibot.skills.common.search_prep import SearchPrepTrajectory
from ninibot.skills.common.search_prep import minimum_jerk


@dataclass(frozen=True)
class MoveCupSearchPrepTrajectory:
    start_deg: list
    goal_deg: list
    start_wrist_offset_deg: float
    final_wrist_offset_deg: float
    target_theta_deg: float
    duration_sec: float
    head_down_duration_sec: float
    head_down_start_sec: float


class MoveCupSearchPrepPlanner:
    def __init__(self, config, wrist_flex_deg):
        self.config = config
        self.wrist_flex_deg = wrist_flex_deg
        self._planner = SearchPrepPlanner(config, wrist_flex_deg)

    def _planner_for_final_offset(self, final_wrist_offset_deg):
        if final_wrist_offset_deg is None:
            return self._planner
        if abs(final_wrist_offset_deg - self.config.final_wrist_offset_deg) <= 1e-9:
            return self._planner

        return SearchPrepPlanner(
            SearchPrepConfig(
                min_duration_sec=self.config.min_duration_sec,
                head_down_duration_sec=self.config.head_down_duration_sec,
                duration_scale=self.config.duration_scale,
                trajectory_samples=self.config.trajectory_samples,
                max_vel_deg=self.config.max_vel_deg,
                max_acc_deg=self.config.max_acc_deg,
                final_wrist_offset_deg=final_wrist_offset_deg,
                wrist_flex_min_deg=self.config.wrist_flex_min_deg,
                wrist_flex_max_deg=self.config.wrist_flex_max_deg,
            ),
            self.wrist_flex_deg,
        )

    @staticmethod
    def _with_target_theta(trajectory, target_theta_deg):
        return MoveCupSearchPrepTrajectory(
            start_deg=trajectory.start_deg,
            goal_deg=trajectory.goal_deg,
            start_wrist_offset_deg=trajectory.start_wrist_offset_deg,
            final_wrist_offset_deg=trajectory.final_wrist_offset_deg,
            target_theta_deg=float(target_theta_deg),
            duration_sec=trajectory.duration_sec,
            head_down_duration_sec=trajectory.head_down_duration_sec,
            head_down_start_sec=trajectory.head_down_start_sec,
        )

    @staticmethod
    def _base_trajectory(trajectory):
        return SearchPrepTrajectory(
            start_deg=trajectory.start_deg,
            goal_deg=trajectory.goal_deg,
            start_wrist_offset_deg=trajectory.start_wrist_offset_deg,
            final_wrist_offset_deg=trajectory.final_wrist_offset_deg,
            duration_sec=trajectory.duration_sec,
            head_down_duration_sec=trajectory.head_down_duration_sec,
            head_down_start_sec=trajectory.head_down_start_sec,
        )

    def make_trajectory(
        self,
        start_deg,
        goal_deg,
        start_wrist_offset_deg,
        target_theta_deg=0.0,
        final_wrist_offset_deg=None,
    ):
        planner = self._planner_for_final_offset(final_wrist_offset_deg)
        trajectory = planner.make_trajectory(
            start_deg,
            goal_deg,
            start_wrist_offset_deg,
        )
        return self._with_target_theta(trajectory, target_theta_deg)

    def duration_sec(
        self,
        start_deg,
        goal_deg,
        start_wrist_offset_deg,
        final_wrist_offset_deg=None,
    ):
        planner = self._planner_for_final_offset(final_wrist_offset_deg)
        return planner.duration_sec(
            start_deg,
            goal_deg,
            start_wrist_offset_deg,
        )

    def sample_limits(
        self,
        start_deg,
        goal_deg,
        start_wrist_offset_deg,
        duration_sec,
        final_wrist_offset_deg=None,
    ):
        planner = self._planner_for_final_offset(final_wrist_offset_deg)
        return planner.sample_limits(
            start_deg,
            goal_deg,
            start_wrist_offset_deg,
            duration_sec,
        )

    def joint_degs_at(self, elapsed_sec, trajectory):
        planner = self._planner_for_final_offset(trajectory.final_wrist_offset_deg)
        return planner.joint_degs_at(
            elapsed_sec,
            self._base_trajectory(trajectory),
        )

    def wrist_offset_at(self, elapsed_sec, trajectory):
        planner = self._planner_for_final_offset(trajectory.final_wrist_offset_deg)
        return planner.wrist_offset_at(
            elapsed_sec,
            self._base_trajectory(trajectory),
        )

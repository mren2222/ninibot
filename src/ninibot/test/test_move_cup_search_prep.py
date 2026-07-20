import math

from ninibot.skills.move_cup.search_prep import MoveCupSearchPrepPlanner
from ninibot.skills.move_cup.search_prep import SearchPrepConfig
from ninibot.skills.move_cup.search_prep import minimum_jerk


def wrist_flex_deg(joint2_deg, joint3_deg):
    return joint2_deg + joint3_deg


def make_planner(**overrides):
    config = {
        "min_duration_sec": 1.0,
        "head_down_duration_sec": 0.4,
        "duration_scale": 1.0,
        "trajectory_samples": 40,
        "max_vel_deg": (1000.0, 1000.0, 1000.0, 1000.0),
        "max_acc_deg": (1000.0, 1000.0, 1000.0, 1000.0),
        "final_wrist_offset_deg": 15.0,
        "wrist_flex_min_deg": -180.0,
        "wrist_flex_max_deg": 180.0,
    }
    config.update(overrides)
    return MoveCupSearchPrepPlanner(SearchPrepConfig(**config), wrist_flex_deg)


def test_minimum_jerk_is_reused_and_clamps_endpoints():
    assert minimum_jerk(-1.0) == 0.0
    assert minimum_jerk(0.0) == 0.0
    assert math.isclose(minimum_jerk(0.5), 0.5)
    assert minimum_jerk(1.0) == 1.0
    assert minimum_jerk(2.0) == 1.0


def test_move_cup_trajectory_keeps_target_theta_metadata():
    planner = make_planner()

    trajectory = planner.make_trajectory(
        start_deg=[0.0, 10.0, 20.0, 35.0],
        goal_deg=[30.0, 20.0, 10.0, 45.0],
        start_wrist_offset_deg=5.0,
        target_theta_deg=-30.0,
    )

    assert trajectory.target_theta_deg == -30.0
    assert trajectory.final_wrist_offset_deg == 15.0
    assert planner.joint_degs_at(0.0, trajectory) == [0.0, 10.0, 20.0, 35.0]
    assert planner.joint_degs_at(trajectory.duration_sec, trajectory) == [
        30.0,
        20.0,
        10.0,
        45.0,
    ]


def test_return_trajectory_can_override_final_wrist_offset():
    planner = make_planner()

    trajectory = planner.make_trajectory(
        start_deg=[0.0, 0.0, 0.0, 12.0],
        goal_deg=[10.0, 20.0, 10.0, 35.0],
        start_wrist_offset_deg=12.0,
        target_theta_deg=-10.0,
        final_wrist_offset_deg=5.0,
    )

    assert trajectory.final_wrist_offset_deg == 5.0
    assert planner.wrist_offset_at(trajectory.duration_sec, trajectory) == 5.0
    assert planner.joint_degs_at(trajectory.duration_sec, trajectory) == [
        10.0,
        20.0,
        10.0,
        35.0,
    ]


def test_duration_and_sample_limits_delegate_to_generic_planner():
    planner = make_planner(
        min_duration_sec=0.1,
        head_down_duration_sec=0.1,
        max_vel_deg=(5.0, 1000.0, 1000.0, 1000.0),
    )

    duration = planner.duration_sec(
        start_deg=[0.0, 0.0, 0.0, 0.0],
        goal_deg=[20.0, 0.0, 0.0, 15.0],
        start_wrist_offset_deg=0.0,
    )
    max_vel_seen, max_acc_seen = planner.sample_limits(
        start_deg=[0.0, 0.0, 0.0, 0.0],
        goal_deg=[20.0, 0.0, 0.0, 15.0],
        start_wrist_offset_deg=0.0,
        duration_sec=duration,
    )

    assert duration > 0.1
    assert max_vel_seen[0] <= 5.0 + 1e-6
    assert max_acc_seen[0] >= 0.0

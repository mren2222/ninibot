import math

from ninibot.skills.grasp_cup.search_prep import SearchPrepConfig
from ninibot.skills.grasp_cup.search_prep import SearchPrepPlanner
from ninibot.skills.grasp_cup.search_prep import minimum_jerk


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
    return SearchPrepPlanner(SearchPrepConfig(**config), wrist_flex_deg)


def test_minimum_jerk_clamps_and_reaches_endpoints():
    assert minimum_jerk(-1.0) == 0.0
    assert minimum_jerk(0.0) == 0.0
    assert math.isclose(minimum_jerk(0.5), 0.5)
    assert minimum_jerk(1.0) == 1.0
    assert minimum_jerk(2.0) == 1.0


def test_search_prep_sample_starts_and_reaches_goal_offset():
    planner = make_planner()
    trajectory = planner.make_trajectory(
        start_deg=[0.0, 10.0, 20.0, 35.0],
        goal_deg=[30.0, 20.0, 10.0, 45.0],
        start_wrist_offset_deg=5.0,
    )

    start = planner.joint_degs_at(0.0, trajectory)
    end = planner.joint_degs_at(trajectory.duration_sec, trajectory)

    assert start == [0.0, 10.0, 20.0, 35.0]
    assert end == [30.0, 20.0, 10.0, 45.0]


def test_search_prep_wrist_offset_decays_before_head_down_phase():
    planner = make_planner(head_down_duration_sec=0.25)
    trajectory = planner.make_trajectory(
        start_deg=[0.0, 0.0, 0.0, 12.0],
        goal_deg=[0.0, 0.0, 0.0, 15.0],
        start_wrist_offset_deg=12.0,
    )

    before_head_down = planner.wrist_offset_at(
        trajectory.head_down_start_sec * 0.5,
        trajectory,
    )
    final = planner.wrist_offset_at(trajectory.duration_sec, trajectory)

    assert 0.0 < before_head_down < 12.0
    assert final == 15.0


def test_search_prep_duration_scales_to_joint_limits():
    planner = make_planner(
        min_duration_sec=0.1,
        head_down_duration_sec=0.1,
        max_vel_deg=(5.0, 1000.0, 1000.0, 1000.0),
    )

    duration = planner.duration_sec(
        start_deg=[0.0, 0.0, 0.0, 0.0],
        goal_deg=[20.0, 0.0, 0.0, 0.0],
        start_wrist_offset_deg=0.0,
    )

    assert duration > 0.1

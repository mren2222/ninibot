import math

from ninibot.skills.press_button.params import PrepTrajectoryConfig
from ninibot.skills.press_button.prep_trajectory import PrepTrajectoryPlanner
from ninibot.skills.press_button.prep_trajectory import joint_pitch_deg


def make_planner():
    return PrepTrajectoryPlanner(
        PrepTrajectoryConfig(
            prep_joint1_deg=0.0,
            prep_joint2_deg=0.0,
            prep_joint3_deg=-45.0,
            prep_joint4_deg=65.0,
            duration_sec=5.0,
            j2_transition_sec=3.0,
            search_pan_min_deg=0.0,
            search_pan_max_deg=90.0,
            joint4_lower_rad=math.radians(30.0),
            joint4_upper_rad=math.radians(80.0),
            joint6_rad=0.25,
        )
    )


def test_prep_trajectory_uses_current_pan_and_joint6_goal():
    planner = make_planner()
    latest = [
        math.radians(45.0),
        math.radians(10.0),
        math.radians(20.0),
        math.radians(30.0),
        0.0,
        0.75,
    ]

    trajectory = planner.make_trajectory(latest)

    assert math.isclose(math.degrees(trajectory.goal[0]), 45.0)
    assert math.isclose(trajectory.goal[4], 0.25)
    assert math.isclose(trajectory.start[4], 0.75)


def test_prep_trajectory_clamps_current_pan_to_search_range():
    planner = make_planner()
    latest = [math.radians(120.0), 0.0, 0.0, 0.0, 0.0, 0.0]

    pan = planner.pan_from_current(latest)
    trajectory = planner.make_trajectory(latest)

    assert math.isclose(pan.clamped_from_deg, 120.0)
    assert math.isclose(math.degrees(trajectory.goal[0]), 90.0)


def test_prep_trajectory_samples_start_and_goal():
    planner = make_planner()
    latest = [
        math.radians(30.0),
        math.radians(10.0),
        math.radians(20.0),
        math.radians(40.0),
        0.0,
        0.6,
    ]
    trajectory = planner.make_trajectory(latest)

    start = planner.sample(trajectory, 0.0)
    goal = planner.sample(trajectory, trajectory.duration_sec)

    assert all(math.isclose(a, b) for a, b in zip(start, trajectory.start))
    assert all(math.isclose(a, b) for a, b in zip(goal, trajectory.goal))


def test_prep_trajectory_holds_joint2_until_transition_window():
    planner = make_planner()
    latest = [
        math.radians(30.0),
        math.radians(25.0),
        math.radians(15.0),
        math.radians(20.0),
        0.0,
        0.0,
    ]
    trajectory = planner.make_trajectory(latest)

    sample = planner.sample(trajectory, 1.0)

    assert math.isclose(sample[1], latest[1])
    assert joint_pitch_deg(sample) != joint_pitch_deg(trajectory.start)


def test_prep_trajectory_clamps_joint4():
    planner = PrepTrajectoryPlanner(
        PrepTrajectoryConfig(
            prep_joint1_deg=0.0,
            prep_joint2_deg=80.0,
            prep_joint3_deg=-10.0,
            prep_joint4_deg=80.0,
            duration_sec=1.0,
            j2_transition_sec=1.0,
            search_pan_min_deg=-90.0,
            search_pan_max_deg=90.0,
            joint4_lower_rad=math.radians(30.0),
            joint4_upper_rad=math.radians(45.0),
            joint6_rad=0.0,
        )
    )
    latest = [0.0, 0.0, 0.0, math.radians(100.0), 0.0, 0.0]
    trajectory = planner.make_trajectory(latest)

    sample = planner.sample(trajectory, 0.5)

    assert math.degrees(sample[3]) <= 45.0

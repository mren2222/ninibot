import math

from ninibot.skills.press_button.params import PressMotionConfig
from ninibot.skills.press_button.press_motion import PressMotionPlanner


def make_planner():
    return PressMotionPlanner(
        PressMotionConfig(
            press_joint2_delta_deg=10.0,
            press_joint3_delta_deg=15.0,
            press_joint4_delta_deg=5.0,
            joint6_rad=0.3,
            ik_lower_bounds=(
                math.radians(-90.0),
                math.radians(-90.0),
                math.radians(-90.0),
                math.radians(-90.0),
            ),
            ik_upper_bounds=(
                math.radians(90.0),
                math.radians(90.0),
                math.radians(90.0),
                math.radians(90.0),
            ),
            joint4_lower_rad=math.radians(-20.0),
            joint4_upper_rad=math.radians(20.0),
        )
    )


def test_press_nudge_adds_joint_deltas_and_sets_gripper():
    planner = make_planner()
    base = [0.0, 0.0, 0.0, 0.0, 0.0]

    goal = planner.make_press_nudge(base)

    assert math.isclose(math.degrees(goal[1]), 10.0)
    assert math.isclose(math.degrees(goal[2]), 15.0)
    assert math.isclose(math.degrees(goal[3]), 5.0)
    assert goal[4] == 0.3


def test_press_lift_subtracts_joint_deltas():
    planner = make_planner()
    press_goal = [
        0.0,
        math.radians(10.0),
        math.radians(15.0),
        math.radians(5.0),
        0.3,
    ]

    lift = planner.make_press_lift(press_goal)

    assert math.isclose(lift[1], 0.0)
    assert math.isclose(lift[2], 0.0)
    assert math.isclose(lift[3], 0.0)


def test_press_motion_clamps_joint4_to_configured_bounds():
    planner = make_planner()
    base = [0.0, 0.0, 0.0, math.radians(19.0), 0.0]

    goal = planner.make_press_nudge(base)

    assert math.isclose(math.degrees(goal[3]), 20.0)


import math

from ninibot.skills.press_button.joint_ramp import ramp_joint_goal


def test_ramp_joint_goal_steps_without_overshoot():
    result = ramp_joint_goal(
        [0.0, 0.0, 0.0, 0.0, 0.1],
        [math.radians(10.0), 0.0, 0.0, 0.0, 0.2],
        [0],
        control_rate_hz=10.0,
        speed_scale=1.0,
    )

    assert math.isclose(result.target[0], math.radians(3.0))
    assert result.target[4] == 0.2
    assert not result.done


def test_ramp_joint_goal_snaps_when_within_step():
    goal = [math.radians(2.0), 0.0, 0.0, 0.0, 0.2]

    result = ramp_joint_goal(
        [0.0, 0.0, 0.0, 0.0, 0.1],
        goal,
        [0],
        control_rate_hz=10.0,
        speed_scale=1.0,
    )

    assert result.target == goal
    assert result.done


def test_ramp_joint_goal_only_changes_selected_joints():
    result = ramp_joint_goal(
        [0.0, 0.0, 0.0, 0.0],
        [math.radians(30.0), math.radians(30.0), 0.0, 0.0],
        [1],
        control_rate_hz=10.0,
        speed_scale=1.0,
    )

    assert result.target[0] == 0.0
    assert math.isclose(result.target[1], math.radians(3.0))


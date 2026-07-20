import math

from ninibot.skills.press_button.target_planner import horizontal_plus_offset_joint4_deg
from ninibot.skills.press_button.target_planner import joint4_rad_from_joint23
from ninibot.skills.press_button.target_planner import pose_from_q1234


def test_joint4_alignment_uses_joint23_and_wrist_pitch_offset():
    assert horizontal_plus_offset_joint4_deg(10.0, 20.0, 59.4) == 29.4


def test_joint4_rad_from_joint23_returns_radians():
    value = joint4_rad_from_joint23(
        math.radians(10.0),
        math.radians(20.0),
        59.4,
    )

    assert math.isclose(math.degrees(value), 29.4)


def test_pose_from_q1234_appends_gripper_joint():
    pose = pose_from_q1234([1.0, 2.0, 3.0, 4.0], joint6_rad=0.5)

    assert pose == [1.0, 2.0, 3.0, 4.0, 0.5]


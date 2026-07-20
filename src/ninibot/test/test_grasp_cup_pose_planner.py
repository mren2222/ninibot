import math
from types import SimpleNamespace

from ninibot.skills.grasp_cup.pose_planner import PosePlanner
from ninibot.skills.grasp_cup.pose_planner import PosePlannerConfig
from ninibot.skills.grasp_cup.pose_planner import gripper_percent_to_rad


class FakeKinematics:
    def __init__(self, solution=None):
        self.solution = solution
        self.last_error = "ik failed"

    def base_deg_from_theta(self, theta_deg):
        return theta_deg + 10.0

    def wrist_flex_deg(self, joint2_deg, joint3_deg):
        return joint2_deg + joint3_deg

    def cylindrical_ik(self, radius_m, theta_deg, z_m):
        return self.solution


def make_planner(solution=None, **overrides):
    config = {
        "search_joint2_deg": 20.0,
        "search_joint3_deg": 30.0,
        "search_joint4_deg": 15.0,
        "wrist_flex_min_deg": -90.0,
        "wrist_flex_max_deg": 60.0,
        "wrist_roll_deg": 5.0,
        "gripper_rad": 0.7,
        "base_min_deg": -45.0,
        "base_max_deg": 90.0,
    }
    config.update(overrides)
    return PosePlanner(PosePlannerConfig(**config), FakeKinematics(solution))


def test_gripper_percent_to_rad_clamps_to_configured_range():
    assert math.isclose(gripper_percent_to_rad(-10.0), -0.174533)
    assert math.isclose(gripper_percent_to_rad(100.0), 1.74533)
    assert math.isclose(
        gripper_percent_to_rad(50.0),
        (-0.174533 + 1.74533) / 2.0,
    )


def test_search_goal_clamps_wrist_and_search_pose_has_six_joints():
    planner = make_planner(wrist_flex_max_deg=55.0)

    goal = planner.search_goal_joint_degs(base_deg=12.0)
    pose = planner.search_pose(base_deg=12.0)

    assert goal == [12.0, 20.0, 30.0, 55.0]
    assert len(pose) == 6
    assert math.isclose(pose[0], math.radians(12.0))
    assert math.isclose(pose[3], math.radians(55.0))
    assert math.isclose(pose[4], math.radians(5.0))
    assert pose[5] == 0.7


def test_wrist_pitch_uses_fine_align_prep_joint4_when_available():
    planner = make_planner()

    pitch = planner.wrist_pitch_rad(
        "FINE_ALIGN_PREP",
        math.radians(20.0),
        math.radians(30.0),
        fine_align_prep_joint4_rad=1.23,
    )

    assert pitch == 1.23


def test_wrist_pitch_adds_head_down_offset_for_tracking_center():
    planner = make_planner()

    pitch = planner.wrist_pitch_rad(
        "TRACKING_CENTER",
        math.radians(20.0),
        math.radians(30.0),
    )

    assert math.isclose(pitch, math.radians(65.0))


def test_cylindrical_servo_pose_returns_positions_and_joint_targets():
    solution = SimpleNamespace(
        base_deg=11.0,
        joint2_deg=22.0,
        joint3_deg=33.0,
        wrist_flex_deg=44.0,
    )
    planner = make_planner(solution=solution)

    plan = planner.cylindrical_servo_pose(
        radius_m=0.2,
        theta_rad=math.radians(1.0),
        z_m=0.1,
    )

    assert plan.warning_key == ""
    assert plan.joint2_deg == 22.0
    assert plan.joint3_deg == 33.0
    assert len(plan.positions) == 6
    assert math.isclose(plan.positions[3], math.radians(44.0))


def test_cylindrical_servo_pose_returns_warning_on_ik_failure():
    planner = make_planner(solution=None)

    plan = planner.cylindrical_servo_pose(
        radius_m=0.2,
        theta_rad=0.0,
        z_m=0.1,
    )

    assert plan.positions is None
    assert plan.warning_key == "cylindrical_servo_ik"
    assert plan.warning_text == "ik failed"

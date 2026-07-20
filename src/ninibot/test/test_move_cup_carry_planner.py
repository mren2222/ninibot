import math

from arm_controller.kinematics.cylindrical import CylindricalPose
from ninibot.skills.move_cup.carry_planner import MoveCupCarryPlanner
from ninibot.skills.move_cup.carry_planner import MoveCupCarryPlannerConfig
from ninibot.skills.move_cup.carry_planner import interpolate_cylindrical_pose
from ninibot.skills.move_cup.carry_planner import interpolate_gripper_percent
from ninibot.skills.move_cup.carry_planner import safe_pre_pan_radius


class FakeKinematics:
    def base_deg_from_theta(self, theta_deg):
        return -theta_deg

    def theta_deg_from_base(self, base_deg, fallback_deg=0.0):
        del fallback_deg
        return -base_deg


def make_planner(**overrides):
    config = {
        "lift_z_offset": 0.03,
        "pre_pan_radius": 0.10,
        "post_release_clearance_retract_m": 0.03,
        "base_min_deg": -90.0,
        "base_max_deg": 90.0,
    }
    config.update(overrides)
    return MoveCupCarryPlanner(
        MoveCupCarryPlannerConfig(**config),
        FakeKinematics(),
    )


def test_safe_pre_pan_radius_only_retracts_when_positive_and_smaller():
    assert safe_pre_pan_radius(0.20, 0.10) == 0.10
    assert safe_pre_pan_radius(0.08, 0.10) == 0.08
    assert safe_pre_pan_radius(0.20, 0.0) == 0.20


def test_basic_carry_goals_preserve_expected_axes():
    planner = make_planner()
    start = CylindricalPose(radius=0.20, theta_deg=-30.0, z=0.05)

    assert planner.lift_goal(start) == CylindricalPose(
        radius=0.20,
        theta_deg=-30.0,
        z=0.08,
    )
    assert planner.retract_radius_goal(start) == CylindricalPose(
        radius=0.10,
        theta_deg=-30.0,
        z=0.05,
    )
    assert planner.horizontal_move_goal(start, 0.0, 0.08) == CylindricalPose(
        radius=0.20,
        theta_deg=0.0,
        z=0.08,
    )
    assert planner.extend_radius_goal(start, 0.25, 0.08) == CylindricalPose(
        radius=0.25,
        theta_deg=-30.0,
        z=0.08,
    )
    lower = planner.lower_goal(start)

    assert lower.radius == 0.20
    assert lower.theta_deg == -30.0
    assert math.isclose(lower.z, 0.02)


def test_release_goal_applies_base_offset_and_reports_clamp():
    planner = make_planner(base_min_deg=-20.0, base_max_deg=20.0)
    start = CylindricalPose(radius=0.20, theta_deg=-15.0, z=0.05)

    unclamped = planner.release_goal(start, base_pan_offset_deg=-5.0)

    assert unclamped.warning_text == ""
    assert math.isclose(unclamped.pose.theta_deg, -10.0)

    clamped = planner.release_goal(start, base_pan_offset_deg=20.0)

    assert "base pan clearance clamped" in clamped.warning_text
    assert math.isclose(clamped.pose.theta_deg, -20.0)


def test_post_release_clearance_retracts_without_negative_radius():
    planner = make_planner(post_release_clearance_retract_m=0.03)

    assert planner.post_release_clearance_goal(
        CylindricalPose(radius=0.20, theta_deg=10.0, z=0.05)
    ) == CylindricalPose(radius=0.17, theta_deg=10.0, z=0.05)
    assert planner.post_release_clearance_goal(
        CylindricalPose(radius=0.02, theta_deg=10.0, z=0.05)
    ) == CylindricalPose(radius=0.0, theta_deg=10.0, z=0.05)


def test_path_and_gripper_interpolation_clamp_alpha():
    start = CylindricalPose(radius=0.10, theta_deg=-20.0, z=0.02)
    goal = CylindricalPose(radius=0.30, theta_deg=20.0, z=0.10)

    mid = interpolate_cylindrical_pose(start, goal, 0.5)

    assert mid == CylindricalPose(radius=0.20, theta_deg=0.0, z=0.06)
    assert interpolate_cylindrical_pose(start, goal, -1.0) == start
    assert interpolate_cylindrical_pose(start, goal, 2.0) == goal
    assert interpolate_gripper_percent(5.0, 50.0, 0.5) == 27.5
    assert interpolate_gripper_percent(5.0, 50.0, 2.0) == 50.0

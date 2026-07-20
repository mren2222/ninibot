import math
from types import SimpleNamespace

from arm_controller.kinematics.cylindrical import CylindricalPose
from ninibot.skills.move_cup.phases import ACTIVE_PHASES
from ninibot.skills.move_cup.phases import EXTENDING_RADIUS
from ninibot.skills.move_cup.phases import PATH_PHASES
from ninibot.skills.move_cup.phases import POST_RELEASE_CLEARANCE
from ninibot.skills.move_cup.phases import RETRACTING_RADIUS
from ninibot.skills.move_cup.phases import SEARCH_TARGET_PREP
from ninibot.skills.move_cup.phases import TARGET_LOCKED
from ninibot.skills.move_cup.pose_planner import GRIPPER_MAX_RAD
from ninibot.skills.move_cup.pose_planner import GRIPPER_MIN_RAD
from ninibot.skills.move_cup.pose_planner import MoveCupPosePlanner
from ninibot.skills.move_cup.pose_planner import MoveCupPosePlannerConfig
from ninibot.skills.move_cup.pose_planner import gripper_percent_to_rad


class FakeKinematics:
    def __init__(self, fail_ik=False):
        self.l1 = 1.0
        self.l2 = 1.0
        self.horizontal_tool_offset = 0.0
        self.z_tool_offset = 0.0
        self.fail_ik = fail_ik
        self.last_error = ""

    def base_deg_from_theta(self, theta_deg):
        return -theta_deg

    def theta_deg_from_base(self, base_deg, fallback_deg=0.0):
        del fallback_deg
        return -base_deg

    def wrist_flex_deg(self, joint2_deg, joint3_deg):
        return -joint2_deg - joint3_deg + 5.0

    def cylindrical_from_joint_degs(self, base_deg, shoulder_deg, elbow_deg):
        del shoulder_deg, elbow_deg
        return CylindricalPose(radius=0.3, theta_deg=-base_deg, z=0.1)

    def cylindrical_ik(self, radius, theta_deg, z):
        del z
        if self.fail_ik or radius > 2.0:
            self.last_error = "no solution"
            return None

        joint2_deg = 10.0
        joint3_deg = 20.0
        return SimpleNamespace(
            base_deg=self.base_deg_from_theta(theta_deg),
            joint2_deg=joint2_deg,
            joint3_deg=joint3_deg,
            wrist_flex_deg=self.wrist_flex_deg(joint2_deg, joint3_deg),
        )


def make_planner(**overrides):
    config = {
        "base_min_deg": -90.0,
        "base_max_deg": 90.0,
        "shoulder_min_deg": -100.0,
        "shoulder_max_deg": 100.0,
        "elbow_min_deg": -90.0,
        "elbow_max_deg": 100.0,
        "wrist_flex_min_deg": -104.0,
        "wrist_flex_max_deg": 100.0,
        "wrist_roll_deg": -90.0,
        "hold_gripper_percent": 5.0,
        "release_gripper_percent": 50.0,
        "target_radius_margin_m": 0.1,
        "target_search_theta_min_deg": -80.0,
        "target_search_theta_max_deg": 0.0,
        "search_joint2_deg": -90.0,
        "search_joint3_deg": 98.0,
        "search_joint4_deg": 25.0,
        "return_joint2_deg": -90.0,
        "return_joint3_deg": 98.0,
        "return_search_prep_final_offset_deg": 25.0,
    }
    config.update(overrides)
    return MoveCupPosePlanner(
        MoveCupPosePlannerConfig(**config),
        FakeKinematics(),
    )


def test_phase_sets_keep_move_cup_external_statuses():
    assert RETRACTING_RADIUS in PATH_PHASES
    assert EXTENDING_RADIUS in PATH_PHASES
    assert POST_RELEASE_CLEARANCE in ACTIVE_PHASES
    assert SEARCH_TARGET_PREP not in PATH_PHASES
    assert TARGET_LOCKED not in ACTIVE_PHASES


def test_gripper_percent_to_rad_clamps_to_physical_range():
    assert gripper_percent_to_rad(-10.0) == GRIPPER_MIN_RAD
    assert gripper_percent_to_rad(0.0) == GRIPPER_MIN_RAD
    assert gripper_percent_to_rad(100.0) == GRIPPER_MAX_RAD
    assert gripper_percent_to_rad(150.0) == GRIPPER_MAX_RAD


def test_make_joint_pose_maps_degrees_and_rejects_out_of_range():
    planner = make_planner()

    pose = planner.make_joint_pose(0.0, 10.0, 20.0, -25.0, 50.0)

    assert pose.error_text == ""
    assert pose.positions[:5] == [
        0.0,
        math.radians(10.0),
        math.radians(20.0),
        math.radians(-25.0),
        math.radians(-90.0),
    ]
    assert math.isclose(pose.positions[5], gripper_percent_to_rad(50.0))

    bad = planner.make_joint_pose(120.0, 10.0, 20.0, -25.0, 50.0)

    assert bad.positions is None
    assert "base out of range" in bad.error_text


def test_search_goal_uses_theta_mapping_and_wrist_offset():
    planner = make_planner()

    goal = planner.target_search_goal_joint_degs(-20.0)

    assert goal == [20.0, -90.0, 98.0, 22.0]


def test_current_pose_helpers_use_feedback_or_fallback():
    planner = make_planner()
    latest = [
        math.radians(12.0),
        math.radians(-90.0),
        math.radians(98.0),
        0.0,
    ]

    assert math.isclose(
        planner.current_pan_deg(latest, fallback_theta_deg=-15.0),
        12.0,
    )
    assert math.isclose(
        planner.current_theta_deg(latest, fallback_theta_deg=-15.0),
        -12.0,
    )
    assert planner.current_pan_deg(None, fallback_theta_deg=-15.0) == 15.0
    assert planner.current_theta_deg(None, fallback_theta_deg=-15.0) == -15.0

    pose = planner.current_cylindrical_pose(latest)

    assert pose.radius == 0.3
    assert math.isclose(pose.theta_deg, -12.0)
    assert pose.z == 0.1


def test_locked_target_forces_release_z_and_clamps_reach_margin():
    planner = make_planner(target_radius_margin_m=0.1)

    plan = planner.locked_target_from_base_xyz(
        [2.2, 0.0, 99.0],
        release_z=0.4,
    )

    assert plan.success
    assert plan.z == 0.4
    assert plan.radius < 2.0
    assert plan.target_base_xyz[2] == 0.4
    assert "radius clamped" in plan.warning_text


def test_locked_target_rejects_zero_radius_and_unreachable_target():
    planner = make_planner()

    zero_radius = planner.locked_target_from_base_xyz(
        [0.0, 0.0, 99.0],
        release_z=0.0,
    )

    assert not zero_radius.success
    assert "invalid target radius" in zero_radius.error_text

    unreachable = MoveCupPosePlanner(
        planner.config,
        FakeKinematics(fail_ik=True),
    ).locked_target_from_base_xyz([0.3, 0.0, 99.0], release_z=0.0)

    assert not unreachable.success
    assert "release target unreachable" in unreachable.error_text


def test_clamp_target_joint_degs_reports_changed_joints():
    planner = make_planner()

    plan = planner.clamp_target_joint_degs([120.0, -120.0, 0.0, 120.0], "CTX")

    assert plan.joint_degs == [90.0, -100.0, 0.0, 100.0]
    assert "base 120.0->90.0deg" in plan.warning_text
    assert "shoulder -120.0->-100.0deg" in plan.warning_text
    assert "wrist 120.0->100.0deg" in plan.warning_text

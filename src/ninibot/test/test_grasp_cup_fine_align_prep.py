import math
from types import SimpleNamespace

from ninibot.skills.grasp_cup.fine_align_prep import FineAlignPrepConfig
from ninibot.skills.grasp_cup.fine_align_prep import FineAlignPrepPlanner


def make_template():
    return {
        "duration_sec": 10.0,
        "processed_end_rad": [0.1, 0.2, 0.3],
        "joint4_mode": "recorded",
        "samples": [
            {
                "s": 0.0,
                "delta_shape_joint2_rad": 0.0,
                "delta_shape_joint3_rad": 0.0,
                "delta_shape_joint4_rad": 0.0,
            },
            {
                "s": 1.0,
                "delta_shape_joint2_rad": 0.0,
                "delta_shape_joint3_rad": 0.0,
                "delta_shape_joint4_rad": 0.0,
            },
        ],
    }


class FakeKinematics:
    def __init__(self, adjusted_joint23=None):
        self.adjusted_joint23 = adjusted_joint23
        self.last_error = "no solution"

    def cylindrical_from_joint_degs(self, base_deg, joint2_deg, joint3_deg):
        return SimpleNamespace(radius=0.2, z=0.1)

    def joint23_ik(self, radius, z):
        return self.adjusted_joint23

    def wrist_flex_deg(self, joint2_deg, joint3_deg):
        return joint2_deg + joint3_deg


def make_planner(**overrides):
    config = {
        "duration_sec": 5.0,
        "shape_scale": 1.0,
        "z_offset_m": 0.0,
        "wrist_flex_min_deg": -180.0,
        "wrist_flex_max_deg": 180.0,
    }
    config.update(overrides)
    return FineAlignPrepPlanner(
        make_template(),
        FakeKinematics(adjusted_joint23=(30.0, 40.0)),
        FineAlignPrepConfig(**config),
    )


def test_fine_align_prep_end_uses_template_and_computes_joint4():
    planner = make_planner()

    end = planner.end_j234()

    assert end.warning_key == ""
    assert end.j234_rad == [0.1, 0.2, 0.30000000000000004]


def test_fine_align_prep_end_applies_valid_z_offset_solution():
    planner = make_planner(z_offset_m=0.02)

    end = planner.end_j234()

    assert end.warning_key == ""
    assert math.isclose(end.j234_rad[0], math.radians(30.0))
    assert math.isclose(end.j234_rad[1], math.radians(40.0))
    assert math.isclose(end.j234_rad[2], math.radians(70.0))


def test_fine_align_prep_end_returns_warning_when_z_offset_ik_fails():
    planner = FineAlignPrepPlanner(
        make_template(),
        FakeKinematics(adjusted_joint23=None),
        FineAlignPrepConfig(
            duration_sec=5.0,
            shape_scale=1.0,
            z_offset_m=0.02,
            wrist_flex_min_deg=-180.0,
            wrist_flex_max_deg=180.0,
        ),
    )

    end = planner.end_j234()

    assert end.warning_key == "fine_align_prep_z_offset_ik"
    assert "no solution" in end.warning_text
    assert end.j234_rad == [0.1, 0.2, 0.30000000000000004]


def test_fine_align_prep_progress_and_sample_retarget_linearly():
    planner = make_planner()
    trajectory = planner.start(
        start_j234=[0.0, 0.0, 0.0],
        now_sec=10.0,
        end_j234=[1.0, 2.0, 3.0],
    )

    assert planner.progress(trajectory, now_sec=12.5) == 0.5
    assert planner.sample_j234(trajectory, now_sec=12.5) == [0.5, 1.0, 1.5]


def test_fine_align_prep_ready_requires_progress_and_joint_tolerance():
    planner = make_planner()
    trajectory = planner.start(
        start_j234=[0.0, 0.0, 0.0],
        now_sec=10.0,
        end_j234=[1.0, 2.0, 3.0],
    )

    assert not planner.ready(
        trajectory,
        latest_joint_positions=[0.0, 1.0, 2.0, 3.0],
        joint_goal_tolerance=0.01,
        now_sec=12.5,
    )
    assert planner.ready(
        trajectory,
        latest_joint_positions=[0.0, 1.0, 2.0, 3.0],
        joint_goal_tolerance=0.01,
        now_sec=15.0,
    )

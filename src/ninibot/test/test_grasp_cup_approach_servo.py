import math
from types import SimpleNamespace

from ninibot.skills.grasp_cup.approach_servo import ApproachServo
from ninibot.skills.grasp_cup.approach_servo import ApproachServoConfig


class FakeKinematics:
    def __init__(self, solution=None):
        self.solution = solution
        self.last_error = "no ik"
        self.calls = []

    def cylindrical_ik(self, radius_m, theta_deg, z_m):
        self.calls.append((radius_m, theta_deg, z_m))
        return self.solution


def make_servo():
    return ApproachServo(
        ApproachServoConfig(
            bbox_tolerance_px=5.0,
            bbox_resume_tolerance_px=8.0,
            radius_kp_m_per_px=0.01,
            max_radius_step_m=0.03,
        )
    )


def test_update_lock_uses_enter_and_resume_tolerances():
    servo = make_servo()

    assert servo.update_lock(False, bbox_error=4.0)
    assert servo.update_lock(True, bbox_error=7.0)
    assert not servo.update_lock(True, bbox_error=9.0)


def test_should_step_requires_started_unlocked_and_outside_tolerance():
    servo = make_servo()

    assert not servo.should_step(False, False, bbox_error=10.0)
    assert not servo.should_step(True, True, bbox_error=10.0)
    assert not servo.should_step(True, False, bbox_error=5.0)
    assert servo.should_step(True, False, bbox_error=5.1)


def test_step_cylindrical_clamps_radius_step_and_returns_targets():
    servo = make_servo()
    kinematics = FakeKinematics(
        SimpleNamespace(joint2_deg=21.0, joint3_deg=32.0)
    )

    step = servo.step_cylindrical(
        radius_m=0.2,
        theta_rad=math.radians(30.0),
        z_m=0.1,
        kinematics=kinematics,
        bbox_error=20.0,
    )

    assert step.success
    assert math.isclose(step.radius_m, 0.23)
    assert step.joint2_deg == 21.0
    assert step.joint3_deg == 32.0
    radius_m, theta_deg, z_m = kinematics.calls[0]
    assert math.isclose(radius_m, 0.23)
    assert math.isclose(theta_deg, 30.0)
    assert z_m == 0.1


def test_step_cylindrical_returns_warning_on_ik_failure():
    servo = make_servo()
    kinematics = FakeKinematics(solution=None)

    step = servo.step_cylindrical(
        radius_m=0.2,
        theta_rad=0.0,
        z_m=0.1,
        kinematics=kinematics,
        bbox_error=-20.0,
    )

    assert not step.success
    assert step.warning_key == "cylindrical_approach_ik"
    assert step.warning_text == "no ik"

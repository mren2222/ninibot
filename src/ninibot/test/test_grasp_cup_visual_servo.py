import math

from ninibot.skills.grasp_cup.visual_servo import DetectionCenterSmoother
from ninibot.skills.grasp_cup.visual_servo import PanServo
from ninibot.skills.grasp_cup.visual_servo import PanServoConfig


def make_servo():
    return PanServo(
        PanServoConfig(
            theta_kp=0.1,
            theta_sign=-1.0,
            theta_max_step=0.2,
            tracking_theta_max_step=0.05,
            tracking_settle_tolerance_px=3.0,
            target_bbox_width=100.0,
        )
    )


def test_tracking_errors_use_detection_center_and_bbox_width():
    servo = make_servo()
    detection = {"center_x": 130.0, "bbox": (10, 20, 70, 80)}

    error_x, bbox_error, bbox_w = servo.tracking_errors(detection, image_target_x=120.0)

    assert error_x == 10.0
    assert bbox_error == 30.0
    assert bbox_w == 70


def test_pan_centered_uses_default_or_explicit_tolerance():
    servo = make_servo()

    assert servo.pan_centered(3.0)
    assert not servo.pan_centered(3.1)
    assert servo.pan_centered(5.0, tolerance=5.0)


def test_theta_delta_respects_tolerance_sign_and_max_step():
    servo = make_servo()

    assert servo.theta_delta(1.0, tolerance=1.0) == 0.0
    assert math.isclose(servo.theta_delta(10.0), -0.2)
    assert math.isclose(servo.theta_delta(-10.0), 0.2)


def test_tracking_theta_delta_uses_tracking_limit_and_tolerance():
    servo = make_servo()

    assert servo.tracking_theta_delta(3.0) == 0.0
    assert math.isclose(servo.tracking_theta_delta(10.0), -0.05)


def test_detection_center_smoother_preserves_other_fields():
    smoother = DetectionCenterSmoother(alpha=0.25)
    first = smoother.smooth({"center_x": 100.0, "bbox": (1, 2, 3, 4)})
    second = smoother.smooth({"center_x": 140.0, "bbox": (5, 6, 7, 8)})

    assert first["center_x"] == 100.0
    assert math.isclose(second["center_x"], 110.0)
    assert second["bbox"] == (5, 6, 7, 8)


def test_detection_center_smoother_reset_starts_from_next_detection():
    smoother = DetectionCenterSmoother(alpha=0.25)
    smoother.smooth({"center_x": 100.0})
    smoother.reset()

    smoothed = smoother.smooth({"center_x": 140.0})

    assert smoothed["center_x"] == 140.0

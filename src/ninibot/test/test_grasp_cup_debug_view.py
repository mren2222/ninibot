import math

from ninibot.skills.grasp_cup.debug_view import DebugTelemetry
from ninibot.skills.grasp_cup.debug_view import make_debug_text


def test_make_debug_text_formats_state_summary():
    text = make_debug_text(
        DebugTelemetry(
            status="FINE_ALIGN",
            radius_now=0.1234,
            theta_rad=math.radians(15.0),
            z_now=0.2345,
            approach_servo_started=True,
            approach_locked=False,
            tracking_settle_tolerance_px=3.0,
            tracking_centered_count=1,
            tracking_settle_cycles=2,
            fine_align_centered_count=3,
            fine_align_centered_cycles=4,
            servo_radius_m=None,
            servo_z_m=0.2,
            approach_joint2_deg=21.2,
            approach_joint3_deg=32.3,
            joint4_target_rad=math.radians(43.4),
            prep_progress=0.56,
            error_x=None,
            bbox_w=77.0,
            bbox_error=-8.5,
            target_bbox_width=100.0,
        )
    )

    assert "status=FINE_ALIGN" in text
    assert "radius_now=0.123" in text
    assert "theta_deg=15.0" in text
    assert "tracking_settle=1/2" in text
    assert "fine_settle=3/4" in text
    assert "servo_radius=0.000" in text
    assert "joint4_target=43.4" in text
    assert "error_x=0.0" in text
    assert "bbox_error=-8.5" in text

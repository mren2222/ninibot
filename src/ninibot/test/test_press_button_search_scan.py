import math

from ninibot.skills.press_button.params import SearchScanConfig
from ninibot.skills.press_button.search_scan import SearchScanner


def make_scanner():
    return SearchScanner(
        SearchScanConfig(
            prep_joint1_deg=0.0,
            prep_joint2_deg=1.0,
            prep_joint3_deg=-2.0,
            prep_joint4_deg=3.0,
            search_pan_min_deg=0.0,
            search_pan_max_deg=90.0,
            search_pan_speed_deg_s=30.0,
            joint6_rad=0.5,
        )
    )


def test_search_scan_initial_state_uses_current_pan():
    scanner = make_scanner()

    state = scanner.initial_state([math.radians(45.0)])

    assert state.pan_deg == 45.0
    assert state.direction == 1.0


def test_search_scan_clamps_initial_pan():
    scanner = make_scanner()

    state = scanner.initial_state([math.radians(120.0)])

    assert state.pan_deg == 90.0


def test_search_scan_updates_and_reverses_at_max():
    scanner = make_scanner()
    state = scanner.initial_state([math.radians(85.0)])

    state = scanner.update(state, 1.0)

    assert state.pan_deg == 90.0
    assert state.direction == -1.0


def test_search_scan_pose_uses_configured_arm_pose():
    scanner = make_scanner()

    pose = scanner.pose(30.0)

    assert [round(math.degrees(value), 6) for value in pose[:4]] == [
        30.0,
        1.0,
        -2.0,
        3.0,
    ]
    assert pose[4] == 0.5


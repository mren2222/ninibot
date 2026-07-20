from ninibot.skills.move_cup.phases import FAILED
from ninibot.skills.move_cup.phases import SEARCH_TARGET
from ninibot.skills.move_cup.phases import SEARCH_TARGET_PREP
from ninibot.skills.move_cup.phases import TARGET_LOCKED
from ninibot.skills.move_cup.state_machine import MOVE_ACTION_ACTIVE_STATUSES


def target_scan_state_after_status(status, state="SCANNING_MOVE_TARGET"):
    target_locked = False
    target_scan_active = False
    task_state = state

    if task_state == "SCANNING_MOVE_TARGET":
        if status in (SEARCH_TARGET_PREP, SEARCH_TARGET):
            target_scan_active = True
        elif status == TARGET_LOCKED:
            target_locked = True
            target_scan_active = False
            task_state = "WAITING_FOR_GRASP"
        elif status in ("FAILED", "CANCELED"):
            target_scan_active = False
            task_state = FAILED

    return task_state, target_scan_active, target_locked


def test_fake_full_demo_scan_contract():
    assert target_scan_state_after_status(SEARCH_TARGET_PREP) == (
        "SCANNING_MOVE_TARGET",
        True,
        False,
    )
    assert target_scan_state_after_status(SEARCH_TARGET) == (
        "SCANNING_MOVE_TARGET",
        True,
        False,
    )
    assert target_scan_state_after_status(TARGET_LOCKED) == (
        "WAITING_FOR_GRASP",
        False,
        True,
    )
    assert target_scan_state_after_status(FAILED) == (FAILED, False, False)


def test_fake_action_contract_treats_scan_as_activity():
    for status in (SEARCH_TARGET_PREP, SEARCH_TARGET, TARGET_LOCKED):
        assert status in MOVE_ACTION_ACTIVE_STATUSES

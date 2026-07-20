import numpy as np

from ninibot.skills.move_cup.target_scan import TargetScanConfig
from ninibot.skills.move_cup.target_scan import TargetScanPlanner


def make_planner(**overrides):
    config = {
        "theta_min_deg": -80.0,
        "theta_max_deg": 0.0,
        "theta_step_deg": 10.0,
        "lock_samples": 3,
        "lock_max_std_m": 0.01,
    }
    config.update(overrides)
    return TargetScanPlanner(TargetScanConfig(**config))


def test_next_pan_sweeps_and_reverses_at_bounds():
    planner = make_planner()

    update = planner.next_pan(theta_deg=-10.0, direction=1.0)

    assert update.theta_deg == 0.0
    assert update.direction == -1.0

    update = planner.next_pan(theta_deg=-75.0, direction=-1.0)

    assert update.theta_deg == -80.0
    assert update.direction == 1.0


def test_next_pan_pauses_while_locking_or_zero_step():
    planner = make_planner()

    assert planner.next_pan(-20.0, 1.0, locking=True).theta_deg == -20.0
    assert make_planner(theta_step_deg=0.0).next_pan(-20.0, 1.0).theta_deg == -20.0


def test_update_samples_starts_collection_and_locks_stable_mean():
    planner = make_planner(lock_samples=3, lock_max_std_m=0.01)
    samples = []

    first = planner.update_samples(samples, [0.10, 0.00, 0.02], locking=False)
    second = planner.update_samples(
        first.samples,
        [0.11, 0.00, 0.02],
        locking=first.locking,
    )
    third = planner.update_samples(
        second.samples,
        [0.10, 0.01, 0.02],
        locking=second.locking,
    )

    assert first.collecting_started
    assert not first.lock_ready
    assert second.sample_count == 2
    assert third.lock_ready
    assert np.allclose(third.target_mean, [0.1033333333, 0.0033333333, 0.02])


def test_update_samples_reports_unstable_without_dropping_window():
    planner = make_planner(lock_samples=3, lock_max_std_m=0.001)

    update = planner.update_samples([], [0.10, 0.00, 0.02], locking=False)
    update = planner.update_samples(update.samples, [0.12, 0.00, 0.02], locking=True)
    update = planner.update_samples(update.samples, [0.08, 0.00, 0.02], locking=True)

    assert update.unstable
    assert not update.lock_ready
    assert len(update.samples) == 3


def test_update_samples_resets_when_target_lost():
    planner = make_planner()

    update = planner.update_samples([[0.10, 0.00, 0.02]], None, locking=True)

    assert update.lost_target
    assert not update.locking
    assert update.samples == []

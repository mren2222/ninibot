from dataclasses import dataclass

import numpy as np


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class TargetScanConfig:
    theta_min_deg: float
    theta_max_deg: float
    theta_step_deg: float
    lock_samples: int
    lock_max_std_m: float

    def __post_init__(self):
        object.__setattr__(self, "lock_samples", max(1, int(self.lock_samples)))
        object.__setattr__(self, "lock_max_std_m", max(0.0, float(self.lock_max_std_m)))


@dataclass(frozen=True)
class SearchPanUpdate:
    theta_deg: float
    direction: float


@dataclass(frozen=True)
class TargetSampleUpdate:
    locking: bool
    samples: list
    sample_count: int = 0
    collecting_started: bool = False
    lost_target: bool = False
    lock_ready: bool = False
    unstable: bool = False
    target_mean: object = None
    max_std: float = 0.0


class TargetScanPlanner:
    def __init__(self, config):
        self.config = config

    def clamp_theta(self, theta_deg):
        return clamp(
            theta_deg,
            self.config.theta_min_deg,
            self.config.theta_max_deg,
        )

    def next_pan(self, theta_deg, direction, locking=False):
        if locking or self.config.theta_step_deg <= 0.0:
            return SearchPanUpdate(theta_deg=theta_deg, direction=direction)

        theta_deg += direction * self.config.theta_step_deg
        if theta_deg >= self.config.theta_max_deg:
            theta_deg = self.config.theta_max_deg
            direction = -1.0
        elif theta_deg <= self.config.theta_min_deg:
            theta_deg = self.config.theta_min_deg
            direction = 1.0

        return SearchPanUpdate(theta_deg=theta_deg, direction=direction)

    def update_samples(self, samples, target_base_xyz, locking=False):
        if target_base_xyz is None:
            return TargetSampleUpdate(
                locking=False,
                samples=[],
                lost_target=bool(locking),
            )

        collecting_started = not locking
        if collecting_started:
            samples = []

        samples = [np.asarray(sample, dtype=float) for sample in samples]
        samples.append(np.asarray(target_base_xyz, dtype=float))
        if len(samples) > self.config.lock_samples:
            samples = samples[-self.config.lock_samples :]

        sample_count = len(samples)
        if sample_count < self.config.lock_samples:
            return TargetSampleUpdate(
                locking=True,
                samples=samples,
                sample_count=sample_count,
                collecting_started=collecting_started,
            )

        stacked = np.vstack(samples)
        target_mean = stacked.mean(axis=0)
        max_std = float(stacked.std(axis=0).max())
        unstable = max_std > self.config.lock_max_std_m
        return TargetSampleUpdate(
            locking=True,
            samples=samples,
            sample_count=sample_count,
            collecting_started=collecting_started,
            lock_ready=not unstable,
            unstable=unstable,
            target_mean=target_mean,
            max_std=max_std,
        )

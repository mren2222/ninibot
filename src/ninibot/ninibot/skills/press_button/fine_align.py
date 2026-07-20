import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class FineAlignStepContext:
    visual_step_target: np.ndarray
    ik_step_target: np.ndarray
    distance: float
    max_step: float
    command_ahead: float

    def as_dict(self):
        return {
            "visual_step_target": self.visual_step_target,
            "ik_step_target": self.ik_step_target,
            "distance": self.distance,
            "max_step": self.max_step,
            "command_ahead": self.command_ahead,
        }


class FineAlignCommandAhead:
    def __init__(self, integral_enable, max_command_ahead):
        self.integral_enable = bool(integral_enable)
        self.max_command_ahead = max(0.0, float(max_command_ahead))
        self.command_ahead = 0.0
        self.command_direction = None

    def reset(self):
        self.command_ahead = 0.0
        self.command_direction = None

    def step_target(self, fk_tip_xyz, step_direction, max_step):
        fk_tip_xyz = np.asarray(fk_tip_xyz, dtype=float)
        step_direction = np.asarray(step_direction, dtype=float)
        max_step = float(max_step)

        if not self.integral_enable:
            return fk_tip_xyz + step_direction * max_step, max_step

        direction_norm = float(np.linalg.norm(step_direction))
        if direction_norm <= 1e-9:
            self.reset()
            return fk_tip_xyz.copy(), 0.0

        step_direction = step_direction / direction_norm
        if self.command_direction is not None:
            direction_dot = float(np.dot(self.command_direction, step_direction))
            if direction_dot < 0.5:
                self.command_ahead = 0.0

        self.command_direction = step_direction.copy()
        self.command_ahead += max_step
        if self.max_command_ahead > 0.0:
            self.command_ahead = min(self.command_ahead, self.max_command_ahead)

        return (
            fk_tip_xyz + step_direction * self.command_ahead,
            self.command_ahead,
        )


def average_samples(samples):
    if not samples:
        return None
    return np.vstack(samples).mean(axis=0)


def make_step_context(
    target_base_xyz,
    tip_base_xyz,
    fk_tip_xyz,
    command_ahead,
    max_step,
):
    target_base_xyz = np.asarray(target_base_xyz, dtype=float)
    tip_base_xyz = np.asarray(tip_base_xyz, dtype=float)
    error = target_base_xyz - tip_base_xyz
    distance = float(np.linalg.norm(error))
    if distance <= 1e-12:
        step_direction = np.zeros(3, dtype=float)
        max_step = 0.0
    else:
        max_step = min(float(max_step), distance)
        step_direction = error / distance
    visual_step_target = tip_base_xyz + step_direction * max_step
    ik_step_target, command_ahead_value = command_ahead.step_target(
        fk_tip_xyz,
        step_direction,
        max_step,
    )
    return FineAlignStepContext(
        visual_step_target=visual_step_target,
        ik_step_target=ik_step_target,
        distance=distance,
        max_step=max_step,
        command_ahead=command_ahead_value,
    )


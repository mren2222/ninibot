import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RampResult:
    target: list
    done: bool


def ramp_joint_goal(
    current_target,
    goal,
    joint_indices,
    control_rate_hz,
    speed_scale=1.0,
    base_speed_deg_s=30.0,
):
    if current_target is None or goal is None:
        return RampResult(None if current_target is None else list(current_target), False)

    target = list(current_target)
    dt = 1.0 / max(float(control_rate_hz), 1e-6)
    max_step = math.radians(float(base_speed_deg_s) * float(speed_scale)) * dt
    done = True
    for index in joint_indices:
        error = goal[index] - target[index]
        if abs(error) <= max_step or max_step <= 1e-9:
            target[index] = goal[index]
        else:
            target[index] += math.copysign(max_step, error)
            done = False

    if len(goal) > 4 and len(target) > 4:
        target[4] = goal[4]
    return RampResult(target, done)


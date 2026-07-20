GRIPPER_MIN_RAD = -0.174533
GRIPPER_MAX_RAD = 1.74533


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def minimum_jerk(u):
    u = clamp(float(u), 0.0, 1.0)
    return (10.0 * u * u * u) - (15.0 * u * u * u * u) + (
        6.0 * u * u * u * u * u
    )


def gripper_percent_to_rad(percent):
    percent = clamp(float(percent), 0.0, 100.0)
    return GRIPPER_MIN_RAD + (percent / 100.0) * (
        GRIPPER_MAX_RAD - GRIPPER_MIN_RAD
    )

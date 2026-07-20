import math

import cv2
import numpy as np


def draw_text(frame, text, y, color=(0, 255, 255)):
    cv2.putText(
        frame,
        text,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def format_cm(xyz):
    if xyz is None:
        return "(n/a, n/a, n/a)"
    return (
        f"({xyz[0] * 100.0:+.1f}, "
        f"{xyz[1] * 100.0:+.1f}, "
        f"{xyz[2] * 100.0:+.1f})"
    )


def distance_cm(a, b):
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) * 100.0)


def format_distance_cm(a, b):
    distance = distance_cm(a, b)
    if distance is None:
        return "n/a"
    return f"{distance:.1f}cm"


def format_degrees(values):
    return "[" + ", ".join(f"{math.degrees(value):+.1f}" for value in values) + "]"


def format_cm_delta(values):
    return "(" + ", ".join(f"{value * 100.0:+.1f}" for value in values) + ")"


def format_joint_target_deg(target):
    if target is None:
        return "(n/a)"
    return "[" + ", ".join(f"{math.degrees(value):+.1f}" for value in target[:4]) + "]"


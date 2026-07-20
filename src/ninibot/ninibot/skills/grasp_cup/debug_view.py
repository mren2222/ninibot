import math
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class DebugTelemetry:
    status: str
    radius_now: float
    theta_rad: float
    z_now: float
    approach_servo_started: bool
    approach_locked: bool
    tracking_settle_tolerance_px: float
    tracking_centered_count: int
    tracking_settle_cycles: int
    fine_align_centered_count: int
    fine_align_centered_cycles: int
    servo_radius_m: float
    servo_z_m: float
    approach_joint2_deg: float
    approach_joint3_deg: float
    joint4_target_rad: float
    prep_progress: float
    error_x: float
    bbox_w: float
    bbox_error: float
    target_bbox_width: float


def make_debug_text(data):
    return (
        f"status={data.status}, "
        f"radius_now={data.radius_now:.3f}, "
        f"theta_deg={math.degrees(data.theta_rad):.1f}, "
        f"z_now={data.z_now:.3f}, "
        f"approach_servo={data.approach_servo_started}, "
        f"approach_locked={data.approach_locked}, "
        f"center_tol={data.tracking_settle_tolerance_px:.1f}, "
        f"tracking_settle={data.tracking_centered_count}/{data.tracking_settle_cycles}, "
        f"fine_settle={data.fine_align_centered_count}/{data.fine_align_centered_cycles}, "
        f"servo_radius={0.0 if data.servo_radius_m is None else data.servo_radius_m:.3f}, "
        f"servo_z={0.0 if data.servo_z_m is None else data.servo_z_m:.3f}, "
        f"joint2_target={data.approach_joint2_deg:.1f}, "
        f"joint3_target={data.approach_joint3_deg:.1f}, "
        f"joint4_target={math.degrees(data.joint4_target_rad):.1f}, "
        f"prep_progress={data.prep_progress:.2f}, "
        f"error_x={0.0 if data.error_x is None else data.error_x:.1f}, "
        f"bbox_w={data.bbox_w:.1f}, "
        f"bbox_error={0.0 if data.bbox_error is None else data.bbox_error:.1f}"
    )


def draw_overlay(
    frame,
    detection,
    status,
    target_x,
    target_bbox_width,
    radius_now,
    theta_rad,
    z_now,
    error_x=None,
    bbox_error=None,
    bbox_w=None,
):
    height, width = frame.shape[:2]
    target_point = (int(target_x), height // 2)
    half_target = int(0.5 * target_bbox_width)
    target_left = target_point[0] - half_target
    target_top = target_point[1] - half_target
    target_right = target_point[0] + half_target
    target_bottom = target_point[1] + half_target

    cv2.drawMarker(
        frame,
        target_point,
        (255, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=34,
        thickness=2,
    )
    cv2.rectangle(
        frame,
        (target_left, target_top),
        (target_right, target_bottom),
        (255, 0, 255),
        2,
    )

    if detection is not None:
        x, y, w, h = detection["bbox"]
        cup_center = (int(detection["center_x"]), int(detection["center_y"]))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, cup_center, 6, (0, 255, 0), -1)
        cv2.line(frame, target_point, cup_center, (0, 255, 255), 2)

        if error_x is None:
            error_x = detection["center_x"] - target_x

    lines = [
        f"status={status}",
        f"radius={radius_now:.3f} m",
        f"theta={math.degrees(theta_rad):.1f} deg",
        f"z={z_now:.3f} m",
    ]

    lines.append(
        f"error_x={'n/a' if error_x is None else f'{error_x:.0f} px'}"
    )

    if bbox_error is not None:
        lines.append(f"bbox_error={bbox_error:.0f} px")

    if bbox_w is not None:
        lines.append(f"bbox_w={bbox_w:.0f} px")

    lines.append(f"target_w={target_bbox_width:.0f} px")

    for i, line in enumerate(lines):
        yy = 28 + i * 26

        cv2.putText(
            frame,
            line,
            (16, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            line,
            (16, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

import cv2
import numpy as np


def project_marker_offset_pixel(
    rvec,
    tvec,
    marker_offset_xyz,
    camera_matrix,
    dist_coeffs,
):
    if rvec is None or tvec is None:
        return None

    try:
        projected, _ = cv2.projectPoints(
            np.asarray(marker_offset_xyz, dtype=float).reshape(1, 3),
            np.asarray(rvec, dtype=float).reshape(3, 1),
            np.asarray(tvec, dtype=float).reshape(3, 1),
            camera_matrix,
            dist_coeffs,
        )
    except cv2.error:
        return None

    x, y = projected.reshape(-1, 2)[0]
    if not np.isfinite(x) or not np.isfinite(y):
        return None

    return int(round(x)), int(round(y))


def draw_marker_offset_target(
    frame,
    rvec,
    tvec,
    marker_offset_xyz,
    camera_matrix,
    dist_coeffs,
):
    pixel = project_marker_offset_pixel(
        rvec,
        tvec,
        marker_offset_xyz,
        camera_matrix,
        dist_coeffs,
    )
    if pixel is None:
        return None

    cv2.circle(frame, pixel, 8, (0, 0, 0), 2)
    cv2.circle(frame, pixel, 6, (0, 255, 0), -1)
    return pixel


def render_aruco_debug_frame(
    frame,
    camera_matrix,
    dist_coeffs,
    marker_offset_xyz,
    marker_length,
    corners=None,
    ids=None,
    rvec=None,
    tvec=None,
    status="",
):
    debug = frame.copy()
    if corners is not None and ids is not None:
        cv2.aruco.drawDetectedMarkers(debug, corners, ids)
    if rvec is not None and tvec is not None and hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(
            debug,
            camera_matrix,
            dist_coeffs,
            np.asarray(rvec, dtype=float).reshape(3, 1),
            np.asarray(tvec, dtype=float).reshape(3, 1),
            marker_length * 0.5,
        )

    draw_marker_offset_target(
        debug,
        rvec,
        tvec,
        marker_offset_xyz,
        camera_matrix,
        dist_coeffs,
    )

    if status:
        cv2.putText(
            debug,
            status,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return debug

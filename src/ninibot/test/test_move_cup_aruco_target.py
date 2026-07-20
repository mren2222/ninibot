import math

import numpy as np

from ninibot.skills.move_cup.aruco_target import estimate_target_from_marker_poses
from ninibot.skills.move_cup.aruco_target import target_camera_xyz_from_marker
from ninibot.skills.move_cup.aruco_target import target_marker_index
from ninibot.skills.move_cup.debug_view import project_marker_offset_pixel


def test_target_marker_index_uses_requested_id_or_first_marker():
    ids = np.asarray([[3], [7], [9]])

    assert target_marker_index(ids, 7) == 1
    assert target_marker_index(ids, -1) == 0
    assert target_marker_index(ids, 5) is None
    assert target_marker_index(None, 7) is None


def test_target_camera_xyz_applies_marker_rotation_and_offset():
    marker_xyz = np.asarray([1.0, 2.0, 3.0])
    offset = np.asarray([0.1, 0.0, 0.0])

    identity = target_camera_xyz_from_marker(marker_xyz, [0.0, 0.0, 0.0], offset)
    rotated = target_camera_xyz_from_marker(marker_xyz, [0.0, 0.0, math.pi / 2.0], offset)

    assert np.allclose(identity, [1.1, 2.0, 3.0])
    assert np.allclose(rotated, [1.0, 2.1, 3.0])


def test_estimate_target_from_marker_poses_selects_requested_marker():
    ids = np.asarray([[1], [2]])
    rvecs = [np.zeros(3), np.zeros(3)]
    tvecs = [np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 1.0])]

    estimate = estimate_target_from_marker_poses(
        ids,
        rvecs,
        tvecs,
        target_marker_id=2,
        marker_offset_xyz=[0.1, 0.0, 0.0],
    )

    assert np.allclose(estimate.marker_camera_xyz, [1.0, 0.0, 1.0])
    assert np.allclose(estimate.target_camera_xyz, [1.1, 0.0, 1.0])
    assert estimate_target_from_marker_poses(ids, rvecs, tvecs, 9, [0, 0, 0]) is None


def test_project_marker_offset_pixel_returns_image_point():
    camera_matrix = np.asarray(
        [
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 40.0],
            [0.0, 0.0, 1.0],
        ]
    )
    dist_coeffs = np.zeros(5)

    pixel = project_marker_offset_pixel(
        rvec=[0.0, 0.0, 0.0],
        tvec=[0.0, 0.0, 1.0],
        marker_offset_xyz=[0.1, 0.2, 0.0],
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )

    assert pixel == (60, 60)

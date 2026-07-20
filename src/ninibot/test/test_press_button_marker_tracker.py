import math

import cv2
import numpy as np

from ninibot.skills.press_button.marker_tracker import MarkerTracker


def fake_pose_estimator(corners, marker_length, camera_matrix, dist_coeffs):
    rvec = np.asarray([[0.0], [0.0], [math.pi / 2.0]], dtype=float)
    tvec = np.asarray([1.0, 2.0, 3.0], dtype=float)
    return [rvec], [tvec]


def make_tracker(target_marker_id=4):
    return MarkerTracker(
        target_marker_id=target_marker_id,
        marker_length=0.05,
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros(5),
        target_marker_offset_xyz=np.asarray([0.10, 0.0, 0.0], dtype=float),
        pose_estimator=fake_pose_estimator,
    )


def test_marker_tracker_selects_target_marker_id():
    tracker = make_tracker(target_marker_id=4)

    index = tracker.select_target_index(np.asarray([[2], [4], [6]]))

    assert index == 1


def test_marker_tracker_negative_target_accepts_first_marker():
    tracker = make_tracker(target_marker_id=-1)

    index = tracker.select_target_index(np.asarray([[8], [9]]))

    assert index == 0


def test_marker_tracker_rotates_target_offset_into_camera_frame():
    tracker = make_tracker(target_marker_id=4)
    ids = np.asarray([[4]])

    result = tracker.track(corners=[np.zeros((4, 2))], ids=ids)

    assert result.reason == ""
    assert np.allclose(result.observation.marker_camera_xyz, [1.0, 2.0, 3.0])
    assert np.allclose(result.observation.target_camera_xyz, [1.0, 2.1, 3.0])


def test_marker_tracker_reports_missing_target():
    tracker = make_tracker(target_marker_id=7)

    result = tracker.track(corners=[np.zeros((4, 2))], ids=np.asarray([[4]]))

    assert result.observation is None
    assert result.reason == "target_not_visible"


def test_marker_tracker_target_offset_matches_cv_rodrigues():
    tracker = make_tracker(target_marker_id=4)
    rvec = np.asarray([[0.0], [0.0], [math.pi / 2.0]], dtype=float)
    rotation, _ = cv2.Rodrigues(rvec)

    target = tracker.target_camera_xyz(np.asarray([1.0, 2.0, 3.0]), rvec)

    assert np.allclose(target, np.asarray([1.0, 2.0, 3.0]) + rotation @ [0.1, 0.0, 0.0])


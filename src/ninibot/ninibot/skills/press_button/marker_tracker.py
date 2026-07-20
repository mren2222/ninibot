from dataclasses import dataclass

import numpy as np

from ninibot.skills.common.aruco_target import target_camera_xyz_from_marker
from ninibot.skills.common.aruco_target import target_marker_index
from vision_controller.aruco_detection import estimate_marker_poses


@dataclass(frozen=True)
class MarkerObservation:
    target_index: int
    marker_camera_xyz: np.ndarray
    target_camera_xyz: np.ndarray
    marker_rvec: np.ndarray
    marker_tvec: np.ndarray


@dataclass(frozen=True)
class MarkerTrackResult:
    observation: MarkerObservation | None
    reason: str = ""


class MarkerTracker:
    def __init__(
        self,
        target_marker_id,
        marker_length,
        camera_matrix,
        dist_coeffs,
        target_marker_offset_xyz,
        pose_estimator=estimate_marker_poses,
    ):
        self.target_marker_id = int(target_marker_id)
        self.marker_length = float(marker_length)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.target_marker_offset_xyz = np.asarray(
            target_marker_offset_xyz,
            dtype=float,
        )
        self.pose_estimator = pose_estimator

    def select_target_index(self, ids):
        return target_marker_index(ids, self.target_marker_id)

    def target_camera_xyz(self, marker_camera_xyz, marker_rvec):
        return target_camera_xyz_from_marker(
            marker_camera_xyz,
            marker_rvec,
            self.target_marker_offset_xyz,
        )

    def track(self, corners, ids):
        if ids is None:
            return MarkerTrackResult(None, "no_markers")

        target_index = self.select_target_index(ids)
        if target_index is None:
            return MarkerTrackResult(None, "target_not_visible")

        rvecs, tvecs = self.pose_estimator(
            corners,
            self.marker_length,
            self.camera_matrix,
            self.dist_coeffs,
        )
        if target_index >= len(tvecs):
            return MarkerTrackResult(None, "target_has_no_pose")

        marker_rvec = rvecs[target_index] if target_index < len(rvecs) else None
        if marker_rvec is None:
            return MarkerTrackResult(None, "target_has_no_pose")

        marker_camera_xyz = np.asarray(tvecs[target_index], dtype=float)
        target_camera_xyz = self.target_camera_xyz(marker_camera_xyz, marker_rvec)
        return MarkerTrackResult(
            MarkerObservation(
                target_index=target_index,
                marker_camera_xyz=marker_camera_xyz,
                target_camera_xyz=target_camera_xyz,
                marker_rvec=marker_rvec,
                marker_tvec=tvecs[target_index],
            ),
            "",
        )

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ArucoTargetEstimate:
    marker_camera_xyz: object
    marker_rvec: object
    target_camera_xyz: object


def target_marker_index(ids, target_marker_id):
    if ids is None:
        return None

    for index, marker_id in enumerate(np.asarray(ids).flatten()):
        if target_marker_id < 0 or int(marker_id) == target_marker_id:
            return index

    return None


def target_camera_xyz_from_marker(marker_camera_xyz, marker_rvec, marker_offset_xyz):
    marker_camera_xyz = np.asarray(marker_camera_xyz, dtype=float).reshape(3)
    marker_rvec = np.asarray(marker_rvec, dtype=float).reshape(3)
    marker_offset_xyz = np.asarray(marker_offset_xyz, dtype=float).reshape(3)
    marker_rotation_camera, _ = cv2.Rodrigues(marker_rvec.reshape(3, 1))
    return marker_camera_xyz + marker_rotation_camera @ marker_offset_xyz


def estimate_target_from_marker_poses(
    ids,
    rvecs,
    tvecs,
    target_marker_id,
    marker_offset_xyz,
):
    index = target_marker_index(ids, target_marker_id)
    if index is None or index >= len(tvecs) or index >= len(rvecs):
        return None

    marker_camera_xyz = np.asarray(tvecs[index], dtype=float).reshape(3)
    marker_rvec = np.asarray(rvecs[index], dtype=float).reshape(3)
    target_camera_xyz = target_camera_xyz_from_marker(
        marker_camera_xyz,
        marker_rvec,
        marker_offset_xyz,
    )
    return ArucoTargetEstimate(
        marker_camera_xyz=marker_camera_xyz,
        marker_rvec=marker_rvec,
        target_camera_xyz=target_camera_xyz,
    )

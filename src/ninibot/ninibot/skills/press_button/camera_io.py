import cv2

from vision_controller.aruco_detection import open_camera as open_cv_camera
from vision_controller.aruco_tf import transform_camera_point_to_base_xyz


class PressButtonCameraIo:
    def __init__(self, node):
        self.node = node

    def open_camera(self):
        node = self.node
        if node.cap is not None and node.cap.isOpened():
            return True

        node.cap = open_cv_camera(
            node.camera_source,
            node.capture_width,
            node.capture_height,
            30,
        )
        if node.cap.isOpened():
            node.get_logger().info(f"press button: opened camera source {node.camera_source}")
            return True

        self.release_camera()
        node.log_progress(f"press button: waiting for camera source {node.camera_source}")
        return False

    def release_camera(self):
        node = self.node
        if node.cap is not None:
            node.cap.release()
            node.cap = None

    def show_locked_marker_debug(self):
        node = self.node
        node.marker_camera_xyz = node.locked_marker_camera_xyz
        node.marker_base_xyz = node.locked_marker_base_xyz
        node.marker_target_base_xyz = node.locked_marker_target_base_xyz

        if not self.open_camera():
            return
        ok, frame = node.cap.read()
        if not ok:
            node.log_progress("press button: cannot read camera frame")
            return

        corners = None
        ids = None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = node.detect_markers(gray)
        except cv2.error:
            pass

        node.show_debug(
            frame,
            corners,
            ids,
            status=f"locked; auto move in {node.lock_wait_sec:.1f}s",
        )

    def detect_marker(self):
        node = self.node
        if not self.open_camera():
            return None
        ok, frame = node.cap.read()
        if not ok:
            node.log_progress("press button: cannot read camera frame")
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = node.detect_markers(gray)
        if ids is None:
            node.marker_target_base_xyz = None
            node.show_debug(frame, status="waiting for ArUco")
            return None

        track = node.marker_tracker.track(corners, ids)
        if track.observation is None:
            node.marker_target_base_xyz = None
            if track.reason == "target_has_no_pose":
                status = "target marker has no pose"
            else:
                status = f"target {node.target_marker_id} not visible"
            node.show_debug(frame, corners, ids, status=status)
            return None

        observation = track.observation
        marker_camera_xyz = observation.marker_camera_xyz
        target_camera_xyz = observation.target_camera_xyz
        marker_rvec = observation.marker_rvec
        marker_tvec = observation.marker_tvec
        node.marker_camera_xyz = marker_camera_xyz
        try:
            marker_base_xyz = transform_camera_point_to_base_xyz(
                node,
                node.tf_buffer,
                marker_camera_xyz,
                node.base_frame,
                node.camera_frame,
                node.tf_timeout_sec,
                spin=False,
            )
            marker_target_base_xyz = transform_camera_point_to_base_xyz(
                node,
                node.tf_buffer,
                target_camera_xyz,
                node.base_frame,
                node.camera_frame,
                node.tf_timeout_sec,
                spin=False,
            )
        except Exception as exc:
            node.show_debug(
                frame,
                corners,
                ids,
                marker_rvec,
                marker_tvec,
                status="waiting for TF",
            )
            node.log_progress(f"press button: waiting for TF: {exc}")
            return None

        node.marker_base_xyz = marker_base_xyz
        node.marker_target_base_xyz = marker_target_base_xyz
        node.show_debug(
            frame,
            corners,
            ids,
            marker_rvec,
            marker_tvec,
            status=f"target={node.format_cm(marker_target_base_xyz)}",
        )
        return marker_base_xyz


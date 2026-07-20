import math
import time

import cv2
import numpy as np
from geometry_msgs.msg import PointStamped

from ninibot.skills.common.aruco_target import estimate_target_from_marker_poses
from ninibot.skills.move_cup.debug_view import draw_marker_offset_target
from ninibot.skills.move_cup.debug_view import render_aruco_debug_frame
from ninibot.skills.move_cup.phases import SEARCH_TARGET
from ninibot.skills.move_cup.phases import TARGET_LOCKED
from vision_controller.aruco_detection import destroy_windows
from vision_controller.aruco_detection import estimate_marker_poses
from vision_controller.aruco_detection import load_camera_info
from vision_controller.aruco_detection import make_detector
from vision_controller.aruco_detection import open_camera as open_cv_camera
from vision_controller.aruco_tf import transform_camera_point_to_base_xyz


class MoveCupTargetRuntime:
    def __init__(self, node):
        self.node = node

    def ensure_aruco_ready(self):
        node = self.node
        if node.detect_markers is not None:
            return True

        try:
            (
                node.calibration_width,
                node.calibration_height,
                node.camera_matrix,
                node.dist_coeffs,
            ) = load_camera_info(node.calibration_path)
            node.detect_markers = make_detector(node.aruco_dictionary)
            node.camera_ok = True
            return True
        except Exception as exc:
            node.camera_matrix = None
            node.dist_coeffs = None
            node.detect_markers = None
            node.camera_ok = False
            node.last_target_error = f"failed to load ArUco calibration: {exc}"
            node.get_logger().error(f"move cup: {node.last_target_error}")
            return False

    def open_camera(self):
        node = self.node
        if node.cap is not None and node.cap.isOpened():
            return True

        if not self.ensure_aruco_ready():
            return False

        node.cap = open_cv_camera(
            node.camera_source,
            node.frame_width,
            node.frame_height,
            30,
        )
        if node.cap.isOpened():
            node.get_logger().info(f"move cup: opened camera {node.camera_source}")
            return True

        self.release_camera()
        node.log_progress(f"move cup: waiting for camera {node.camera_source}")
        return False

    def release_camera(self):
        node = self.node
        if node.cap is not None:
            node.cap.release()
            node.cap = None
        if node.aruco_debug_window and not node.debug_window_failed:
            destroy_windows()

    def publish_target_base(self, target_base_xyz):
        node = self.node
        msg = PointStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = node.base_frame
        msg.point.x = float(target_base_xyz[0])
        msg.point.y = float(target_base_xyz[1])
        msg.point.z = float(target_base_xyz[2])
        node.target_base_pub.publish(msg)

    def apply_locked_target(self, target_base_xyz):
        node = self.node
        plan = node.pose_planner.locked_target_from_base_xyz(
            target_base_xyz,
            node.release_z(),
        )
        if plan.warning_text:
            node.get_logger().warn(plan.warning_text)
        if not plan.success:
            node.last_target_error = plan.error_text
            node.get_logger().warn(f"move cup: {node.last_target_error}")
            return False

        node.target_radius = plan.radius
        node.target_theta_deg = plan.theta_deg
        node.target_z = plan.z
        node.last_target_error = ""
        return True

    def clear_locked_target(self):
        node = self.node
        node.target_samples = []
        node.locked_target_base_xyz = None
        node.locked_marker_base_xyz = None
        node.locked_marker_camera_xyz = None
        node.target_locking = False
        node.search_prep_trajectory = None
        node.return_search_prep_trajectory = None
        node.last_target_error = ""

    def lock_target(self, target_base_xyz, marker_base_xyz, marker_camera_xyz):
        node = self.node
        target_base_xyz = np.asarray(target_base_xyz, dtype=float).copy()
        target_base_xyz[2] = node.release_z()

        if not self.apply_locked_target(target_base_xyz):
            node.phase = "FAILED"
            self.release_camera()
            node.publish_status(node.phase)
            return

        target_theta_rad = math.radians(node.target_theta_deg)
        effective_target_base_xyz = np.asarray(
            [
                node.target_radius * math.cos(target_theta_rad),
                node.target_radius * math.sin(target_theta_rad),
                node.target_z,
            ],
            dtype=float,
        )
        node.locked_target_base_xyz = effective_target_base_xyz
        node.locked_marker_base_xyz = np.asarray(marker_base_xyz, dtype=float)
        node.locked_marker_camera_xyz = np.asarray(marker_camera_xyz, dtype=float)
        self.publish_target_base(effective_target_base_xyz)
        self.release_camera()

        node.phase = TARGET_LOCKED
        node.current_target = None
        node.clear_path()
        node.phase_started_at = time.monotonic()
        node.last_progress_log_time = 0.0
        node.publish_status(node.phase)
        node.get_logger().info(
            "move cup: TARGET_LOCKED "
            f"marker_base={node.format_cm(node.locked_marker_base_xyz)} "
            f"target_base={node.format_cm(node.locked_target_base_xyz)} "
            f"radius={node.target_radius:.3f}m "
            f"theta={node.target_theta_deg:.1f}deg "
            f"target_z={node.target_z:+.3f}m"
        )

    def draw_marker_offset_target(self, frame, rvec, tvec):
        node = self.node
        return draw_marker_offset_target(
            frame,
            rvec,
            tvec,
            node.target_marker_offset_xyz,
            node.camera_matrix,
            node.dist_coeffs,
        )

    def show_aruco_debug(
        self,
        frame,
        corners=None,
        ids=None,
        rvec=None,
        tvec=None,
        status="",
    ):
        node = self.node
        if not node.aruco_debug_window or node.debug_window_failed:
            return

        debug = render_aruco_debug_frame(
            frame,
            node.camera_matrix,
            node.dist_coeffs,
            node.target_marker_offset_xyz,
            node.marker_length,
            corners=corners,
            ids=ids,
            rvec=rvec,
            tvec=tvec,
            status=status,
        )

        try:
            cv2.imshow(node.aruco_debug_window_name, debug)
            cv2.waitKey(1)
        except cv2.error as exc:
            node.debug_window_failed = True
            node.get_logger().warn(f"OpenCV ArUco debug window unavailable: {exc}")

    def detect_target(self):
        node = self.node
        if not self.open_camera():
            return None

        ok, frame = node.cap.read()
        if not ok:
            node.log_progress("move cup: cannot read camera frame")
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = node.detect_markers(gray)
        if ids is None:
            self.show_aruco_debug(frame, status="waiting for ArUco")
            return None

        rvecs, tvecs = estimate_marker_poses(
            corners,
            node.marker_length,
            node.camera_matrix,
            node.dist_coeffs,
        )
        estimate = estimate_target_from_marker_poses(
            ids,
            rvecs,
            tvecs,
            node.target_marker_id,
            node.target_marker_offset_xyz,
        )
        if estimate is None:
            self.show_aruco_debug(
                frame,
                corners,
                ids,
                status=f"target {node.target_marker_id} not visible",
            )
            return None

        try:
            marker_base_xyz = transform_camera_point_to_base_xyz(
                node,
                node.tf_buffer,
                estimate.marker_camera_xyz,
                node.base_frame,
                node.camera_frame,
                node.tf_timeout_sec,
                spin=False,
            )
            target_base_xyz = transform_camera_point_to_base_xyz(
                node,
                node.tf_buffer,
                estimate.target_camera_xyz,
                node.base_frame,
                node.camera_frame,
                node.tf_timeout_sec,
                spin=False,
            )
        except Exception as exc:
            self.show_aruco_debug(
                frame,
                corners,
                ids,
                estimate.marker_rvec,
                estimate.marker_camera_xyz,
                status="waiting for TF",
            )
            node.log_progress(f"move cup: waiting for TF: {exc}")
            return None

        target_base_xyz = np.asarray(target_base_xyz, dtype=float).copy()
        target_base_xyz[2] = node.release_z()

        self.show_aruco_debug(
            frame,
            corners,
            ids,
            estimate.marker_rvec,
            estimate.marker_camera_xyz,
            status=f"target={node.format_cm(target_base_xyz)}",
        )
        return {
            "target_base_xyz": target_base_xyz,
            "marker_base_xyz": np.asarray(marker_base_xyz, dtype=float),
            "marker_camera_xyz": estimate.marker_camera_xyz,
        }

    def update_target_search_pan(self):
        node = self.node
        update = node.target_scan_planner.next_pan(
            node.search_theta_deg,
            node.search_direction,
            locking=node.target_locking,
        )
        node.search_theta_deg = update.theta_deg
        node.search_direction = update.direction

    def update_target_search(self):
        node = self.node
        if (
            node.target_search_timeout_sec > 0.0
            and node.phase_elapsed_sec() >= node.target_search_timeout_sec
        ):
            node.phase = "FAILED"
            node.current_target = None
            self.release_camera()
            node.publish_status(node.phase)
            node.get_logger().warn(
                "move cup: failed to lock ArUco target after "
                f"{node.target_search_timeout_sec:.1f}s"
            )
            return

        self.update_target_search_pan()
        node.current_target = node.make_target_search_pose(node.search_theta_deg)
        if node.current_target is None:
            node.phase = "FAILED"
            self.release_camera()
            node.publish_status(node.phase)
            node.get_logger().warn("move cup: invalid ArUco search target")
            return

        detection = self.detect_target()
        target_base_xyz = None if detection is None else detection["target_base_xyz"]
        sample_update = node.target_scan_planner.update_samples(
            node.target_samples,
            target_base_xyz,
            locking=node.target_locking,
        )
        if sample_update.lost_target:
            node.log_progress("move cup: lost ArUco target; resuming search")
        node.target_locking = sample_update.locking
        node.target_samples = sample_update.samples

        if detection is None:
            return

        if sample_update.collecting_started:
            node.get_logger().info(
                "move cup: ArUco target visible; collecting stable samples"
            )

        if sample_update.sample_count < node.target_lock_samples:
            node.log_progress(
                "move cup: collecting ArUco target samples "
                f"{sample_update.sample_count}/{node.target_lock_samples} "
                f"latest={node.format_cm(detection['target_base_xyz'])}"
            )
            return

        if sample_update.unstable:
            node.log_progress(
                "move cup: ArUco target samples unstable "
                f"max_std={sample_update.max_std * 100.0:.1f}cm "
                f"limit={node.target_lock_max_std_m * 100.0:.1f}cm "
                f"mean={node.format_cm(sample_update.target_mean)}"
            )
            return

        self.lock_target(
            sample_update.target_mean,
            detection["marker_base_xyz"],
            detection["marker_camera_xyz"],
        )

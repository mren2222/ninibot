import cv2
import numpy as np

from sensor_msgs.msg import Image

from ninibot.skills.press_button import debug_view


class DebugRenderer:
    def __init__(self, node):
        self.node = node

    def publish_debug_frame(self, frame):
        node = self.node
        if node.debug_image_pub is None:
            return
        image = np.ascontiguousarray(frame)
        msg = Image()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = node.camera_frame
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = image.shape[1] * image.shape[2]
        msg.data = image.tobytes()
        node.debug_image_pub.publish(msg)

    def draw_text(self, frame, text, y, color=(0, 255, 255)):
        debug_view.draw_text(frame, text, y, color)

    def draw_marker_offset_target(self, frame, rvec, tvec):
        node = self.node
        if rvec is None or tvec is None:
            return
        try:
            projected, _ = cv2.projectPoints(
                node.target_marker_offset_xyz.reshape(1, 3),
                rvec.reshape(3, 1),
                tvec.reshape(3, 1),
                node.camera_matrix,
                node.dist_coeffs,
            )
        except cv2.error:
            return

        x, y = projected.reshape(-1, 2)[0]
        if not np.isfinite(x) or not np.isfinite(y):
            return
        pixel = (int(round(x)), int(round(y)))
        cv2.circle(frame, pixel, 8, (0, 0, 0), 2)
        cv2.circle(frame, pixel, 6, (0, 255, 0), -1)

    def draw_grippertip_camera_point(self, frame):
        node = self.node
        gripper_camera_xyz = node.lookup_gripper_camera_xyz()
        if gripper_camera_xyz is None or gripper_camera_xyz[2] <= 0.0:
            return
        try:
            projected, _ = cv2.projectPoints(
                gripper_camera_xyz.reshape(1, 3),
                np.zeros((3, 1), dtype=float),
                np.zeros((3, 1), dtype=float),
                node.camera_matrix,
                node.dist_coeffs,
            )
        except cv2.error:
            return

        x, y = projected.reshape(-1, 2)[0]
        if not np.isfinite(x) or not np.isfinite(y):
            return
        pixel = (int(round(x)), int(round(y)))
        cv2.circle(frame, pixel, 8, (0, 0, 0), 2)
        cv2.circle(frame, pixel, 6, (255, 0, 255), -1)

    def show_debug(self, frame, corners=None, ids=None, rvec=None, tvec=None, status=""):
        node = self.node
        debug = frame.copy()
        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(debug, corners, ids)
        if rvec is not None and tvec is not None and hasattr(cv2, "drawFrameAxes"):
            cv2.drawFrameAxes(
                debug,
                node.camera_matrix,
                node.dist_coeffs,
                rvec.reshape(3, 1),
                tvec.reshape(3, 1),
                node.marker_length * 0.5,
            )
        self.draw_marker_offset_target(debug, rvec, tvec)
        self.draw_grippertip_camera_point(debug)

        node.gripper_base_xyz = node.lookup_gripper_base_xyz()
        status_text = f" | {status}" if status else ""
        lines = [
            f"phase={node.phase}{status_text}",
            f"{node.gripper_frame}={node.format_cm(node.gripper_base_xyz)}",
            node.format_tip_target_distance(),
        ]
        for index, line in enumerate(lines):
            self.draw_text(debug, line, 30 + index * 28, (0, 255, 255))

        self.publish_debug_frame(debug)
        if node.debug_window and not node.debug_window_failed:
            try:
                cv2.imshow(node.debug_window_name, debug)
                key = cv2.waitKey(1)
                if key != -1:
                    node.last_key = key & 0xFF
            except cv2.error as exc:
                node.debug_window_failed = True
                node.get_logger().warn(f"OpenCV debug window unavailable: {exc}")

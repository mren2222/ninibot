# flake8: noqa
"""ArUco marker-to-base debug tool using ROS TF."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

try:
    from geometry_msgs.msg import PointStamped
    import rclpy
    from sensor_msgs.msg import Image
    import tf2_ros
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This tool needs ROS 2 Python packages. Source ROS first, for example: "
        "source /opt/ros/jazzy/setup.bash"
    ) from exc

TOOL_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = TOOL_DIR.parents[1]
VISION_SRC = WORKSPACE_ROOT / "src" / "vision_controller"
if VISION_SRC.exists():
    sys.path.insert(0, str(VISION_SRC))
NINIBOT_SRC = WORKSPACE_ROOT / "src" / "ninibot"
if NINIBOT_SRC.exists():
    sys.path.insert(0, str(NINIBOT_SRC))

from vision_controller.aruco_detection import ARUCO_DICTIONARIES
from vision_controller.aruco_detection import destroy_windows
from vision_controller.aruco_detection import draw_text
from vision_controller.aruco_detection import estimate_marker_poses
from vision_controller.aruco_detection import load_camera_info
from vision_controller.aruco_detection import load_flat_ros_params
from vision_controller.aruco_detection import make_detector
from vision_controller.aruco_detection import open_camera
from vision_controller.aruco_detection import resolve_path
from vision_controller.aruco_detection import show_window
from vision_controller.aruco_tf import lookup_transform_matrix
from vision_controller.aruco_tf import transform_camera_point_to_base_xyz
from ninibot.task_params import PRESS_BUTTON_PARAMS


DATA_DIR = WORKSPACE_ROOT / "data" / "calibration"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print target ArUco xyz in robot base coordinates. "
            "The marker tvec is interpreted in the camera optical frame, then "
            "converted through ROS TF."
        )
    )
    parser.add_argument("--calibration", default=str(DATA_DIR / "camera_calibration.yaml"))
    parser.add_argument(
        "--press-button-config",
        default=None,
        help="Optional legacy YAML override; defaults come from task_params.PRESS_BUTTON_PARAMS.",
    )
    parser.add_argument("--camera-id", type=int, help="OpenCV camera index used when --camera-device is empty.")
    parser.add_argument(
        "--camera-device",
        default=None,
        help="Stable OpenCV camera device path, for example /dev/v4l/by-id/...",
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="", help="Optional forced FOURCC, for example YUYV or MJPG.")
    parser.add_argument("--marker-length", type=float)
    parser.add_argument("--dictionary", choices=sorted(ARUCO_DICTIONARIES))
    parser.add_argument("--target-marker-id", type=int)
    parser.add_argument("--base-frame")
    parser.add_argument("--camera-frame")
    parser.add_argument("--gripper-frame")
    parser.add_argument("--tf-timeout", type=float)
    parser.add_argument(
        "--camera-base-z-offset",
        type=float,
        default=None,
        help="Optional debug base-frame Z offset after TF conversion; normally 0.0.",
    )
    parser.add_argument("--image")
    parser.add_argument("--output")
    parser.add_argument("--save-last")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--debug-image-topic", default="/aruco_extrinsics/debug_image")
    parser.add_argument("--marker-base-topic", default="/aruco_extrinsics/marker_base")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--print-every", type=float, default=0.5)
    return parser


def resolve_runtime(args):
    config_path = None
    params = dict(PRESS_BUTTON_PARAMS)
    if args.press_button_config:
        config_path = resolve_path(args.press_button_config)
        params.update(load_flat_ros_params(config_path))

    camera_id = args.camera_id if args.camera_id is not None else int(params.get("camera_index", 0))
    camera_device = args.camera_device
    if camera_device is None:
        camera_device = str(params.get("camera_device", "")).strip()
    else:
        camera_device = str(camera_device).strip()
    camera_source = camera_device if camera_device else camera_id
    marker_length = args.marker_length if args.marker_length is not None else float(params.get("marker_length", 0.05))
    dictionary = args.dictionary or str(params.get("aruco_dictionary", "4X4_50"))
    marker_id = args.target_marker_id if args.target_marker_id is not None else int(params.get("target_marker_id", 0))
    base_frame = args.base_frame or str(params.get("base_frame", "base"))
    camera_frame = args.camera_frame or str(params.get("camera_frame", "camera_optical_frame"))
    gripper_frame = args.gripper_frame or str(params.get("gripper_frame", "grippertip"))
    tf_timeout = args.tf_timeout if args.tf_timeout is not None else float(params.get("tf_timeout_sec", 0.5))
    z_offset = args.camera_base_z_offset
    if z_offset is None:
        z_offset = float(params.get("camera_base_z_offset", 0.0))

    return (
        config_path,
        camera_source,
        marker_length,
        dictionary,
        marker_id,
        base_frame,
        camera_frame,
        gripper_frame,
        tf_timeout,
        z_offset,
    )


def publish_debug_image(node, publisher, frame: np.ndarray, frame_id: str) -> None:
    if publisher is None:
        return

    image = np.ascontiguousarray(frame)
    msg = Image()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.height = image.shape[0]
    msg.width = image.shape[1]
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = image.shape[1] * image.shape[2]
    msg.data = image.tobytes()
    publisher.publish(msg)


def publish_point(node, publisher, xyz, frame_id: str) -> None:
    if publisher is None or xyz is None:
        return

    point = PointStamped()
    point.header.stamp = node.get_clock().now().to_msg()
    point.header.frame_id = frame_id
    point.point.x = float(xyz[0])
    point.point.y = float(xyz[1])
    point.point.z = float(xyz[2])
    publisher.publish(point)


def meters_to_cm(vector):
    if vector is None:
        return None
    return np.asarray(vector, dtype=float) * 100.0


def format_xyz(vector, digits: int = 1, unit: str = "cm") -> str:
    if vector is None:
        return f"(n/a, n/a, n/a){unit}"
    value = meters_to_cm(vector)
    return f"({value[0]:.{digits}f}, {value[1]:.{digits}f}, {value[2]:.{digits}f}){unit}"


def format_axis(vector) -> str:
    if vector is None:
        return "(n/a, n/a, n/a)"
    return f"({vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f})"


def draw_debug_overlay(
    frame,
    base_frame: str,
    target_marker_id: int,
    camera_in_base_xyz,
    camera_axes_in_base,
    gripper_base_xyz=None,
    marker_base_xyz=None,
    marker_camera_xyz=None,
    visible_marker_id=None,
    camera_base_z_offset: float = 0.0,
    status: str | None = None,
) -> None:
    marker_label = (
        f"id={visible_marker_id}"
        if visible_marker_id is not None
        else f"id=n/a target={target_marker_id}"
    )
    lines = [
        f"{marker_label} base_frame={base_frame}",
        f"marker_base_xyz={format_xyz(marker_base_xyz)}",
        f"marker_camera_xyz={format_xyz(marker_camera_xyz)}",
        f"grippertip_base_xyz={format_xyz(gripper_base_xyz)}",
        f"camera_in_base_xyz={format_xyz(camera_in_base_xyz)}",
        f"camera_base_z_offset={camera_base_z_offset * 100.0:+.1f}cm",
        f"camera_x_axis_base={format_axis(camera_axes_in_base[0])}",
        f"camera_y_axis_base={format_axis(camera_axes_in_base[1])}",
        f"camera_z_axis_base={format_axis(camera_axes_in_base[2])}",
    ]
    if status:
        lines.append(status)

    for line_index, line in enumerate(lines):
        color = (0, 0, 255) if status and line_index == len(lines) - 1 else (0, 255, 255)
        draw_text(frame, line, 30 + line_index * 28, color)


def check_frame(
    frame,
    detector,
    marker_length,
    target_marker_id,
    camera_matrix,
    dist_coeffs,
    tf_node,
    tf_buffer,
    t_base_camera,
    gripper_base_xyz,
    base_frame,
    camera_frame,
    tf_timeout,
    camera_base_z_offset,
):
    base_offset = np.array([0.0, 0.0, camera_base_z_offset], dtype=float)
    camera_in_base_xyz = t_base_camera[:3, 3] + base_offset
    camera_axes_in_base = [t_base_camera[:3, axis] for axis in range(3)]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector(gray)
    if ids is None:
        draw_debug_overlay(
            frame,
            base_frame,
            target_marker_id,
            camera_in_base_xyz,
            camera_axes_in_base,
            gripper_base_xyz=gripper_base_xyz,
            camera_base_z_offset=camera_base_z_offset,
            status="No ArUco marker detected",
        )
        return None

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    rvecs, tvecs = estimate_marker_poses(corners, marker_length, camera_matrix, dist_coeffs)

    for index, marker_id in enumerate(ids.flatten()):
        if int(marker_id) != target_marker_id or index >= len(tvecs):
            continue

        marker_camera_xyz = tvecs[index].astype(float)
        marker_base_xyz = transform_camera_point_to_base_xyz(
            tf_node,
            tf_buffer,
            marker_camera_xyz,
            base_frame,
            camera_frame,
            tf_timeout,
            base_xyz_offset=base_offset,
        )

        if hasattr(cv2, "drawFrameAxes"):
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                rvecs[index].reshape(3, 1),
                tvecs[index].reshape(3, 1),
                marker_length * 0.5,
            )

        draw_debug_overlay(
            frame,
            base_frame,
            target_marker_id,
            camera_in_base_xyz,
            camera_axes_in_base,
            gripper_base_xyz=gripper_base_xyz,
            marker_base_xyz=marker_base_xyz,
            marker_camera_xyz=marker_camera_xyz,
            visible_marker_id=int(marker_id),
            camera_base_z_offset=camera_base_z_offset,
        )
        lines = [
            f"id={int(marker_id)} base_frame={base_frame}",
            f"marker_base_xyz={format_xyz(marker_base_xyz)}",
            f"marker_camera_xyz={format_xyz(marker_camera_xyz)}",
            f"grippertip_base_xyz={format_xyz(gripper_base_xyz)}",
            f"camera_in_base_xyz={format_xyz(camera_in_base_xyz)}",
            f"camera_base_z_offset={camera_base_z_offset * 100.0:+.1f}cm",
            f"camera_x_axis_base={format_axis(camera_axes_in_base[0])}",
            f"camera_y_axis_base={format_axis(camera_axes_in_base[1])}",
            f"camera_z_axis_base={format_axis(camera_axes_in_base[2])}",
        ]
        return (
            marker_base_xyz.astype(float),
            marker_camera_xyz.astype(float),
            camera_in_base_xyz.astype(float),
            None if gripper_base_xyz is None else np.asarray(gripper_base_xyz, dtype=float),
            " | ".join(lines),
        )

    draw_debug_overlay(
        frame,
        base_frame,
        target_marker_id,
        camera_in_base_xyz,
        camera_axes_in_base,
        gripper_base_xyz=gripper_base_xyz,
        camera_base_z_offset=camera_base_z_offset,
        status=f"Target marker {target_marker_id} not visible",
    )
    return None


def summarize(results):
    if not results:
        print("No usable target marker observations.")
        return 1

    base = np.vstack([item[0] for item in results])
    marker_camera = np.vstack([item[1] for item in results])
    camera_base = np.vstack([item[2] for item in results])
    gripper_base_values = [item[3] for item in results if item[3] is not None]
    base_mean = meters_to_cm(base.mean(axis=0))
    base_std = meters_to_cm(base.std(axis=0))
    marker_camera_mean = meters_to_cm(marker_camera.mean(axis=0))
    camera_base_mean = meters_to_cm(camera_base.mean(axis=0))
    camera_base_std = meters_to_cm(camera_base.std(axis=0))
    print("\nSummary:")
    print(f"  observations={len(results)}")
    print(f"  marker_base_xyz_cm mean=({base_mean[0]:.1f}, {base_mean[1]:.1f}, {base_mean[2]:.1f})")
    print(f"  marker_base_xyz_cm std =({base_std[0]:.2f}, {base_std[1]:.2f}, {base_std[2]:.2f})")
    print(
        "  marker_camera_xyz_cm "
        f"mean=({marker_camera_mean[0]:.1f}, {marker_camera_mean[1]:.1f}, "
        f"{marker_camera_mean[2]:.1f})"
    )
    print(
        "  camera_in_base_xyz_cm "
        f"mean=({camera_base_mean[0]:.1f}, {camera_base_mean[1]:.1f}, "
        f"{camera_base_mean[2]:.1f}), "
        f"std=({camera_base_std[0]:.2f}, {camera_base_std[1]:.2f}, "
        f"{camera_base_std[2]:.2f})"
    )
    if gripper_base_values:
        gripper_base = np.vstack(gripper_base_values)
        gripper_base_mean = meters_to_cm(gripper_base.mean(axis=0))
        gripper_base_std = meters_to_cm(gripper_base.std(axis=0))
        print(
            "  grippertip_base_xyz_cm "
            f"mean=({gripper_base_mean[0]:.1f}, {gripper_base_mean[1]:.1f}, "
            f"{gripper_base_mean[2]:.1f}), "
            f"std=({gripper_base_std[0]:.2f}, {gripper_base_std[1]:.2f}, "
            f"{gripper_base_std[2]:.2f})"
        )
    return 0


def run_image(
    args,
    detector,
    marker_length,
    marker_id,
    camera_matrix,
    dist_coeffs,
    tf_node,
    tf_buffer,
    base_frame,
    camera_frame,
    gripper_frame,
    tf_timeout,
    camera_base_z_offset,
    debug_pub,
    marker_base_pub,
):
    frame = cv2.imread(str(resolve_path(args.image)))
    if frame is None:
        print(f"Failed to read image: {args.image}")
        return 1

    try:
        t_base_camera = lookup_transform_matrix(
            tf_node,
            tf_buffer,
            base_frame,
            camera_frame,
            tf_timeout,
        )
    except RuntimeError as exc:
        print(exc)
        return 1
    try:
        t_base_gripper = lookup_transform_matrix(
            tf_node,
            tf_buffer,
            base_frame,
            gripper_frame,
            tf_timeout,
        )
        gripper_base_xyz = t_base_gripper[:3, 3]
    except RuntimeError as exc:
        print(exc)
        gripper_base_xyz = None

    result = check_frame(
        frame,
        detector,
        marker_length,
        marker_id,
        camera_matrix,
        dist_coeffs,
        tf_node,
        tf_buffer,
        t_base_camera,
        gripper_base_xyz,
        base_frame,
        camera_frame,
        tf_timeout,
        camera_base_z_offset,
    )
    if result:
        print(result[4])
        publish_point(tf_node, marker_base_pub, result[0], base_frame)

    publish_debug_image(tf_node, debug_pub, frame, camera_frame)

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        print(f"Saved annotated image to {output_path}")

    if not args.no_display:
        show_window("aruco extrinsics check", frame, 0)
        destroy_windows()
    return summarize([] if result is None else [result])


def run_camera(
    args,
    camera_source,
    image_width,
    image_height,
    detector,
    marker_length,
    marker_id,
    camera_matrix,
    dist_coeffs,
    tf_node,
    tf_buffer,
    base_frame,
    camera_frame,
    gripper_frame,
    tf_timeout,
    camera_base_z_offset,
    debug_pub,
    marker_base_pub,
):
    cap = open_camera(
        camera_source,
        args.width or image_width,
        args.height or image_height,
        args.fps,
        args.fourcc,
    )
    if not cap.isOpened():
        print(f"Failed to open camera source {camera_source}")
        return 1

    print(f"Camera opened: source={camera_source}")
    display_enabled = not args.no_display
    if display_enabled:
        print("OpenCV debug window enabled. Press q or ESC to quit.")
    results = []
    last_print = 0.0
    last_missing_marker_print = 0.0
    last_tf_warn = 0.0
    last_frame = None
    frame_count = 0
    interrupted = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame.")
                return 1

            waiting_for_tf = False
            try:
                t_base_camera = lookup_transform_matrix(
                    tf_node,
                    tf_buffer,
                    base_frame,
                    camera_frame,
                    tf_timeout,
                )
                t_base_gripper = lookup_transform_matrix(
                    tf_node,
                    tf_buffer,
                    base_frame,
                    gripper_frame,
                    tf_timeout,
                )
                gripper_base_xyz = t_base_gripper[:3, 3]
            except RuntimeError as exc:
                waiting_for_tf = True
                now = time.monotonic()
                if now - last_tf_warn >= args.print_every:
                    print(exc)
                    last_tf_warn = now
                draw_text(
                    frame,
                    f"Waiting for TF {base_frame} <- {camera_frame}/{gripper_frame}",
                    30,
                    (0, 0, 255),
                )
                result = None
            else:
                result = check_frame(
                    frame,
                    detector,
                    marker_length,
                    marker_id,
                    camera_matrix,
                    dist_coeffs,
                    tf_node,
                    tf_buffer,
                    t_base_camera,
                    gripper_base_xyz,
                    base_frame,
                    camera_frame,
                    tf_timeout,
                    camera_base_z_offset,
                )

            last_frame = frame
            if result:
                results.append(result)
                publish_point(tf_node, marker_base_pub, result[0], base_frame)
                now = time.monotonic()
                if now - last_print >= args.print_every:
                    print(result[4])
                    last_print = now
            elif not waiting_for_tf:
                now = time.monotonic()
                if now - last_missing_marker_print >= args.print_every:
                    print(
                        "No usable target marker yet "
                        f"(target id={marker_id}, dictionary={args.dictionary or 'default'}). "
                        "Check the OpenCV window or /aruco_extrinsics/debug_image for details."
                    )
                    last_missing_marker_print = now

            publish_debug_image(tf_node, debug_pub, frame, camera_frame)

            if display_enabled:
                key, display_enabled = show_window("aruco extrinsics check", frame, 1)
                if key in (ord("q"), 27):
                    break
            else:
                time.sleep(0.01)

            frame_count += 1
            if args.frames is not None and frame_count >= args.frames:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted.")
    finally:
        cap.release()
        if args.save_last and last_frame is not None:
            save_last_path = resolve_path(args.save_last)
            save_last_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_last_path), last_frame)
        destroy_windows()

    summary_code = summarize(results)
    return 0 if interrupted else summary_code


def main() -> int:
    args, ros_args = build_arg_parser().parse_known_args()
    (
        config_path,
        camera_source,
        marker_length,
        dictionary,
        marker_id,
        base_frame,
        camera_frame,
        gripper_frame,
        tf_timeout,
        camera_base_z_offset,
    ) = resolve_runtime(args)
    calibration_path = resolve_path(args.calibration)
    image_width, image_height, camera_matrix, dist_coeffs = load_camera_info(calibration_path)
    detector = make_detector(dictionary)

    print(f"Loaded intrinsics: {calibration_path}")
    if config_path is None:
        print("Loaded defaults: ninibot.task_params.PRESS_BUTTON_PARAMS")
    else:
        print(f"Loaded config override: {config_path}")
    print(f"Dictionary: {dictionary}, marker_id={marker_id}, marker_length={marker_length:.3f} m")
    print(f"TF: {base_frame} <- {camera_frame}, {base_frame} <- {gripper_frame}")
    print("Camera frame should be optical: +X image right, +Y image down, +Z forward.")
    print(
        "Optional debug correction: "
        f"camera_base_z_offset={camera_base_z_offset * 100.0:+.1f} cm"
    )

    rclpy.init(args=ros_args if ros_args else None)
    tf_node = rclpy.create_node("check_aruco_extrinsics")
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, tf_node)
    _ = tf_listener
    debug_pub = None
    if args.debug_image_topic:
        debug_pub = tf_node.create_publisher(Image, args.debug_image_topic, 10)
        print(f"Publishing debug image: {args.debug_image_topic}")
    marker_base_pub = None
    if args.marker_base_topic:
        marker_base_pub = tf_node.create_publisher(PointStamped, args.marker_base_topic, 10)
        print(f"Publishing marker base point: {args.marker_base_topic}")

    try:
        if args.image:
            return run_image(
                args,
                detector,
                marker_length,
                marker_id,
                camera_matrix,
                dist_coeffs,
                tf_node,
                tf_buffer,
                base_frame,
                camera_frame,
                gripper_frame,
                tf_timeout,
                camera_base_z_offset,
                debug_pub,
                marker_base_pub,
            )

        return run_camera(
            args,
            camera_source,
            image_width,
            image_height,
            detector,
            marker_length,
            marker_id,
            camera_matrix,
            dist_coeffs,
            tf_node,
            tf_buffer,
            base_frame,
            camera_frame,
            gripper_frame,
            tf_timeout,
            camera_base_z_offset,
            debug_pub,
            marker_base_pub,
        )
    finally:
        tf_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

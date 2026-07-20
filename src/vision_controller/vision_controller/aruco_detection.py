# flake8: noqa
"""Shared camera and ArUco helpers for calibration tools and ROS nodes."""

from __future__ import annotations

import ast
from pathlib import Path
import re

import cv2
import numpy as np


ARUCO_DICTIONARIES = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
}


def resolve_path(path_text: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists() or base_dir is None:
        return cwd_candidate
    return (base_dir / path).resolve()


def parse_yaml_scalar(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing '{key}' in calibration YAML")
    return match.group(1).strip()


def parse_yaml_data(text: str, section: str) -> list[float]:
    pattern = rf"^{re.escape(section)}:\n(?:[ \t]+.*\n)*?[ \t]+data:\s*(\[[^\]]*\])"
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing '{section}.data' in calibration YAML")

    values = ast.literal_eval(match.group(1))
    if not isinstance(values, list):
        raise ValueError(f"'{section}.data' must be a list")
    return [float(value) for value in values]


def load_camera_info(calibration_path: str | Path) -> tuple[int, int, np.ndarray, np.ndarray]:
    text = Path(calibration_path).expanduser().read_text(encoding="utf-8")
    width = int(parse_yaml_scalar(text, "image_width"))
    height = int(parse_yaml_scalar(text, "image_height"))
    camera_matrix = np.array(
        parse_yaml_data(text, "camera_matrix"),
        dtype=np.float64,
    ).reshape(3, 3)
    dist_coeffs = np.array(
        parse_yaml_data(text, "distortion_coefficients"),
        dtype=np.float64,
    ).reshape(-1)
    return width, height, camera_matrix, dist_coeffs


def parse_config_value(value_text: str):
    value = value_text.strip()
    if value.startswith(("'", '"')):
        return ast.literal_eval(value)
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_flat_ros_params(config_path: str | Path) -> dict[str, object]:
    path = Path(config_path).expanduser()
    params: dict[str, object] = {}
    if not path.exists():
        return params

    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s{4,}([A-Za-z_][A-Za-z0-9_]*):\s*(.+?)\s*$", line)
        if match:
            params[match.group(1)] = parse_config_value(match.group(2))
    return params


def make_aruco_dictionary(dictionary: str | int):
    dictionary_id = ARUCO_DICTIONARIES[dictionary] if isinstance(dictionary, str) else dictionary
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def make_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


def make_detector(dictionary: str | int):
    aruco_dictionary = make_aruco_dictionary(dictionary)
    parameters = make_detector_parameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dictionary, parameters)
        return lambda gray: detector.detectMarkers(gray)
    return lambda gray: cv2.aruco.detectMarkers(
        gray,
        aruco_dictionary,
        parameters=parameters,
    )


def marker_object_points(marker_length: float) -> np.ndarray:
    half = marker_length / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def estimate_marker_poses(
    corners: tuple[np.ndarray, ...] | list[np.ndarray],
    marker_length: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            marker_length,
            camera_matrix,
            dist_coeffs,
        )
        return rvecs.reshape(-1, 3), tvecs.reshape(-1, 3)

    objp = marker_object_points(marker_length)
    rvecs = []
    tvecs = []
    for marker_corners in corners:
        ok, rvec, tvec = cv2.solvePnP(
            objp,
            marker_corners.reshape(4, 2).astype(np.float32),
            camera_matrix,
            dist_coeffs,
        )
        if ok:
            rvecs.append(rvec.reshape(3))
            tvecs.append(tvec.reshape(3))

    return np.array(rvecs, dtype=np.float64), np.array(tvecs, dtype=np.float64)


def reprojection_errors(
    object_points: np.ndarray,
    image_corners: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[float, float, float]:
    projected, _ = cv2.projectPoints(
        object_points,
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        camera_matrix,
        dist_coeffs,
    )
    projected = projected.reshape(4, 2)
    detected = image_corners.reshape(4, 2).astype(np.float64)
    errors = np.linalg.norm(projected - detected, axis=1)
    mean = float(errors.mean())
    rms = float(np.sqrt(np.mean(errors * errors)))
    max_error = float(errors.max())
    return mean, rms, max_error


def draw_text(frame: np.ndarray, text: str, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def show_window(window_name: str, frame: np.ndarray, wait_ms: int) -> tuple[int, bool]:
    try:
        cv2.imshow(window_name, frame)
        return cv2.waitKey(wait_ms) & 0xFF, True
    except cv2.error as exc:
        print("OpenCV window display is unavailable; continuing without a window.")
        print(f"HighGUI error: {exc}")
        return -1, False


def destroy_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def normalize_camera_source(camera_source: int | str) -> int | str:
    if isinstance(camera_source, int):
        return camera_source

    source_text = str(camera_source).strip()
    if source_text.lstrip("-").isdigit():
        return int(source_text)
    return str(Path(source_text).expanduser())


def open_camera(
    camera_id: int | str,
    width: int,
    height: int,
    fps: int,
    fourcc_text: str = "",
) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(normalize_camera_source(camera_id), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if fourcc_text:
        fourcc = fourcc_text.upper()
        if len(fourcc) != 4:
            raise ValueError("--fourcc must be exactly 4 characters, for example YUYV or MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    return cap

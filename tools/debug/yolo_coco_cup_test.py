#!/usr/bin/env python3
"""OpenCV + YOLO11 COCO cup smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
VISION_SRC = WORKSPACE_ROOT / "src" / "vision_controller"
if VISION_SRC.exists():
    sys.path.insert(0, str(VISION_SRC))
NINIBOT_SRC = WORKSPACE_ROOT / "src" / "ninibot"
if NINIBOT_SRC.exists():
    sys.path.insert(0, str(NINIBOT_SRC))

from vision_controller.yolo_detector import YoloCupDetector
from vision_controller.aruco_detection import normalize_camera_source
from ninibot.task_params import INNOMAKER_CAMERA_DEVICE


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Test COCO cup detection with YOLO.")
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--camera-device", default=None)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--class-name", default="cup")
    parser.add_argument("--class-id", type=int, default=-1)
    return parser


def draw_detection(frame, detection):
    if detection is None:
        cv2.putText(frame, "cup: not detected", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return

    x, y, w, h = detection["bbox"]
    score = detection.get("score", 0.0)
    center = (int(detection["center_x"]), int(detection["center_y"]))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, center, 5, (0, 255, 255), -1)
    cv2.putText(
        frame,
        f"cup {score:.2f} bbox=({x},{y},{w},{h})",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


def main():
    args = build_arg_parser().parse_args()
    camera_source = normalize_camera_source(
        args.camera_device
        if args.camera_device
        else args.camera_id
        if args.camera_id is not None
        else INNOMAKER_CAMERA_DEVICE
    )
    detector = YoloCupDetector(
        model_path=args.model,
        conf=args.conf,
        class_name=args.class_name,
        class_id=args.class_id,
        imgsz=args.imgsz,
    )

    cap = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera source {camera_source}")

    print(
        "YOLO COCO cup test started; "
        f"model={args.model}, camera={camera_source}, conf={args.conf:.2f}. "
        "Press q to quit."
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Cannot read camera frame")
                break

            detection = detector.detect(frame)
            draw_detection(frame, detection)
            cv2.imshow("YOLO COCO Cup Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

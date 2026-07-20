# flake8: noqa
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

from vision_controller.hsv_detector import HsvCupDetector
from ninibot.task_params import CYLINDRICAL_VISION_PARAMS


DEFAULT_CAMERA_INDEX = 0

# Image center tolerance in pixels
CENTER_TOLERANCE_PX = 40


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HSV cup detection smoke test.")
    parser.add_argument("--camera-id", type=int, default=DEFAULT_CAMERA_INDEX)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_id)
    detector = HsvCupDetector(
        lower_h=CYLINDRICAL_VISION_PARAMS["lower_h"],
        lower_s=CYLINDRICAL_VISION_PARAMS["lower_s"],
        lower_v=CYLINDRICAL_VISION_PARAMS["lower_v"],
        upper_h=CYLINDRICAL_VISION_PARAMS["upper_h"],
        upper_s=CYLINDRICAL_VISION_PARAMS["upper_s"],
        upper_v=CYLINDRICAL_VISION_PARAMS["upper_v"],
        min_area=CYLINDRICAL_VISION_PARAMS["min_area"],
    )

    if not cap.isOpened():
        print(f"Cannot open camera index {args.camera_id}")
        return

    print("Camera opened.")
    print("Show a blue cup/object to the camera.")
    print(
        f"HSV lower={detector.lower_color.tolist()}, "
        f"upper={detector.upper_color.tolist()}, "
        f"min_area={detector.min_area:.1f}"
    )
    print("Press q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame.")
            break

        height, width = frame.shape[:2]
        image_center_x = width // 2
        image_center_y = height // 2

        detection, mask = detector.detect_with_mask(frame)
        status = "LOST"

        if detection is not None:
            x, y, w, h = detection["bbox"]
            area = detection["area"]

            cup_center_x = int(detection["center_x"])
            cup_center_y = int(detection["center_y"])

            error_x = cup_center_x - image_center_x
            error_y = cup_center_y - image_center_y

            if abs(error_x) < CENTER_TOLERANCE_PX and abs(error_y) < CENTER_TOLERANCE_PX:
                status = "CENTERED"
            else:
                status = "FOUND"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cup_center_x, cup_center_y), 6, (0, 255, 0), -1)

            text1 = f"status={status}"
            text2 = f"error_x={error_x}, error_y={error_y}, bbox_w={w}, bbox_h={h}, area={int(area)}"

            cv2.putText(frame, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, text2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            print(text1, "|", text2)

        # Draw image center
        cv2.circle(frame, (image_center_x, image_center_y), 6, (0, 0, 255), -1)
        cv2.line(frame, (image_center_x - 20, image_center_y), (image_center_x + 20, image_center_y), (0, 0, 255), 2)
        cv2.line(frame, (image_center_x, image_center_y - 20), (image_center_x, image_center_y + 20), (0, 0, 255), 2)

        if status == "LOST":
            cv2.putText(frame, "status=LOST", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Cup Color Detection Test", frame)
        cv2.imshow("Color Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

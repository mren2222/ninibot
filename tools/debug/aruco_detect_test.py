# flake8: noqa
import cv2
from cv2 import aruco
import numpy as np

DEFAULT_CAMERA_INDEX = 0
ARUCO_DICT = aruco.DICT_4X4_50

# Table marker IDs
TOP_LEFT_ID = 0
TOP_RIGHT_ID = 1
BOTTOM_LEFT_ID = 2
BOTTOM_RIGHT_ID = 3

BASE_ID = 4
CUP_ID = 5

TABLE_IDS = [TOP_LEFT_ID, TOP_RIGHT_ID, BOTTOM_LEFT_ID, BOTTOM_RIGHT_ID]

# Distance between marker centers, in cm
TABLE_WIDTH_CM = 60.0
TABLE_HEIGHT_CM = 30.0


def get_marker_center(marker_corners):
    pts = marker_corners[0]
    center_x = float(pts[:, 0].mean())
    center_y = float(pts[:, 1].mean())
    return center_x, center_y


def draw_text(frame, text, pos, color=(0, 255, 0)):
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def pixel_to_table_cm(center, H):
    pixel = np.array([[[center[0], center[1]]]], dtype=np.float32)
    real = cv2.perspectiveTransform(pixel, H)
    return real[0][0][0], real[0][0][1]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Table/base/cup ArUco homography test.")
    parser.add_argument("--camera-id", type=int, default=DEFAULT_CAMERA_INDEX)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():
        print(f"Cannot open camera index {args.camera_id}")
        return

    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    params = aruco.DetectorParameters()

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(aruco_dict, params)
        detect_markers = detector.detectMarkers
    else:
        detect_markers = lambda frame: aruco.detectMarkers(
            frame,
            aruco_dict,
            parameters=params,
        )

    base_calibrated = False
    base_pos_cm = None

    print("Camera opened.")
    print("Table markers: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right")
    print("Base marker: 4")
    print("Cup marker: 5")
    print("Press c = calibrate base")
    print("Press r = reset base calibration")
    print("Press q = quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame.")
            break

        corners, ids, rejected = detect_markers(frame)
        centers = {}
        table_H = None

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            for marker_corners, marker_id in zip(corners, ids.flatten()):
                marker_id = int(marker_id)
                center = get_marker_center(marker_corners)
                centers[marker_id] = center

                cx, cy = int(center[0]), int(center[1])

                if marker_id in TABLE_IDS:
                    label = f"TABLE {marker_id}"
                    color = (0, 255, 0)
                elif marker_id == BASE_ID:
                    label = "BASE 4"
                    color = (255, 0, 0)
                elif marker_id == CUP_ID:
                    label = "CUP 5"
                    color = (0, 255, 255)
                else:
                    label = f"ID {marker_id}"
                    color = (200, 200, 200)

                cv2.circle(frame, (cx, cy), 5, color, -1)
                draw_text(frame, label, (cx + 10, cy), color)

            # Build table homography if all 4 table markers are visible
            if all(marker_id in centers for marker_id in TABLE_IDS):
                pixel_points = np.array([
                    centers[TOP_LEFT_ID],
                    centers[TOP_RIGHT_ID],
                    centers[BOTTOM_LEFT_ID],
                    centers[BOTTOM_RIGHT_ID],
                ], dtype=np.float32)

                real_points = np.array([
                    [0.0, 0.0],
                    [TABLE_WIDTH_CM, 0.0],
                    [0.0, TABLE_HEIGHT_CM],
                    [TABLE_WIDTH_CM, TABLE_HEIGHT_CM],
                ], dtype=np.float32)

                table_H = cv2.getPerspectiveTransform(pixel_points, real_points)

                table_poly = np.array(pixel_points, dtype=np.int32)
                cv2.polylines(
                    frame,
                    [table_poly],
                    isClosed=True,
                    color=(0, 255, 255),
                    thickness=2,
                )

                # Show base current position if visible
                if BASE_ID in centers:
                    base_x_cm, base_y_cm = pixel_to_table_cm(centers[BASE_ID], table_H)
                    draw_text(
                        frame,
                        f"BASE current: x={base_x_cm:.1f} cm, y={base_y_cm:.1f} cm",
                        (20, 40),
                        (255, 0, 0),
                    )

                # Show calibrated base position
                if base_calibrated:
                    draw_text(
                        frame,
                        f"BASE calibrated: x={base_pos_cm[0]:.1f} cm, y={base_pos_cm[1]:.1f} cm",
                        (20, 70),
                        (255, 0, 0),
                    )
                else:
                    draw_text(
                        frame,
                        "BASE not calibrated. Show ID 4 and press c.",
                        (20, 70),
                        (0, 0, 255),
                    )

                # Cup tracking
                if CUP_ID in centers:
                    cup_x_cm, cup_y_cm = pixel_to_table_cm(centers[CUP_ID], table_H)

                    draw_text(
                        frame,
                        f"CUP: x={cup_x_cm:.1f} cm, y={cup_y_cm:.1f} cm",
                        (20, 100),
                        (0, 255, 255),
                    )

                    if base_calibrated:
                        dx_cm = cup_x_cm - base_pos_cm[0]
                        dy_cm = cup_y_cm - base_pos_cm[1]

                        # Temporary robot coordinate assumption
                        robot_x_m = dy_cm / 100.0
                        robot_y_m = -dx_cm / 100.0
                        robot_z_m = 0.10

                        draw_text(
                            frame,
                            f"CUP rel BASE: dx={dx_cm:.1f} cm, dy={dy_cm:.1f} cm",
                            (20, 130),
                            (0, 255, 255),
                        )

                        draw_text(
                            frame,
                            f"target_position: x={robot_x_m:.3f}, y={robot_y_m:.3f}, z={robot_z_m:.3f}",
                            (20, 160),
                            (255, 255, 0),
                        )

                        print(
                            f"target_position: "
                            f"x={robot_x_m:.3f}, y={robot_y_m:.3f}, z={robot_z_m:.3f}"
                        )

            else:
                missing = [marker_id for marker_id in TABLE_IDS if marker_id not in centers]
                draw_text(
                    frame,
                    f"Missing table markers: {missing}",
                    (20, 40),
                    (0, 0, 255),
                )

        cv2.imshow("ArUco Table Calibration Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            base_calibrated = False
            base_pos_cm = None
            print("Base calibration reset.")

        elif key == ord("c"):
            if table_H is None:
                print("Cannot calibrate base: table markers not fully visible.")
            elif BASE_ID not in centers:
                print("Cannot calibrate base: base marker ID 4 not visible.")
            else:
                base_x_cm, base_y_cm = pixel_to_table_cm(centers[BASE_ID], table_H)
                base_pos_cm = (base_x_cm, base_y_cm)
                base_calibrated = True
                print(f"Base calibrated: x={base_x_cm:.2f} cm, y={base_y_cm:.2f} cm")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

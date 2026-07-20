# flake8: noqa
"""HSV color cup detector wrapper used by grasp_cup_node and debug tools."""

from __future__ import annotations

import cv2
import numpy as np


class HsvCupDetector:
    def __init__(
        self,
        lower_h: int = 90,
        lower_s: int = 50,
        lower_v: int = 30,
        upper_h: int = 130,
        upper_s: int = 255,
        upper_v: int = 255,
        min_area: float = 500.0,
    ):
        self.lower_color = np.array(
            [int(lower_h), int(lower_s), int(lower_v)],
            dtype=np.uint8,
        )
        self.upper_color = np.array(
            [int(upper_h), int(upper_s), int(upper_v)],
            dtype=np.uint8,
        )
        self.min_area = float(min_area)
        self.kernel = np.ones((5, 5), np.uint8)

    def clean_mask(self, mask):
        mask = cv2.erode(mask, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        return mask

    def detect_candidates_with_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        mask = self.clean_mask(mask)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return [], mask

        detections = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            detections.append(
                {
                    "bbox": (x, y, w, h),
                    "area": area,
                    "center_x": x + 0.5 * w,
                    "center_y": y + 0.5 * h,
                }
            )

        detections.sort(
            key=lambda detection: detection.get("area", 0.0),
            reverse=True,
        )
        return detections, mask

    def detect_with_mask(self, frame):
        detections, mask = self.detect_candidates_with_mask(frame)

        if not detections:
            return None, mask

        return detections[0], mask

    def detect(self, frame):
        detection, _ = self.detect_with_mask(frame)
        return detection

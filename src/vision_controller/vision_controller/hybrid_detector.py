# flake8: noqa
"""Hybrid YOLO + HSV cup detector.

The intended control split is:
- YOLO acts as the semantic gate and reacquire signal.
- HSV gives the precise centroid, preferably inside a YOLO or tracked ROI.

This module is intentionally standalone. It is not wired into ninibot yet.
"""

from __future__ import annotations

import math

import numpy as np

from vision_controller.hsv_detector import HsvCupDetector
from vision_controller.yolo_detector import YoloCupDetector


class HybridCupDetector:
    SEARCH_MODE = "search"
    TRACK_MODE = "track"
    FINE_ALIGN_MODE = "fine_align"

    def __init__(
        self,
        hsv_detector=None,
        yolo_detector=None,
        hsv_kwargs=None,
        yolo_kwargs=None,
        roi_scale: float = 1.25,
        min_roi_padding_px: int = 12,
        use_yolo_when_hsv_missing: bool = True,
        allow_full_frame_hsv_fallback: bool = True,
        search_full_frame_hsv_fallback: bool = False,
        full_frame_hsv_after_yolo_missed_frames: int = 2,
        last_roi_frames: int = 5,
        yolo_center_distance_weight: float = 0.8,
        yolo_area_change_weight: float = 0.15,
        hsv_yolo_min_overlap_ratio: float = 0.05,
        hsv_yolo_max_center_distance_ratio: float = 0.85,
        hsv_candidate_area_weight: float = 1.0,
        hsv_candidate_center_distance_weight: float = 1.25,
    ):
        self.hsv_detector = hsv_detector or HsvCupDetector(**(hsv_kwargs or {}))
        self.yolo_detector = yolo_detector or YoloCupDetector(**(yolo_kwargs or {}))

        self.roi_scale = float(roi_scale)
        self.min_roi_padding_px = int(min_roi_padding_px)
        self.use_yolo_when_hsv_missing = bool(use_yolo_when_hsv_missing)
        self.allow_full_frame_hsv_fallback = bool(allow_full_frame_hsv_fallback)
        self.search_full_frame_hsv_fallback = bool(search_full_frame_hsv_fallback)
        self.full_frame_hsv_after_yolo_missed_frames = max(
            0,
            int(full_frame_hsv_after_yolo_missed_frames),
        )
        self.last_roi_frames = max(0, int(last_roi_frames))
        self.yolo_center_distance_weight = float(yolo_center_distance_weight)
        self.yolo_area_change_weight = float(yolo_area_change_weight)
        self.hsv_yolo_min_overlap_ratio = max(
            0.0,
            float(hsv_yolo_min_overlap_ratio),
        )
        self.hsv_yolo_max_center_distance_ratio = max(
            0.0,
            float(hsv_yolo_max_center_distance_ratio),
        )
        self.hsv_candidate_area_weight = float(hsv_candidate_area_weight)
        self.hsv_candidate_center_distance_weight = float(
            hsv_candidate_center_distance_weight
        )

        self.last_detection = None
        self.last_yolo_detection = None
        self.yolo_missed_frames = 0
        self.hsv_missed_frames = 0
        self.last_error = None

    def reset(self):
        self.last_detection = None
        self.last_yolo_detection = None
        self.yolo_missed_frames = 0
        self.hsv_missed_frames = 0
        self.last_error = None

    def normalize_mode(self, mode):
        if mode is None:
            return self.TRACK_MODE

        text = str(mode).strip().lower()
        if text in ("search", "search_prep", "searching", "lost", "starting"):
            return self.SEARCH_MODE
        if text in ("fine_align", "fine_align_prep", "finealign"):
            return self.FINE_ALIGN_MODE

        return self.TRACK_MODE

    def detect(self, frame, mode=None):
        detection, _ = self.detect_with_mask(frame, mode=mode)
        return detection

    def detect_with_mask(self, frame, mode=None):
        mode = self.normalize_mode(mode)
        yolo_detection = self.detect_yolo(frame)

        if yolo_detection is not None:
            self.last_yolo_detection = dict(yolo_detection)
            self.yolo_missed_frames = 0
            detection, mask = self.detect_hsv_gated_by_yolo(
                frame,
                mode,
                yolo_detection,
            )
            if detection is not None:
                return detection, mask

            if self.use_yolo_when_hsv_missing:
                return self.accept_detection(
                    yolo_detection,
                    source="yolo",
                    mode=mode,
                    yolo_detection=yolo_detection,
                    hsv_detection=None,
                    roi=self.expanded_roi(frame.shape, yolo_detection["bbox"]),
                    mask=None,
                )
        else:
            self.yolo_missed_frames += 1

        if mode in (self.TRACK_MODE, self.FINE_ALIGN_MODE):
            detection, mask = self.detect_hsv_in_last_roi(frame, mode)
            if detection is not None:
                return detection, mask

        if self.should_try_full_frame_hsv(mode):
            detection, mask = self.detect_hsv_with_yolo_reference(
                frame,
                self.last_yolo_detection,
            )
            if detection is not None:
                return self.accept_detection(
                    detection,
                    source="hsv",
                    mode=mode,
                    yolo_detection=yolo_detection,
                    hsv_detection=detection,
                    roi=None,
                    mask=mask,
                )

            self.hsv_missed_frames += 1
            return None, mask

        self.hsv_missed_frames += 1
        return None, None

    def detect_yolo(self, frame):
        try:
            if hasattr(self.yolo_detector, "detect_all"):
                detections = self.yolo_detector.detect_all(frame)
                return self.select_yolo_detection(detections, frame.shape)

            return self.yolo_detector.detect(frame)
        except Exception as exc:
            self.last_error = str(exc)
            return None

    def select_yolo_detection(self, detections, frame_shape):
        if not detections:
            return None

        if self.last_detection is None:
            return dict(detections[0])

        frame_h, frame_w = frame_shape[:2]
        max_dim = max(1.0, float(max(frame_w, frame_h)))
        last_center_x = float(self.last_detection["center_x"])
        last_center_y = float(self.last_detection["center_y"])
        last_area_detection = self.last_yolo_detection or self.last_detection
        last_area = max(1.0, float(last_area_detection.get("area", 1.0)))

        best = None
        best_score = None
        for detection in detections:
            center_dx = float(detection["center_x"]) - last_center_x
            center_dy = float(detection["center_y"]) - last_center_y
            center_distance = math.hypot(center_dx, center_dy) / max_dim
            area = max(1.0, float(detection.get("area", 1.0)))
            area_change = abs(math.log(area / last_area))
            track_score = (
                float(detection.get("score", 0.0))
                - self.yolo_center_distance_weight * center_distance
                - self.yolo_area_change_weight * area_change
            )

            if best_score is None or track_score > best_score:
                best_score = track_score
                best = dict(detection)
                best["track_score"] = track_score

        return best

    def detect_hsv_gated_by_yolo(self, frame, mode, yolo_detection):
        roi = self.expanded_roi(frame.shape, yolo_detection["bbox"])
        hsv_detection, mask = self.detect_hsv_in_roi(
            frame,
            roi,
            yolo_detection=yolo_detection,
        )
        if hsv_detection is None:
            self.hsv_missed_frames += 1
            return None, mask

        return self.accept_detection(
            hsv_detection,
            source="yolo_hsv",
            mode=mode,
            yolo_detection=yolo_detection,
            hsv_detection=hsv_detection,
            roi=roi,
            mask=mask,
        )

    def detect_hsv_in_last_roi(self, frame, mode):
        if self.last_detection is None:
            return None, None
        if self.yolo_missed_frames > self.last_roi_frames:
            return None, None

        roi = self.expanded_roi(frame.shape, self.last_detection["bbox"])
        reference_yolo = (
            self.last_detection.get("yolo_detection")
            or self.last_yolo_detection
        )
        hsv_detection, mask = self.detect_hsv_in_roi(
            frame,
            roi,
            yolo_detection=reference_yolo,
        )
        if hsv_detection is None:
            self.hsv_missed_frames += 1
            return None, mask

        return self.accept_detection(
            hsv_detection,
            source="last_roi_hsv",
            mode=mode,
            yolo_detection=None,
            hsv_detection=hsv_detection,
            roi=roi,
            mask=mask,
        )

    def detect_hsv_with_yolo_reference(self, frame, yolo_detection=None):
        candidates, mask = self.hsv_detector.detect_candidates_with_mask(frame)
        detection = self.select_hsv_candidate(candidates, yolo_detection)
        return detection, mask

    def detect_hsv_in_roi(self, frame, roi, yolo_detection=None):
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            return None, None

        crop = frame[y : y + h, x : x + w]
        candidates, crop_mask = self.hsv_detector.detect_candidates_with_mask(crop)
        mask = self.full_frame_mask(frame.shape, crop_mask, roi)
        if not candidates:
            return None, mask

        global_candidates = []
        for candidate in candidates:
            candidate = dict(candidate)
            bx, by, bw, bh = candidate["bbox"]
            candidate["bbox"] = (bx + x, by + y, bw, bh)
            candidate["center_x"] = float(candidate["center_x"]) + x
            candidate["center_y"] = float(candidate["center_y"]) + y
            global_candidates.append(candidate)

        detection = self.select_hsv_candidate(
            global_candidates,
            yolo_detection,
        )
        if detection is None:
            return None, mask

        return detection, mask

    def select_hsv_candidate(self, candidates, yolo_detection=None):
        if not candidates:
            return None

        if yolo_detection is None:
            return dict(candidates[0])

        gated = []
        for candidate in candidates:
            metrics = self.hsv_yolo_metrics(candidate, yolo_detection)
            if not self.hsv_candidate_matches_yolo(metrics):
                continue
            scored = dict(candidate)
            scored.update(metrics)
            gated.append(scored)

        if not gated:
            return None

        max_area = max(1.0, max(float(item.get("area", 0.0)) for item in gated))
        for candidate in gated:
            area_norm = float(candidate.get("area", 0.0)) / max_area
            distance_norm = float(candidate["yolo_center_distance_ratio"])
            candidate["hsv_candidate_score"] = (
                self.hsv_candidate_area_weight * area_norm
                - self.hsv_candidate_center_distance_weight * distance_norm
            )

        return max(
            gated,
            key=lambda candidate: candidate["hsv_candidate_score"],
        )

    def hsv_candidate_matches_yolo(self, metrics):
        if (
            metrics["yolo_center_distance_ratio"]
            > self.hsv_yolo_max_center_distance_ratio
        ):
            return False

        return (
            metrics["yolo_overlap_ratio"] >= self.hsv_yolo_min_overlap_ratio
            or metrics["hsv_center_inside_yolo"]
        )

    def hsv_yolo_metrics(self, candidate, yolo_detection):
        yolo_bbox = yolo_detection["bbox"]
        candidate_bbox = candidate["bbox"]
        yolo_x, yolo_y, yolo_w, yolo_h = yolo_bbox
        cand_x, cand_y, cand_w, cand_h = candidate_bbox

        yolo_center_x = float(yolo_x) + 0.5 * float(yolo_w)
        yolo_center_y = float(yolo_y) + 0.5 * float(yolo_h)
        center_distance = math.hypot(
            float(candidate["center_x"]) - yolo_center_x,
            float(candidate["center_y"]) - yolo_center_y,
        )
        yolo_diag = max(1.0, math.hypot(float(yolo_w), float(yolo_h)))

        inter_x1 = max(float(yolo_x), float(cand_x))
        inter_y1 = max(float(yolo_y), float(cand_y))
        inter_x2 = min(float(yolo_x + yolo_w), float(cand_x + cand_w))
        inter_y2 = min(float(yolo_y + yolo_h), float(cand_y + cand_h))
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection_area = inter_w * inter_h
        candidate_area = max(1.0, float(candidate.get("area", cand_w * cand_h)))

        return {
            "yolo_center_distance_px": center_distance,
            "yolo_center_distance_ratio": center_distance / yolo_diag,
            "yolo_overlap_ratio": intersection_area / candidate_area,
            "hsv_center_inside_yolo": (
                float(yolo_x) <= float(candidate["center_x"]) <= float(yolo_x + yolo_w)
                and float(yolo_y) <= float(candidate["center_y"]) <= float(yolo_y + yolo_h)
            ),
        }

    def should_try_full_frame_hsv(self, mode):
        if not self.allow_full_frame_hsv_fallback:
            return False
        if mode == self.SEARCH_MODE:
            return self.search_full_frame_hsv_fallback
        if mode == self.FINE_ALIGN_MODE:
            return True

        return self.yolo_missed_frames >= self.full_frame_hsv_after_yolo_missed_frames

    def accept_detection(
        self,
        detection,
        source,
        mode,
        yolo_detection,
        hsv_detection,
        roi,
        mask,
    ):
        enriched = dict(detection)
        enriched["source"] = source
        enriched["mode"] = mode
        enriched["roi"] = roi
        enriched["yolo_detection"] = self.copy_detection(yolo_detection)
        enriched["hsv_detection"] = self.copy_detection(hsv_detection)
        enriched["yolo_missed_frames"] = self.yolo_missed_frames
        enriched["hsv_missed_frames"] = self.hsv_missed_frames
        enriched["last_error"] = self.last_error

        if hsv_detection is not None:
            self.hsv_missed_frames = 0
            enriched["hsv_missed_frames"] = 0

        self.last_detection = dict(enriched)

        return enriched, mask

    def expanded_roi(self, frame_shape, bbox):
        frame_h, frame_w = frame_shape[:2]
        x, y, w, h = bbox
        center_x = float(x) + 0.5 * float(w)
        center_y = float(y) + 0.5 * float(h)

        scaled_w = max(float(w) * self.roi_scale, float(w) + 2.0 * self.min_roi_padding_px)
        scaled_h = max(float(h) * self.roi_scale, float(h) + 2.0 * self.min_roi_padding_px)

        x1 = max(0, int(round(center_x - 0.5 * scaled_w)))
        y1 = max(0, int(round(center_y - 0.5 * scaled_h)))
        x2 = min(frame_w, int(round(center_x + 0.5 * scaled_w)))
        y2 = min(frame_h, int(round(center_y + 0.5 * scaled_h)))

        return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    def full_frame_mask(self, frame_shape, crop_mask, roi):
        if crop_mask is None:
            return None

        mask = np.zeros(frame_shape[:2], dtype=crop_mask.dtype)
        x, y, w, h = roi
        mask[y : y + h, x : x + w] = crop_mask[:h, :w]

        return mask

    def copy_detection(self, detection):
        if detection is None:
            return None

        return dict(detection)

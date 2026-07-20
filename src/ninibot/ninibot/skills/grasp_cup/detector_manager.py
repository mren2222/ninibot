from vision_controller.hybrid_detector import HybridCupDetector
from vision_controller.hsv_detector import HsvCupDetector
from vision_controller.yolo_detector import YoloCupDetector


class CupDetectorManager:
    def __init__(self, params, logger, warn_throttled):
        self.params = params
        self.logger = logger
        self.warn_throttled = warn_throttled
        self.detector_backend = params.detector_backend
        self.hsv_detector = None
        self.hybrid_detector = None
        self.yolo_detector = None
        self.yolo_error_message = ""

        if self.detector_backend not in self.supported_backends():
            self.logger.warn(
                f"Unknown detector_backend={self.detector_backend!r}; "
                "falling back to hsv"
            )
            self.detector_backend = "hsv"

        self.setup()

    @staticmethod
    def supported_backends():
        return ("hsv", "yolo", "hybrid")

    def setup(self):
        if self.detector_backend == "hsv":
            self.hsv_detector = HsvCupDetector(
                lower_h=self.params.lower_h,
                lower_s=self.params.lower_s,
                lower_v=self.params.lower_v,
                upper_h=self.params.upper_h,
                upper_s=self.params.upper_s,
                upper_v=self.params.upper_v,
                min_area=self.params.min_area,
            )
        elif self.detector_backend == "yolo":
            try:
                self.yolo_detector = YoloCupDetector(
                    model_path=self.params.yolo_model_path,
                    conf=self.params.yolo_conf,
                    class_name=self.params.yolo_class_name,
                    class_id=self.params.yolo_class_id,
                    imgsz=self.params.yolo_imgsz,
                    device=self.params.yolo_device,
                )
            except Exception as exc:
                self.yolo_error_message = str(exc)
                self.logger.error(f"Cannot start YOLO detector: {exc}")
        elif self.detector_backend == "hybrid":
            try:
                self.hybrid_detector = HybridCupDetector(
                    hsv_kwargs={
                        "lower_h": self.params.lower_h,
                        "lower_s": self.params.lower_s,
                        "lower_v": self.params.lower_v,
                        "upper_h": self.params.upper_h,
                        "upper_s": self.params.upper_s,
                        "upper_v": self.params.upper_v,
                        "min_area": self.params.min_area,
                    },
                    yolo_kwargs={
                        "model_path": self.params.yolo_model_path,
                        "conf": self.params.yolo_conf,
                        "class_name": self.params.yolo_class_name,
                        "class_id": self.params.yolo_class_id,
                        "imgsz": self.params.yolo_imgsz,
                        "device": self.params.yolo_device,
                    },
                    roi_scale=self.params.hybrid_roi_scale,
                    min_roi_padding_px=self.params.hybrid_min_roi_padding_px,
                    hsv_yolo_min_overlap_ratio=(
                        self.params.hybrid_hsv_yolo_min_overlap_ratio
                    ),
                    hsv_yolo_max_center_distance_ratio=(
                        self.params.hybrid_hsv_yolo_max_center_distance_ratio
                    ),
                    hsv_candidate_area_weight=(
                        self.params.hybrid_hsv_candidate_area_weight
                    ),
                    hsv_candidate_center_distance_weight=(
                        self.params.hybrid_hsv_candidate_center_distance_weight
                    ),
                )
                self.hsv_detector = self.hybrid_detector.hsv_detector
                self.yolo_detector = self.hybrid_detector.yolo_detector
            except Exception as exc:
                self.yolo_error_message = str(exc)
                self.logger.error(f"Cannot start hybrid detector: {exc}")

    def ready_to_start(self):
        if self.detector_backend not in ("yolo", "hybrid"):
            return True, ""

        if self.yolo_detector is not None:
            return True, ""

        return (
            False,
            self.yolo_error_message
            or f"{self.detector_backend} detector is not ready",
        )

    def reset(self):
        if self.detector_backend == "hybrid" and self.hybrid_detector is not None:
            self.hybrid_detector.reset()

    def summary(self):
        if self.detector_backend == "hsv" and self.hsv_detector is not None:
            return (
                "detector_backend=hsv, "
                f"HSV lower={self.hsv_detector.lower_color.tolist()}, "
                f"upper={self.hsv_detector.upper_color.tolist()}, "
                f"min_area={self.hsv_detector.min_area:.1f}"
            )

        if self.detector_backend == "yolo" and self.yolo_detector is not None:
            return (
                "detector_backend=yolo, "
                f"model={self.yolo_detector.model_path}, "
                f"conf={self.yolo_detector.conf:.2f}, "
                f"class={self.yolo_detector.class_name}"
            )

        if self.detector_backend == "hybrid" and self.hybrid_detector is not None:
            return (
                "detector_backend=hybrid, "
                f"model={self.yolo_detector.model_path}, "
                f"conf={self.yolo_detector.conf:.2f}, "
                f"class={self.yolo_detector.class_name}, "
                f"hsv_min_area={self.hsv_detector.min_area:.1f}"
            )

        return f"detector_backend={self.detector_backend}, detector not ready"

    def detect_hsv(self, frame):
        if self.hsv_detector is None:
            self.warn_throttled("hsv_missing", "HSV detector is not ready", 2.0)
            return None, None

        return self.hsv_detector.detect_with_mask(frame)

    def detect_yolo(self, frame):
        if self.yolo_detector is None:
            self.warn_throttled("yolo_missing", "YOLO detector is not ready", 2.0)
            return None, None

        try:
            return self.yolo_detector.detect(frame), None
        except Exception as exc:
            self.warn_throttled(
                "yolo_detect",
                f"YOLO detection failed: {exc}",
                2.0,
            )
            return None, None

    def detect(self, frame, mode):
        if self.detector_backend == "hybrid":
            if self.hybrid_detector is None:
                self.warn_throttled(
                    "hybrid_missing",
                    "Hybrid detector is not ready",
                    2.0,
                )
                return None, None
            return self.hybrid_detector.detect_with_mask(frame, mode=mode)

        if self.detector_backend == "yolo":
            return self.detect_yolo(frame)

        return self.detect_hsv(frame)

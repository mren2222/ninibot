# flake8: noqa
"""YOLO cup detector wrapper used by grasp_cup_node and debug tools."""

from __future__ import annotations


class YoloCupDetector:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.35,
        class_name: str = "cup",
        class_id: int = -1,
        imgsz: int = 640,
        device: str = "",
    ):
        self.model_path = model_path
        self.conf = conf
        self.class_name = class_name.strip().lower()
        self.class_id = int(class_id)
        self.imgsz = int(imgsz)
        self.device = device.strip()
        self.model = self.load_model()
        self.resolved_class_id = self.resolve_class_id()

    def load_model(self):
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing Python package 'ultralytics'. Install it with: "
                "python3 -m pip install ultralytics"
            ) from exc

        return YOLO(self.model_path)

    def resolve_class_id(self):
        if self.class_id >= 0:
            return self.class_id

        names = getattr(self.model, "names", {}) or {}
        for index, name in names.items():
            if str(name).strip().lower() == self.class_name:
                return int(index)

        return None

    def predict(self, frame):
        kwargs = {
            "imgsz": self.imgsz,
            "conf": self.conf,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device

        return self.model.predict(frame, **kwargs)

    def detect_all(self, frame):
        results = self.predict(frame)
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []

        detections = []
        for box in boxes:
            class_id = int(box.cls[0])
            if self.resolved_class_id is not None and class_id != self.resolved_class_id:
                continue

            score = float(box.conf[0])
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            x = int(round(x1))
            y = int(round(y1))
            w = max(0, int(round(x2 - x1)))
            h = max(0, int(round(y2 - y1)))
            area = float(w * h)

            detection = {
                "bbox": (x, y, w, h),
                "area": area,
                "center_x": x1 + 0.5 * (x2 - x1),
                "center_y": y1 + 0.5 * (y2 - y1),
                "score": score,
                "class_id": class_id,
                "class_name": self.class_name,
            }

            detections.append(detection)

        return sorted(
            detections,
            key=lambda detection: detection.get("score", 0.0),
            reverse=True,
        )

    def detect(self, frame):
        detections = self.detect_all(frame)
        if not detections:
            return None

        return detections[0]

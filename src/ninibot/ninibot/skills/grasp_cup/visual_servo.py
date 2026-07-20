from dataclasses import dataclass


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class PanServoConfig:
    theta_kp: float
    theta_sign: float
    theta_max_step: float
    tracking_theta_max_step: float
    tracking_settle_tolerance_px: float
    target_bbox_width: float


class PanServo:
    def __init__(self, config):
        self.config = config

    def tracking_errors(self, detection, image_target_x):
        bbox_w = detection["bbox"][2]
        error_x = detection["center_x"] - image_target_x
        bbox_error = self.config.target_bbox_width - float(bbox_w)
        return error_x, bbox_error, bbox_w

    def pan_centered(self, error_x, tolerance=None):
        if tolerance is None:
            tolerance = self.config.tracking_settle_tolerance_px

        return abs(error_x) <= tolerance

    def theta_delta(self, error_x, max_step=None, tolerance=0.0):
        if abs(error_x) <= tolerance:
            return 0.0

        if max_step is None:
            max_step = self.config.theta_max_step

        theta_step = self.config.theta_kp * error_x
        theta_step = clamp(theta_step, -max_step, max_step)
        return self.config.theta_sign * theta_step

    def tracking_theta_delta(self, error_x):
        return self.theta_delta(
            error_x,
            max_step=self.config.tracking_theta_max_step,
            tolerance=self.config.tracking_settle_tolerance_px,
        )


class DetectionCenterSmoother:
    def __init__(self, alpha):
        self.alpha = alpha
        self.smoothed_center_x = None

    def reset(self):
        self.smoothed_center_x = None

    def smooth(self, detection):
        alpha = clamp(float(self.alpha), 0.0, 1.0)

        if self.smoothed_center_x is None:
            self.smoothed_center_x = detection["center_x"]
        else:
            self.smoothed_center_x = (
                alpha * detection["center_x"]
                + (1.0 - alpha) * self.smoothed_center_x
            )

        smoothed = dict(detection)
        smoothed["center_x"] = self.smoothed_center_x
        return smoothed

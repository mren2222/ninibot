from dataclasses import dataclass
from statistics import median

from audio_controller.audio_features import AudioFeatures


CALIBRATING = "calibrating"
SILENT = "silent"
FLOW_START = "flow_start"
FLOWING = "flowing"
FLOW_STOP = "flow_stop"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class WaterFlowDetectorConfig:
    baseline_sec: float = 2.0
    start_threshold_db: float = 8.0
    stop_threshold_db: float = 4.0
    min_start_sec: float = 0.30
    min_stop_sec: float = 0.70


@dataclass(frozen=True)
class WaterFlowReading:
    state: str
    rms_db: float
    band_db: float
    score_db: float
    baseline_db: float | None
    start_threshold_db: float | None
    stop_threshold_db: float | None
    calibrated: bool


class WaterFlowDetector:
    def __init__(self, config: WaterFlowDetectorConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._baseline_scores: list[float] = []
        self._baseline_elapsed = 0.0
        self._baseline_db: float | None = None
        self._state = CALIBRATING
        self._above_sec = 0.0
        self._below_sec = 0.0

    @property
    def calibrated(self) -> bool:
        return self._baseline_db is not None

    @property
    def baseline_db(self) -> float | None:
        return self._baseline_db

    def update(self, features: AudioFeatures, dt_sec: float) -> WaterFlowReading:
        dt = max(float(dt_sec), 0.0)

        if self._baseline_db is None:
            self._baseline_scores.append(features.score_db)
            self._baseline_elapsed += dt
            if self._baseline_elapsed >= max(self.config.baseline_sec, dt):
                self._baseline_db = float(median(self._baseline_scores))
                self._state = SILENT
            return self._reading(features, CALIBRATING if self._baseline_db is None else SILENT)

        start_line = self._baseline_db + self.config.start_threshold_db
        stop_line = self._baseline_db + self.config.stop_threshold_db
        published_state = self._state

        if self._state in (SILENT, FLOW_STOP, UNKNOWN):
            if features.score_db >= start_line:
                self._above_sec += dt
            else:
                self._above_sec = 0.0

            if self._above_sec >= self.config.min_start_sec:
                self._state = FLOWING
                published_state = FLOW_START
                self._above_sec = 0.0
                self._below_sec = 0.0
            else:
                published_state = SILENT

        elif self._state in (FLOWING, FLOW_START):
            if features.score_db <= stop_line:
                self._below_sec += dt
            else:
                self._below_sec = 0.0

            if self._below_sec >= self.config.min_stop_sec:
                self._state = SILENT
                published_state = FLOW_STOP
                self._above_sec = 0.0
                self._below_sec = 0.0
            else:
                published_state = FLOWING
        else:
            self._state = UNKNOWN
            published_state = UNKNOWN

        return self._reading(features, published_state)

    def _reading(self, features: AudioFeatures, state: str) -> WaterFlowReading:
        start_line = None
        stop_line = None
        if self._baseline_db is not None:
            start_line = self._baseline_db + self.config.start_threshold_db
            stop_line = self._baseline_db + self.config.stop_threshold_db
        return WaterFlowReading(
            state=state,
            rms_db=features.rms_db,
            band_db=features.band_db,
            score_db=features.score_db,
            baseline_db=self._baseline_db,
            start_threshold_db=start_line,
            stop_threshold_db=stop_line,
            calibrated=self._baseline_db is not None,
        )

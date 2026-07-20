import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

from audio_controller.audio_features import compute_audio_features
from audio_controller.audio_input import AudioInputError, SoundDeviceAudioInput
from audio_controller.water_flow_detector import (
    UNKNOWN,
    WaterFlowDetector,
    WaterFlowDetectorConfig,
)


class WaterFlowMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("water_flow_monitor_node")

        self.declare_parameter("device", "")
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("channels", 1)
        self.declare_parameter("frame_ms", 50.0)
        self.declare_parameter("baseline_sec", 2.0)
        self.declare_parameter("band_low_hz", 800.0)
        self.declare_parameter("band_high_hz", 6000.0)
        self.declare_parameter("band_weight", 0.75)
        self.declare_parameter("start_threshold_db", 8.0)
        self.declare_parameter("stop_threshold_db", 4.0)
        self.declare_parameter("min_start_sec", 0.30)
        self.declare_parameter("min_stop_sec", 0.70)
        self.declare_parameter("state_topic", "/water_flow/state")
        self.declare_parameter("debug_topic", "/water_flow/debug")
        self.declare_parameter("score_topic", "/water_flow/score")
        self.declare_parameter("command_topic", "/water_flow/command")

        self.sample_rate = int(self.get_parameter("sample_rate").value)
        self.frame_ms = float(self.get_parameter("frame_ms").value)
        self.frame_sec = self.frame_ms / 1000.0
        self.band_low_hz = float(self.get_parameter("band_low_hz").value)
        self.band_high_hz = float(self.get_parameter("band_high_hz").value)
        self.band_weight = float(self.get_parameter("band_weight").value)

        detector_config = WaterFlowDetectorConfig(
            baseline_sec=float(self.get_parameter("baseline_sec").value),
            start_threshold_db=float(self.get_parameter("start_threshold_db").value),
            stop_threshold_db=float(self.get_parameter("stop_threshold_db").value),
            min_start_sec=float(self.get_parameter("min_start_sec").value),
            min_stop_sec=float(self.get_parameter("min_stop_sec").value),
        )
        self.detector = WaterFlowDetector(detector_config)

        self.state_pub = self.create_publisher(
            String,
            str(self.get_parameter("state_topic").value),
            10,
        )
        self.debug_pub = self.create_publisher(
            String,
            str(self.get_parameter("debug_topic").value),
            10,
        )
        self.score_pub = self.create_publisher(
            Float32,
            str(self.get_parameter("score_topic").value),
            10,
        )
        self.command_sub = self.create_subscription(
            String,
            str(self.get_parameter("command_topic").value),
            self._on_command,
            10,
        )

        self.audio = None
        try:
            self.audio = SoundDeviceAudioInput(
                device=self.get_parameter("device").value,
                sample_rate=self.sample_rate,
                channels=int(self.get_parameter("channels").value),
                frame_ms=self.frame_ms,
            )
            self.audio.open()
            self.get_logger().info(
                "water flow monitor started: "
                f"sample_rate={self.sample_rate} frame_ms={self.frame_ms:.1f} "
                f"band=({self.band_low_hz:.0f}, {self.band_high_hz:.0f})Hz"
            )
        except AudioInputError as exc:
            self.get_logger().error(str(exc))
        except Exception as exc:  # pragma: no cover - depends on local audio device
            self.get_logger().error(f"failed to open microphone: {exc}")

        self.timer = self.create_timer(max(self.frame_sec, 0.01), self._tick)

    def destroy_node(self) -> bool:
        if self.audio is not None:
            self.audio.close()
        return super().destroy_node()

    def _on_command(self, msg: String) -> None:
        command = msg.data.strip().lower()
        if command in ("reset", "recalibrate", "calibrate"):
            self.detector.reset()
            self.get_logger().info("water flow detector baseline reset")
        else:
            self.get_logger().warning(f"unknown water flow command: {msg.data!r}")

    def _tick(self) -> None:
        if self.audio is None:
            self._publish_unknown("microphone_unavailable")
            return

        try:
            frame, overflowed = self.audio.read_frame()
        except Exception as exc:  # pragma: no cover - depends on local audio device
            self.get_logger().error(f"failed to read microphone frame: {exc}")
            self._publish_unknown("microphone_read_failed")
            return

        features = compute_audio_features(
            frame,
            self.sample_rate,
            self.band_low_hz,
            self.band_high_hz,
            self.band_weight,
        )
        reading = self.detector.update(features, self.frame_sec)
        self._publish_reading(reading.state, reading.__dict__, overflowed=overflowed)
        self.score_pub.publish(Float32(data=float(reading.score_db)))

    def _publish_unknown(self, reason: str) -> None:
        self._publish_reading(
            UNKNOWN,
            {
                "state": UNKNOWN,
                "reason": reason,
                "calibrated": self.detector.calibrated,
                "baseline_db": self.detector.baseline_db,
            },
            overflowed=False,
        )

    def _publish_reading(self, state: str, payload: dict, *, overflowed: bool) -> None:
        self.state_pub.publish(String(data=state))
        debug = dict(payload)
        debug["overflowed"] = overflowed
        self.debug_pub.publish(String(data=json.dumps(debug, sort_keys=True)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WaterFlowMonitorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

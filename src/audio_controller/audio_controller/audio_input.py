import contextlib

import numpy as np


class AudioInputError(RuntimeError):
    pass


def _load_sounddevice():
    try:
        import sounddevice as sd  # type: ignore
    except ImportError as exc:
        raise AudioInputError(
            "Python package 'sounddevice' is required for microphone input. "
            "Install it with 'sudo apt install python3-sounddevice' or "
            "'python3 -m pip install sounddevice'."
        ) from exc
    return sd


def parse_device(device):
    if device is None or device == "":
        return None
    text = str(device).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


class SoundDeviceAudioInput:
    def __init__(
        self,
        *,
        device=None,
        sample_rate: int = 16000,
        channels: int = 1,
        frame_ms: float = 50.0,
    ) -> None:
        self._sd = _load_sounddevice()
        self.device = parse_device(device)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.frame_samples = max(1, int(round(self.sample_rate * frame_ms / 1000.0)))
        self._stream = None

    def open(self) -> None:
        if self._stream is not None:
            return
        self._stream = self._sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            channels=self.channels,
            dtype="float32",
            device=self.device,
        )
        self._stream.start()

    def read_frame(self) -> tuple[np.ndarray, bool]:
        if self._stream is None:
            self.open()
        assert self._stream is not None
        frame, overflowed = self._stream.read(self.frame_samples)
        return np.asarray(frame, dtype=np.float32), bool(overflowed)

    def close(self) -> None:
        if self._stream is None:
            return
        with contextlib.suppress(Exception):
            self._stream.stop()
        with contextlib.suppress(Exception):
            self._stream.close()
        self._stream = None


def query_devices() -> str:
    sd = _load_sounddevice()
    return str(sd.query_devices())

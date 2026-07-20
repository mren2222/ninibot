from dataclasses import dataclass
import math

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class AudioFeatures:
    rms_db: float
    band_db: float
    score_db: float


def to_mono_float32(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    arr = arr.astype(np.float32, copy=False)
    return np.clip(arr, -1.0, 1.0)


def rms_db(samples: np.ndarray) -> float:
    mono = to_mono_float32(samples)
    rms = float(np.sqrt(np.mean(np.square(mono)) + EPS))
    return 20.0 * math.log10(max(rms, EPS))


def band_energy_db(
    samples: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> float:
    mono = to_mono_float32(samples)
    if mono.size == 0:
        return -120.0

    nyquist = sample_rate * 0.5
    low = max(0.0, min(float(low_hz), nyquist))
    high = max(low, min(float(high_hz), nyquist))
    if high <= low:
        return rms_db(mono)

    window = np.hanning(mono.size).astype(np.float32)
    spectrum = np.fft.rfft(mono * window)
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / float(sample_rate))
    band = (freqs >= low) & (freqs <= high)
    if not np.any(band):
        return rms_db(mono)

    power = np.mean(np.abs(spectrum[band]) ** 2) / max(float(mono.size), 1.0)
    return 10.0 * math.log10(max(float(power), EPS))


def compute_audio_features(
    samples: np.ndarray,
    sample_rate: int,
    band_low_hz: float,
    band_high_hz: float,
    band_weight: float = 0.75,
) -> AudioFeatures:
    rms = rms_db(samples)
    band = band_energy_db(samples, sample_rate, band_low_hz, band_high_hz)
    w = max(0.0, min(float(band_weight), 1.0))
    score = (w * band) + ((1.0 - w) * rms)
    return AudioFeatures(rms_db=rms, band_db=band, score_db=score)

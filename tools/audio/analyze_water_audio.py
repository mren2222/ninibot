#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import wave

import numpy as np


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src_pkg = root / "src" / "audio_controller"
    if src_pkg.exists():
        sys.path.insert(0, str(src_pkg))


_add_src_to_path()

from audio_controller.audio_features import compute_audio_features  # noqa: E402


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(f"{path}: only 16-bit PCM wav is supported, got sample width {sample_width}")

    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sample_rate


def frame_features(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    band_low_hz: float,
    band_high_hz: float,
    band_weight: float,
) -> list[float]:
    frame_samples = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    scores: list[float] = []
    for start in range(0, max(samples.size - frame_samples + 1, 0), frame_samples):
        frame = samples[start:start + frame_samples]
        features = compute_audio_features(
            frame,
            sample_rate,
            band_low_hz,
            band_high_hz,
            band_weight,
        )
        scores.append(features.score_db)
    return scores


def summarize(scores: list[float]) -> dict[str, float]:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return {"median": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze recorded water-flow wav files.")
    parser.add_argument("wav", nargs="+", help="Wav file(s) to analyze.")
    parser.add_argument("--baseline", help="Optional baseline/silent wav used for deltas.")
    parser.add_argument("--frame-ms", type=float, default=50.0, help="Analysis frame size.")
    parser.add_argument("--band-low-hz", type=float, default=800.0)
    parser.add_argument("--band-high-hz", type=float, default=6000.0)
    parser.add_argument("--band-weight", type=float, default=0.75)
    args = parser.parse_args()

    baseline_median = None
    if args.baseline:
        samples, sample_rate = read_wav_mono(Path(args.baseline))
        scores = frame_features(
            samples,
            sample_rate,
            args.frame_ms,
            args.band_low_hz,
            args.band_high_hz,
            args.band_weight,
        )
        baseline_summary = summarize(scores)
        baseline_median = baseline_summary["median"]
        print(f"baseline {args.baseline}: median={baseline_median:+.1f}dB")
        print(
            "suggested initial thresholds: "
            f"start>{baseline_median + 8.0:+.1f}dB "
            f"stop<{baseline_median + 4.0:+.1f}dB"
        )

    for wav_path in args.wav:
        path = Path(wav_path)
        samples, sample_rate = read_wav_mono(path)
        scores = frame_features(
            samples,
            sample_rate,
            args.frame_ms,
            args.band_low_hz,
            args.band_high_hz,
            args.band_weight,
        )
        summary = summarize(scores)
        delta = ""
        if baseline_median is not None:
            delta = f" delta_median={summary['median'] - baseline_median:+.1f}dB"
        duration = samples.size / float(sample_rate)
        print(
            f"{path}: duration={duration:.1f}s "
            f"median={summary['median']:+.1f}dB "
            f"p90={summary['p90']:+.1f}dB "
            f"p95={summary['p95']:+.1f}dB "
            f"max={summary['max']:+.1f}dB"
            f"{delta}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

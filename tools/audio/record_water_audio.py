#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path
import sys
import wave

import numpy as np


def _load_sounddevice():
    try:
        import sounddevice as sd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Python package 'sounddevice' is required. Install it with "
            "'sudo apt install python3-sounddevice' or "
            "'python3 -m pip install sounddevice'."
        ) from exc
    return sd


def _device_arg(value: str | None):
    if value is None or value == "":
        return None
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Record USB microphone audio for water-flow tests.")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz.")
    parser.add_argument("--channels", type=int, default=1, help="Number of input channels.")
    parser.add_argument("--device", default="", help="sounddevice device id/name; empty uses default input.")
    parser.add_argument("--label", default="sample", help="Short label used in the output filename.")
    parser.add_argument(
        "--output-dir",
        default="data/audio/water_flow/raw",
        help="Directory for recorded wav files.",
    )
    args = parser.parse_args()

    sd = _load_sounddevice()
    frames = int(round(args.duration * args.sample_rate))
    device = _device_arg(args.device)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.label)
    output = Path(args.output_dir) / f"{stamp}_{safe_label}.wav"

    print(
        f"Recording {args.duration:.1f}s at {args.sample_rate}Hz "
        f"channels={args.channels} device={device if device is not None else 'default'}"
    )
    audio = sd.rec(
        frames,
        samplerate=args.sample_rate,
        channels=args.channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    write_wav(output, np.asarray(audio, dtype=np.float32), args.sample_rate)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)

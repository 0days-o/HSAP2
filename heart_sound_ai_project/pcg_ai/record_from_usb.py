"""Optional laptop/USB-microphone recorder for quick pipeline tests.

This can be used instead of the ESP32. It records a WAV from the system's default
input device, then the same prediction code can score it after the model is trained.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sounddevice as sd
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--out", type=Path, default=Path("recordings/usb_test.wav"))
    args = parser.parse_args()
    if args.seconds < 1 or args.seconds > 30:
        raise SystemExit("Choose a test duration between 1 and 30 seconds.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("Educational audio test. Use a controlled sound source; do not treat results as a diagnosis.")
    audio = sd.rec(int(args.seconds * args.sample_rate), samplerate=args.sample_rate, channels=1, dtype="float32")
    sd.wait()
    sf.write(str(args.out), audio, args.sample_rate)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()

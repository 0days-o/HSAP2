"""Make objective waveform/signal-level reports for controlled hardware tests.

These metrics characterize the recording chain only; they do not describe health.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import csv
import numpy as np
import soundfile as sf
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x) * 0.5))


def describe_wav(path: Path) -> dict:
    raw, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if raw.size == 0 or sr <= 0:
        raise ValueError("empty or invalid audio")
    x = raw.mean(axis=1)
    x = np.nan_to_num(x)
    duration = len(x) / sr
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    clipping_fraction = float(np.mean(np.abs(x) >= 0.999)) if len(x) else 0.0
    freqs, psd = welch(x, fs=sr, nperseg=min(4096, max(256, len(x))))
    total_power = _integrate_trapezoid(psd, freqs) if len(freqs) > 1 else 0.0
    band = (freqs >= 25) & (freqs <= min(800, sr / 2))
    band_power = _integrate_trapezoid(psd[band], freqs[band]) if np.sum(band) > 1 else 0.0
    band_fraction = band_power / total_power if total_power > 0 else 0.0
    return {
        "file": path.name,
        "sample_rate_hz": int(sr),
        "duration_seconds": round(duration, 4),
        "rms_full_scale": rms,
        "peak_full_scale": peak,
        "clipped_sample_fraction": clipping_fraction,
        "estimated_25_800_hz_power_fraction": float(band_fraction),
    }


def plot_wav(path: Path, out_path: Path) -> None:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = np.nan_to_num(x.mean(axis=1))
    n = min(len(x), 5 * sr)
    t = np.arange(n) / sr
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(t, x[:n], linewidth=0.6)
    axes[0].set_title(f"Waveform — {path.name}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (full scale)")
    axes[1].specgram(x, NFFT=min(512, max(64, len(x))), Fs=sr, noverlap=min(384, max(0, min(512, len(x)) - 1)), cmap="viridis")
    axes[1].set_ylim(0, min(1000, sr / 2))
    axes[1].set_title("Spectrogram (display capped at 1 kHz)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure WAV signal properties for hardware coupling tests.")
    parser.add_argument("wavs", nargs="+", type=Path, help="One or more WAV files")
    parser.add_argument("--out", type=Path, default=Path("signal_reports"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in args.wavs:
        try:
            row = describe_wav(path)
            rows.append(row)
            plot_wav(path, args.out / f"{path.stem}_waveform_spectrogram.png")
            print(row)
        except Exception as exc:
            print(f"WARNING: unable to analyze {path}: {exc}")
    if rows:
        csv_path = args.out / "signal_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved metrics to {csv_path}")
    print("These are recording-chain metrics only, not health or diagnosis metrics.")

if __name__ == "__main__":
    main()

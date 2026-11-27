"""Generate synthetic PCG-like test signals for checking the audio pipeline only.

These are mathematical sound effects, not recordings of people or real hearts.
They must never be used for ML training, medical interpretation, or accuracy claims.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1] / "demo_audio"
SR = 2000
DURATION = 12.0


def pulse(t: np.ndarray, center: float, amplitude: float, frequency: float, decay: float, width: float = 0.12) -> np.ndarray:
    dt = t - center
    active = (dt >= 0) & (dt <= width)
    out = np.zeros_like(t)
    tau = np.maximum(dt[active], 0.0)
    envelope = np.exp(-tau / decay)
    out[active] = amplitude * envelope * np.sin(2 * np.pi * frequency * tau)
    return out


def make_clip(pattern: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * DURATION), dtype=np.float64) / SR
    x = np.zeros_like(t)
    if pattern == "steady_demo":
        beat_times = np.arange(0.5, 11.5, 0.85)
        for b in beat_times:
            x += pulse(t, b, 0.55, 48, 0.040, 0.13)
            x += pulse(t, b + 0.31, 0.38, 62, 0.045, 0.12)
    elif pattern == "variable_timing_demo":
        intervals = [0.72, 0.93, 0.78, 1.02, 0.68, 0.88, 0.76, 0.96, 0.73, 0.91]
        beat_times = [0.45]
        for interval in intervals:
            beat_times.append(beat_times[-1] + interval)
        for b in beat_times:
            if b + 0.5 >= DURATION:
                continue
            x += pulse(t, b, 0.55, 48, 0.040, 0.13)
            x += pulse(t, b + 0.30, 0.36, 62, 0.045, 0.12)
    else:
        raise ValueError(pattern)
    x += rng.normal(0.0, 0.008, size=x.shape)
    x += 0.008 * np.sin(2 * np.pi * 60 * t)
    peak = np.max(np.abs(x))
    if peak:
        x = 0.8 * x / peak
    return x.astype(np.float32)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    clips = {
        "synthetic_steady_timing_demo.wav": make_clip("steady_demo", 17),
        "synthetic_variable_timing_demo.wav": make_clip("variable_timing_demo", 29),
    }
    for name, samples in clips.items():
        sf.write(ROOT / name, samples, SR, subtype="PCM_16")
        print(f"Wrote {ROOT / name}")
    print("Reminder: these are synthetic demonstration signals, not real heart recordings.")


if __name__ == "__main__":
    main()

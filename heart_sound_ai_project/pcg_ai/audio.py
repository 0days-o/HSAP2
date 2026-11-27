"""Audio preprocessing shared by training, prediction, and live capture.

This module is an audio-research pipeline, not a medical signal processor.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt, stft

TARGET_SR = 2_000
SEGMENT_SECONDS = 8
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_SECONDS
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40
F_MIN = 25.0
F_MAX = 800.0
INPUT_SHAPE = (N_MELS, 1 + (SEGMENT_SAMPLES - N_FFT) // HOP_LENGTH, 1)


@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray
    sample_rate: int


def read_audio(path: str | Path, target_sr: int = TARGET_SR) -> AudioClip:
    """Read a WAV file, downmix to mono, resample, and band-limit it."""
    path = Path(path)
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.size == 0 or sr <= 0:
        raise ValueError(f"Audio file is empty or has an invalid sample rate: {path}")
    x = np.mean(data, axis=1, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if sr != target_sr:
        factor = gcd(int(sr), int(target_sr))
        up, down = target_sr // factor, int(sr) // factor
        x = resample_poly(x, up, down).astype(np.float32, copy=False)
        sr = target_sr

    # Keep a broad heart-sound band and reduce slow handling drift/high-frequency hiss.
    # Frequencies are in Hz; the target sample rate is 2 kHz (Nyquist = 1 kHz).
    sos = butter(4, [F_MIN, F_MAX], btype="bandpass", fs=sr, output="sos")
    if len(x) > 3 * (2 * len(sos) + 1):
        x = sosfiltfilt(sos, x).astype(np.float32, copy=False)

    if not np.all(np.isfinite(x)):
        raise ValueError(f"Non-finite samples after preprocessing: {path}")
    return AudioClip(samples=x, sample_rate=int(sr))


def make_segments(x: np.ndarray, max_windows: int = 3) -> list[np.ndarray]:
    """Return up to max_windows evenly spaced, fixed-length segments.

    Short recordings are zero-padded to 8 seconds. Longer recordings are sampled
    across their duration so one long record cannot dominate training.
    """
    if max_windows < 1:
        raise ValueError("max_windows must be >= 1")
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("Cannot create segments from empty audio")
    if x.size <= SEGMENT_SAMPLES:
        out = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
        out[: x.size] = x
        return [out]

    count = min(max_windows, max(1, int(np.ceil(x.size / SEGMENT_SAMPLES))))
    starts = np.unique(np.linspace(0, x.size - SEGMENT_SAMPLES, num=count, dtype=int))
    segments = [x[start : start + SEGMENT_SAMPLES].astype(np.float32, copy=False) for start in starts]
    return segments


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filter_bank(
    sample_rate: int = TARGET_SR,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
) -> np.ndarray:
    """Create a triangular Mel filter bank using only NumPy."""
    mel_points = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    fft_freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    filters = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = hz_points[m : m + 3]
        up = (fft_freqs - left) / max(center - left, 1e-12)
        down = (right - fft_freqs) / max(right - center, 1e-12)
        filters[m] = np.maximum(0.0, np.minimum(up, down))
    return filters


_MEL_BANK = mel_filter_bank()


def segment_to_logmel(segment: np.ndarray) -> np.ndarray:
    """Convert an 8 s segment to a standardized (mel, time, channel) tensor."""
    x = np.asarray(segment, dtype=np.float32).reshape(-1)
    if x.size < SEGMENT_SAMPLES:
        x = np.pad(x, (0, SEGMENT_SAMPLES - x.size))
    elif x.size > SEGMENT_SAMPLES:
        x = x[:SEGMENT_SAMPLES]

    _, _, z = stft(
        x,
        fs=TARGET_SR,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        nfft=N_FFT,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
    )
    power = (np.abs(z) ** 2).astype(np.float32)
    mel_power = _MEL_BANK @ power
    log_mel = 10.0 * np.log10(np.maximum(mel_power, 1e-10))

    # Per-segment standardization reduces dependence on microphone gain.
    mean = float(np.mean(log_mel))
    std = float(np.std(log_mel))
    log_mel = (log_mel - mean) / max(std, 1e-5)
    log_mel = np.clip(log_mel, -5.0, 5.0).astype(np.float32)

    expected = INPUT_SHAPE[:2]
    if log_mel.shape != expected:
        raise RuntimeError(f"Unexpected log-Mel shape {log_mel.shape}; expected {expected}")
    return log_mel[:, :, np.newaxis]


def wav_to_spectrograms(path: str | Path, max_windows: int = 3) -> np.ndarray:
    """Load a WAV and return an array shaped (windows, mel, time, 1)."""
    clip = read_audio(path)
    windows = make_segments(clip.samples, max_windows=max_windows)
    return np.stack([segment_to_logmel(w) for w in windows]).astype(np.float32)

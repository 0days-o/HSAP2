from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf

from pcg_ai.audio import (
    INPUT_SHAPE,
    SEGMENT_SAMPLES,
    make_segments,
    read_audio,
    segment_to_logmel,
)
from pcg_ai.dataset import _parse_reference_line


def test_segment_padding_and_length():
    x = np.ones(1000, dtype=np.float32)
    segments = make_segments(x)
    assert len(segments) == 1
    assert segments[0].shape == (SEGMENT_SAMPLES,)
    assert np.allclose(segments[0][:1000], 1.0)
    assert np.allclose(segments[0][1000:], 0.0)


def test_spectrogram_shape_and_finiteness():
    t = np.arange(SEGMENT_SAMPLES, dtype=np.float32) / 2000.0
    x = 0.2 * np.sin(2 * np.pi * 100 * t) + 0.1 * np.sin(2 * np.pi * 45 * t)
    feature = segment_to_logmel(x)
    assert feature.shape == INPUT_SHAPE
    assert np.all(np.isfinite(feature))


def test_read_audio_resamples_and_downmixes():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.wav"
        sr = 8000
        t = np.arange(sr, dtype=np.float32) / sr
        mono = (0.2 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
        stereo = np.column_stack([mono, mono])
        sf.write(path, stereo, sr)
        clip = read_audio(path)
        assert clip.sample_rate == 2000
        assert len(clip.samples) == 2000
        assert np.all(np.isfinite(clip.samples))


def test_challenge_reference_label_mapping():
    assert _parse_reference_line("a0001 -1") == ("a0001", 0)
    assert _parse_reference_line("a0002,1") == ("a0002", 1)
    assert _parse_reference_line("record label") is None

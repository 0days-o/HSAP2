from pathlib import Path
import numpy as np
import soundfile as sf


def test_bundled_demo_audio_is_loadable_and_marked_synthetic():
    root = Path(__file__).resolve().parents[1] / "demo_audio"
    names = ["synthetic_steady_timing_demo.wav", "synthetic_variable_timing_demo.wav"]
    for name in names:
        audio, sample_rate = sf.read(root / name)
        assert sample_rate == 2000
        assert audio.ndim == 1
        assert len(audio) >= 2000
        assert np.isfinite(audio).all()
    note = (root / "README.md").read_text(encoding="utf-8").lower()
    assert "not real" in note or "not recordings" in note
    assert "must not be used" in note

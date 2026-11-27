"""Command-line prediction and helper functions. Educational use only."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .audio import wav_to_spectrograms


def latest_model_path(experiment_root: str | Path = "experiments") -> Path | None:
    """Return the model saved by the most recently completed training run."""
    root = Path(experiment_root)
    pointer = root / "latest_run.txt"
    if pointer.exists():
        run_id = pointer.read_text(encoding="utf-8").strip()
        candidate = root / run_id / "heart_sound_cnn.keras"
        if candidate.exists():
            return candidate
    legacy = Path("models/heart_sound_cnn.keras")
    return legacy if legacy.exists() else None


def load_keras_model(path: str | Path):
    import tensorflow as tf
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Train it first (see README).")
    return tf.keras.models.load_model(path)


def score_wav(model, wav_path: str | Path, max_windows: int = 3) -> dict:
    tensors = wav_to_spectrograms(wav_path, max_windows=max_windows)
    scores = np.asarray(model.predict(tensors, verbose=0)).reshape(-1)
    score = float(np.mean(scores))
    return {
        "recording": Path(wav_path).name,
        "model_score_for_reference_abnormal_class": score,
        "model_class_at_threshold_0_5": "abnormal-like" if score >= 0.5 else "normal-like",
        "windows_aggregated": int(len(scores)),
        "warning": "Research-demo output only; not a diagnosis or medical recommendation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one WAV for a science-fair audio demo.")
    parser.add_argument("wav", type=Path, help="Input WAV recording")
    parser.add_argument("--model", type=Path, default=None, help="Model file; defaults to the latest completed experiment")
    parser.add_argument("--max-windows", type=int, default=3)
    args = parser.parse_args()

    model_path = args.model or latest_model_path()
    if model_path is None:
        raise FileNotFoundError("No trained model found. Complete menu option 2 first or provide --model.")
    model = load_keras_model(model_path)
    print(f"Using model: {model_path}")
    result = score_wav(model, args.wav, max_windows=args.max_windows)
    print("\nEDUCATIONAL AUDIO-CLASSIFICATION DEMO — NOT FOR MEDICAL USE")
    print(f"Recording: {result['recording']}")
    print(f"Model score for class 1: {result['model_score_for_reference_abnormal_class']:.3f}")
    print(f"Model output at 0.5 threshold: {result['model_class_at_threshold_0_5']}")
    print(f"Windows aggregated: {result['windows_aggregated']}")
    print(result["warning"])


if __name__ == "__main__":
    main()

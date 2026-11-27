"""LAN-only receiver for 8 kHz, signed 16-bit mono PCM from an ESP32.

Run on a Raspberry Pi or laptop on a private local network. Never port-forward it
or expose it to the public internet. Use only controlled educational recordings.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hmac
import os
from pathlib import Path

from flask import Flask, jsonify, request
import numpy as np
import soundfile as sf

from .predict import latest_model_path, load_keras_model, score_wav

SAMPLE_RATE = 8_000
MIN_BYTES = SAMPLE_RATE * 2 * 2  # 2 s minimum, 16-bit samples
MAX_BYTES = SAMPLE_RATE * 2 * 10  # 10 s maximum
RECORDINGS_DIR = Path(os.getenv("PCG_RECORDINGS_DIR", "recordings"))
MODEL_PATH = Path(os.getenv("PCG_MODEL_PATH")) if os.getenv("PCG_MODEL_PATH") else latest_model_path()
UPLOAD_TOKEN = os.getenv("PCG_UPLOAD_TOKEN", "")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES + 1024
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = None
if MODEL_PATH is not None and MODEL_PATH.exists():
    try:
        MODEL = load_keras_model(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as exc:
        print(f"WARNING: could not load model ({exc}). Receiver will save WAVs only.")
else:
    print(f"No trained model found at {MODEL_PATH}; receiver will save WAVs without scoring until trained.")


@app.get("/")
def index():
    return jsonify({
        "service": "Educational heart-sound recording receiver",
        "status": "ready",
        "model_loaded": MODEL is not None,
        "notice": "Not a medical device; do not use for diagnosis or treatment decisions.",
    })


@app.post("/upload")
def upload():
    if UPLOAD_TOKEN:
        provided = request.headers.get("X-Upload-Token", "")
        if not hmac.compare_digest(provided, UPLOAD_TOKEN):
            return jsonify({"error": "Unauthorized upload token"}), 401

    body = request.get_data(cache=False, as_text=False)
    if len(body) < MIN_BYTES or len(body) > MAX_BYTES or len(body) % 2:
        return jsonify({"error": f"Expected 2 to 10 seconds of 8 kHz int16 PCM; received {len(body)} bytes."}), 400

    pcm = np.frombuffer(body, dtype="<i2")
    now = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    wav_path = RECORDINGS_DIR / f"pcg_{now}.wav"
    sf.write(str(wav_path), pcm, SAMPLE_RATE, subtype="PCM_16")

    result = {
        "saved_file": wav_path.name,
        "sample_rate_hz": SAMPLE_RATE,
        "duration_seconds": round(len(pcm) / SAMPLE_RATE, 2),
        "model_loaded": MODEL is not None,
        "warning": "Research/demo output only; not a diagnosis or medical recommendation.",
    }
    if MODEL is not None:
        try:
            result.update(score_wav(MODEL, wav_path))
        except Exception as exc:
            result["scoring_error"] = str(exc)
    return jsonify(result), 200


if __name__ == "__main__":
    # Keep this on your private LAN. The ESP32 needs to reach the host's LAN IP.
    app.run(host="0.0.0.0", port=8000, debug=False)

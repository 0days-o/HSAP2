"""Friendly menu for the educational heart-sound science-fair project."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_module(module: str, *args: str) -> int:
    return subprocess.call([sys.executable, "-m", module, *args], cwd=ROOT)


def pause() -> None:
    input("\nPress Enter to return to the menu...")


def latest_model_path() -> Path | None:
    experiment_root = ROOT / "experiments"
    pointer = experiment_root / "latest_run.txt"
    if pointer.exists():
        run_id = pointer.read_text(encoding="utf-8").strip()
        candidate = experiment_root / run_id / "heart_sound_cnn.keras"
        if candidate.exists():
            return candidate
    # Fallback for older project versions.
    old_model = ROOT / "models" / "heart_sound_cnn.keras"
    return old_model if old_model.exists() else None


def main() -> None:
    print("\n" + "=" * 70)
    print(" HEART-SOUND AI SCIENCE FAIR PROJECT")
    print(" Educational prototype only — NOT a medical diagnostic device")
    print("=" * 70)
    while True:
        print("\nChoose an action:")
        print("  1. Download the full public training dataset (one-time, ~181 MB)")
        print("  2. Train the AI (saves a NEW experiment folder every time)")
        print("  3. Start the ESP32 recording receiver")
        print("  4. Analyze a WAV file with the latest trained model")
        print("  5. Create waveform/spectrogram reports for demo and recorded audio")
        print("  6. Compare all saved training experiments")
        print("  7. Exit")
        choice = input("\nEnter 1-7: ").strip()

        if choice == "1":
            run_module("pcg_ai.download_dataset")
            pause()
        elif choice == "2":
            epochs_text = input("Maximum training epochs [20]: ").strip()
            epochs = epochs_text or "20"
            if not epochs.isdigit() or not (1 <= int(epochs) <= 100):
                print("Please enter a whole number from 1 to 100.")
                pause()
                continue
            seed_text = input("Random seed [42] (use a different number for a repeat experiment): ").strip()
            seed = seed_text or "42"
            if not seed.isdigit():
                print("Please enter a whole-number random seed, such as 42 or 123.")
                pause()
                continue
            run_name = input("Optional experiment label (press Enter to skip): ").strip()
            args = ["--data", "data/physionet2016", "--out", "experiments", "--epochs", epochs, "--seed", seed]
            if run_name:
                args.extend(["--run-name", run_name])
            code = run_module("pcg_ai.train", *args)
            if code == 0:
                print("\nTraining finished. Your run has its own folder under experiments/.")
                print("Start with EXPERIMENT_REPORT.txt and metrics_summary.csv.")
                print("Choose option 6 to compare runs.")
            else:
                print("\nTraining did not finish. Check the messages above and docs/BUILD_GUIDE.md.")
                print("Any files already written remain in that run's folder for troubleshooting.")
            pause()
        elif choice == "3":
            print("Receiver starting on this computer's private network. Press Ctrl+C to stop it.")
            try:
                run_module("pcg_ai.server")
            except KeyboardInterrupt:
                print("\nReceiver stopped.")
            pause()
        elif choice == "4":
            model_path = latest_model_path()
            if model_path is None:
                print("No trained model was found. Complete menu option 2 first.")
            else:
                wav = input("Enter the path to a WAV file: ").strip().strip('"')
                if not wav or not Path(wav).expanduser().exists():
                    print("That file path does not exist.")
                else:
                    print(f"Using model: {model_path.relative_to(ROOT)}")
                    run_module("pcg_ai.predict", wav, "--model", str(model_path.relative_to(ROOT)))
            pause()
        elif choice == "5":
            wavs = sorted((ROOT / "demo_audio").glob("*.wav")) + sorted((ROOT / "recordings").glob("*.wav"))
            if not wavs:
                print("No WAV files found in demo_audio/ or recordings/ yet.")
            else:
                run_module("pcg_ai.report_audio", *[str(p) for p in wavs], "--out", "signal_reports")
                print("Reports are saved in signal_reports/.")
            pause()
        elif choice == "6":
            run_module("pcg_ai.compare_experiments", "--root", "experiments")
            pause()
        elif choice == "7":
            print("Good luck with your science fair project.")
            break
        else:
            print("Please enter a number from 1 to 7.")


if __name__ == "__main__":
    main()
